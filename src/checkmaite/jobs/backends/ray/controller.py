from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypedDict, cast

import ray
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError, TaskCancelledError

from checkmaite.core.analytics_store import AnalyticsStore, Provenance, ProvenanceLike
from checkmaite.core.capability_core import CapabilityRunBase
from checkmaite.jobs._result import build_capability_run_ref
from checkmaite.jobs._store import AnalyticsStoreConfig, build_analytics_store, write_run_and_get_store_uri
from checkmaite.jobs._submission import prepare_job_submission_run_kwargs
from checkmaite.jobs.protocol import CapabilityRunRef, CapabilityRunRefPayload, CapabilityType

from .registry import RegistryStatus

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_UPDATE_TIMEOUT_S = 5.0
DEFAULT_CONTROLLER_NUM_CPUS = 0.01
DEFAULT_CONTROLLER_HEARTBEAT_INTERVAL_S = 10.0
DEFAULT_CONTROLLER_TERMINAL_RETRY_INTERVAL_S = 1.0


@dataclass(frozen=True, slots=True)
class RayTaskResources:
    """Ray worker resources for one capability task.

    ``num_cpus`` and ``num_gpus`` are the standard Ray task options. ``resources``
    holds custom Ray resources, such as accelerator labels or node-affinity keys.
    All quantities are normalized to non-negative floats.
    """

    num_cpus: float = 1.0
    num_gpus: float = 0.0
    resources: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def normalize_quantity(name: str, value: object) -> float:
        """Convert one resource value to a non-negative float."""
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a non-negative numeric resource quantity")
        try:
            quantity = float(cast(Any, value))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a non-negative numeric resource quantity") from exc
        if quantity < 0:
            raise ValueError(f"{name} must be a non-negative numeric resource quantity")
        return quantity

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_cpus", self.normalize_quantity("num_cpus", self.num_cpus))
        object.__setattr__(self, "num_gpus", self.normalize_quantity("num_gpus", self.num_gpus))

        normalized_custom_resources: dict[str, float] = {}
        for key, value in self.resources.items():
            resource_name = str(key)
            normalized_custom_resources[resource_name] = self.normalize_quantity(
                f"resources[{resource_name!r}]",
                value,
            )
        object.__setattr__(self, "resources", normalized_custom_resources)

    @classmethod
    def from_mapping(cls, resources: Mapping[str, object] | RayTaskResources) -> RayTaskResources:
        """Build resources from a user mapping or return an existing instance.

        Custom resources may be nested under ``resources['resources']`` or passed
        as top-level keys other than ``num_cpus`` and ``num_gpus``.
        """
        if isinstance(resources, cls):
            return resources
        if not isinstance(resources, Mapping):
            raise TypeError("resources must be a mapping or RayTaskResources")

        nested_resources = resources.get("resources")
        custom_resources: dict[str, object] = {}
        if nested_resources is not None:
            if not isinstance(nested_resources, Mapping):
                raise TypeError("resources['resources'] must be a mapping of Ray custom resources")
            custom_resources.update({str(key): value for key, value in nested_resources.items()})

        custom_resources.update(
            {str(key): value for key, value in resources.items() if key not in {"num_cpus", "num_gpus", "resources"}}
        )

        return cls(
            num_cpus=cast(Any, resources.get("num_cpus", 1.0)),
            num_gpus=cast(Any, resources.get("num_gpus", 0.0)),
            resources=cast(dict[str, float], custom_resources),
        )

    def as_dict(self) -> dict[str, float | dict[str, float]]:
        """Return keyword options suitable for ``ray.remote``."""
        payload: dict[str, float | dict[str, float]] = {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
        }
        if self.resources:
            payload["resources"] = dict(self.resources)
        return payload


class ControllerStatePayload(TypedDict):
    """Serializable controller state returned to clients/backends."""

    job_id: str
    status: RegistryStatus
    result_ref: CapabilityRunRefPayload | None
    error: str | None
    terminal_at_ts: float | None


def _get_worker_store(store_config: AnalyticsStoreConfig | dict[str, Any]) -> AnalyticsStore:
    return build_analytics_store(store_config)


def _write_run_and_collect_store_metadata(
    store: AnalyticsStore,
    run: CapabilityRunBase[Any, Any],
    *,
    provenance: ProvenanceLike | None = None,
) -> str | None:
    return write_run_and_get_store_uri(store, run, provenance=provenance)


def _update_registry_terminal_best_effort(
    registry: ActorHandle,
    *,
    scope: str,
    job_id: str,
    status: RegistryStatus,
    error: str | None = None,
    result_ref: CapabilityRunRefPayload | None = None,
    controller_actor_name: str | None = None,
    controller_token: str | None = None,
    timeout_s: float = DEFAULT_REGISTRY_UPDATE_TIMEOUT_S,
) -> bool:
    """Write a terminal job state to the registry without blocking forever.

    Controllers call this after the worker task completes, fails, or is
    cancelled. The registry is the shared source of truth used by later clients,
    so this write includes the final status plus either the result reference or
    the error text.

    The call is intentionally best-effort: it uses a short timeout, logs timeout
    or registry failures, and returns ``False`` instead of raising. That lets the
    controller keep its own terminal state and retry later instead of getting
    stuck in a control-plane call.
    """
    try:
        return bool(
            ray.get(
                registry.update_terminal.remote(
                    scope,
                    job_id,
                    status,
                    error,
                    result_ref,
                    controller_actor_name,
                    controller_token,
                ),
                timeout=float(timeout_s),
            )
        )
    except GetTimeoutError:
        logger.warning(
            "Timed out updating registry terminal state",
            extra={"job_id": job_id, "scope": scope, "status": status, "timeout_s": timeout_s},
        )
    except Exception:
        logger.exception(
            "Registry terminal update failed",
            extra={"job_id": job_id, "scope": scope, "status": status},
        )
    return False


def _heartbeat_registry_best_effort(
    registry: ActorHandle,
    *,
    scope: str,
    job_id: str,
    controller_actor_name: str,
    controller_token: str,
    timeout_s: float = DEFAULT_REGISTRY_UPDATE_TIMEOUT_S,
) -> bool:
    """Refresh the controller lease in the registry without blocking forever.

    Live controllers call this periodically while they own a running job. The
    heartbeat tells the registry that the controller actor is still alive, so the
    job should not be treated as stale.

    The call is best-effort for the same reason as terminal updates: registry or
    network delays should not wedge the controller. Failures are swallowed and
    reported as ``False`` so the next heartbeat can try again.
    """
    try:
        return bool(
            ray.get(
                registry.heartbeat_controller.remote(
                    scope,
                    job_id,
                    controller_actor_name,
                    controller_token,
                ),
                timeout=float(timeout_s),
            )
        )
    except Exception:  # noqa: BLE001
        return False


def _execute_capability_ref(capability: CapabilityType, run_kwargs: dict[str, Any]) -> CapabilityRunRef:
    """Execute one capability submission inside a Ray worker process."""
    # TODO: Future work should support a remote/shared cache backend
    # (for example object storage) that workers can read from. At that point,
    # worker execution can safely opt into cache usage.
    run_kwargs = prepare_job_submission_run_kwargs(run_kwargs)

    report_threshold = float(run_kwargs.pop("report_threshold", 0.5))
    raw_store_config = run_kwargs.pop("_analytics_store")
    raw_provenance = run_kwargs.pop("_provenance", None)

    run = capability.run(**run_kwargs)

    store = _get_worker_store(raw_store_config)
    provenance = Provenance.from_optional(raw_provenance).merge({"completed_at": datetime.now(timezone.utc)})
    store_uri = _write_run_and_collect_store_metadata(store, run, provenance=provenance)

    return build_capability_run_ref(
        run,
        store_uri=store_uri,
        report_threshold=report_threshold,
    )


class JobController:
    """Detached actor that manages one submitted Ray job.

    The registry stores job metadata so clients can dedupe and reattach. This
    controller owns the live worker task: it starts it, watches it, handles
    cancellation, sends heartbeats, and writes the final result or error back to
    the registry.
    """

    def __init__(
        self,
        *,
        actor_name: str,
        registry_name: str,
        registry_namespace: str,
        scope: str,
        job_id: str,
        registry_update_timeout_s: float = DEFAULT_REGISTRY_UPDATE_TIMEOUT_S,
        controller_heartbeat_interval_s: float = DEFAULT_CONTROLLER_HEARTBEAT_INTERVAL_S,
        controller_terminal_retry_interval_s: float = DEFAULT_CONTROLLER_TERMINAL_RETRY_INTERVAL_S,
        controller_token: str | None = None,
    ) -> None:
        """Initialize a controller for one reserved registry job.

        Parameters
        ----------
        actor_name
            Name of this detached controller actor. Stored in the registry so
            clients can find the controller again.
        registry_name
            Name of the shared job registry actor.
        registry_namespace
            Ray namespace containing both this controller and the registry.
        scope
            Job backend idempotency scope for the job.
        job_id
            Registry job ID reserved by the submitter.
        registry_update_timeout_s
            Maximum time to wait for each registry heartbeat or state update.
        controller_heartbeat_interval_s
            How often the controller refreshes its registry lease while live.
        controller_terminal_retry_interval_s
            How often to retry writing terminal state if the first write fails.
        controller_token
            Reservation/controller token used to prove this actor owns the job.
        """
        self._actor_name = actor_name
        self._registry_name = registry_name
        self._registry_namespace = registry_namespace
        self._scope = scope
        self._job_id = job_id
        self._registry_update_timeout_s = float(registry_update_timeout_s)
        self._controller_heartbeat_interval_s = float(controller_heartbeat_interval_s)
        self._controller_terminal_retry_interval_s = float(controller_terminal_retry_interval_s)
        self._controller_token = controller_token

        self._lock = threading.RLock()
        self._obj_ref: ray.ObjectRef[CapabilityRunRef] | None = None
        self._status = RegistryStatus.SUBMITTING
        self._result_ref: CapabilityRunRefPayload | None = None
        self._error: str | None = None
        self._terminal_at_ts: float | None = None
        self._watcher_started = False
        self._heartbeat_started = False
        self._heartbeat_stop = threading.Event()
        self._terminal_committed = False
        self._terminal_retry_started = False

    def _registry(self) -> ActorHandle | None:
        try:
            return ray.get_actor(self._registry_name, namespace=self._registry_namespace)
        except Exception:  # noqa: BLE001
            return None

    def _state_locked(self) -> ControllerStatePayload:
        """Return the controller's current state while ``self._lock`` is held."""
        return {
            "job_id": self._job_id,
            "status": self._status,
            "result_ref": self._result_ref,
            "error": self._error,
            "terminal_at_ts": self._terminal_at_ts,
        }

    @staticmethod
    def _is_terminal(status: RegistryStatus) -> bool:
        return status in {
            RegistryStatus.COMPLETED,
            RegistryStatus.FAILED,
            RegistryStatus.CANCELLED,
        }

    def _controller_token_matches(self, controller_token: str | None) -> bool:
        with self._lock:
            expected_token = self._controller_token
        return expected_token is not None and controller_token == expected_token

    def _push_terminal_best_effort(self) -> bool:
        """Try to write this controller's terminal state to the registry."""
        registry = self._registry()
        if registry is None:
            return False
        return _update_registry_terminal_best_effort(
            registry,
            scope=self._scope,
            job_id=self._job_id,
            status=self._status,
            error=self._error,
            result_ref=self._result_ref,
            controller_actor_name=self._actor_name,
            controller_token=self._controller_token,
            timeout_s=self._registry_update_timeout_s,
        )

    def _heartbeat_loop(self) -> None:
        """Periodically refresh the registry lease until heartbeats are stopped."""
        while True:
            with self._lock:
                token = self._controller_token
            if token is not None:
                registry = self._registry()
                if registry is not None:
                    _heartbeat_registry_best_effort(
                        registry,
                        scope=self._scope,
                        job_id=self._job_id,
                        controller_actor_name=self._actor_name,
                        controller_token=token,
                        timeout_s=self._registry_update_timeout_s,
                    )
            if self._heartbeat_stop.wait(self._controller_heartbeat_interval_s):
                return

    def _start_heartbeat_locked(self) -> None:
        """Start the heartbeat thread once while ``self._lock`` is held."""
        if self._heartbeat_started:
            return
        self._heartbeat_started = True
        thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        thread.start()

    @staticmethod
    def _jittered_retry_delay_s(delay_s: float) -> float:
        """Spread retry attempts by +/-25% to avoid synchronized controller herds."""
        jitter_per_thousand = 750 + secrets.randbelow(501)
        return max(0.01, delay_s * jitter_per_thousand / 1000.0)

    def _terminal_retry_loop(self) -> None:
        """Retry failed terminal registry writes indefinitely with jittered exponential backoff."""
        delay_s = max(0.01, self._controller_terminal_retry_interval_s)
        while True:
            time.sleep(self._jittered_retry_delay_s(delay_s))
            with self._lock:
                if self._terminal_committed or not self._is_terminal(self._status):
                    return
            if self._push_terminal_best_effort():
                with self._lock:
                    self._terminal_committed = True
                self._heartbeat_stop.set()
                return
            delay_s = min(10.0, delay_s * 2.0)

    def _start_terminal_retry_locked(self) -> None:
        """Start the terminal-write retry thread once while ``self._lock`` is held."""
        if self._terminal_retry_started or self._terminal_committed:
            return
        self._terminal_retry_started = True
        thread = threading.Thread(target=self._terminal_retry_loop, daemon=True)
        thread.start()

    def _set_terminal(
        self,
        status: RegistryStatus,
        *,
        error: str | None = None,
        result_ref: CapabilityRunRefPayload | None = None,
    ) -> ControllerStatePayload:
        """Move the controller to a terminal state and publish it to the registry."""
        with self._lock:
            if self._is_terminal(self._status):
                state = self._state_locked()
            else:
                if status == RegistryStatus.COMPLETED and result_ref is None:
                    status = RegistryStatus.FAILED
                    error = error or "completed job missing result_ref"
                self._status = status
                self._error = error
                self._result_ref = result_ref
                self._terminal_at_ts = time.time()
                self._obj_ref = None
                state = self._state_locked()

        if self._push_terminal_best_effort():
            with self._lock:
                self._terminal_committed = True
            self._heartbeat_stop.set()
        else:
            with self._lock:
                self._start_terminal_retry_locked()
        return state

    def _watch_object_ref(self, obj_ref: ray.ObjectRef[CapabilityRunRef]) -> None:
        """Wait for the worker task result and record its terminal state."""
        try:
            ref = ray.get(obj_ref)
            if isinstance(ref, CapabilityRunRef):
                result_ref = cast(CapabilityRunRefPayload, ref.model_dump(mode="json"))
            else:
                result_ref = cast(CapabilityRunRefPayload, CapabilityRunRef.model_validate(ref).model_dump(mode="json"))
            self._set_terminal(RegistryStatus.COMPLETED, result_ref=result_ref)
        except TaskCancelledError:
            self._set_terminal(RegistryStatus.CANCELLED)
        except Exception as exc:  # noqa: BLE001
            self._set_terminal(RegistryStatus.FAILED, error=str(exc))

    def _start_watcher_locked(self, obj_ref: ray.ObjectRef[CapabilityRunRef]) -> None:
        """Start the worker-result watcher thread once while ``self._lock`` is held."""
        if self._watcher_started:
            return
        self._watcher_started = True
        thread = threading.Thread(target=self._watch_object_ref, args=(obj_ref,), daemon=True)
        thread.start()

    def start(
        self,
        capability: CapabilityType,
        run_kwargs: dict[str, Any],
        resources: RayTaskResources | Mapping[str, object],
        max_retries: int,
        reservation_token: str,
    ) -> ControllerStatePayload:
        """Start the Ray worker task for this job.

        The submitter calls this after it has reserved a job in the registry and
        created this controller actor. The controller first proves ownership with
        the reservation token and asks the registry to move the job to
        ``RUNNING``. If the reservation is no longer valid, no worker is started.

        Once the registry accepts the start, the controller builds the Ray remote
        options from ``resources``, launches the worker task, stores the task
        ``ObjectRef``, starts heartbeats, and starts a watcher thread that will
        publish the final result or error. Calling ``start`` again is safe: if
        work is already running or the job is terminal, the current state is
        returned instead of launching a second task.
        """
        started_running = False
        with self._lock:
            if self._controller_token is None:
                self._controller_token = reservation_token
            elif self._controller_token != reservation_token:
                raise ValueError(f"Invalid controller token for job {self._job_id!r}")
            if self._obj_ref is not None or self._is_terminal(self._status):
                return self._state_locked()

        try:
            registry = self._registry()
            if registry is None:
                raise RuntimeError("job registry unavailable")
            can_start = bool(
                ray.get(
                    registry.mark_running.remote(
                        self._scope,
                        self._job_id,
                        reservation_token,
                        self._actor_name,
                        self._registry_namespace,
                    ),
                    timeout=self._registry_update_timeout_s,
                )
            )
            if not can_start:
                with self._lock:
                    return self._state_locked()

            with self._lock:
                self._status = RegistryStatus.RUNNING
                self._start_heartbeat_locked()
            started_running = True

            resolved_resources = RayTaskResources.from_mapping(resources)
            remote_options: dict[str, Any] = {
                "num_gpus": resolved_resources.num_gpus,
                "num_cpus": resolved_resources.num_cpus,
                "max_retries": max_retries,
            }
            if resolved_resources.resources:
                remote_options["resources"] = dict(resolved_resources.resources)
            remote_fn = ray.remote(**remote_options)(_execute_capability_ref)

            obj_ref = cast(ray.ObjectRef[CapabilityRunRef], remote_fn.remote(capability, dict(run_kwargs)))
            with self._lock:
                if self._is_terminal(self._status):
                    with suppress(Exception):
                        ray.cancel(obj_ref)
                    return self._state_locked()
                self._obj_ref = obj_ref
                self._status = RegistryStatus.RUNNING
                self._start_heartbeat_locked()
                self._start_watcher_locked(obj_ref)
                return self._state_locked()
        except Exception as exc:
            state = self._set_terminal(RegistryStatus.FAILED, error=str(exc))
            if started_running:
                return state
            raise
        finally:
            # Drop local references before this actor method returns; submitted
            # objects are not stored on controller state after launch/failure.
            del capability, run_kwargs

    def reconcile(self) -> ControllerStatePayload:
        """Refresh this controller's state from the worker task.

        This is a non-blocking check. If the worker task is still running, the
        current controller state is returned. If the task is ready, the result is
        read and the controller moves to ``COMPLETED``, ``FAILED``, or
        ``CANCELLED`` and tries to publish that terminal state to the registry.
        """
        with self._lock:
            if self._is_terminal(self._status):
                state = self._state_locked()
                obj_ref: ray.ObjectRef[CapabilityRunRef] | None = None
            else:
                state = None
                obj_ref = self._obj_ref
                if obj_ref is None:
                    return self._state_locked()

        if state is not None:
            # Terminal state is already in controller memory. Return it quickly;
            # callers that need shared truth are responsible for a bounded
            # registry commit before treating it as authoritative.
            return state

        if obj_ref is None:
            with self._lock:
                return self._state_locked()
        ready, _ = ray.wait([obj_ref], timeout=0)
        if not ready:
            with self._lock:
                return self._state_locked()

        try:
            ref = ray.get(obj_ref, timeout=0)
            if isinstance(ref, CapabilityRunRef):
                result_ref = cast(CapabilityRunRefPayload, ref.model_dump(mode="json"))
            else:
                result_ref = cast(CapabilityRunRefPayload, CapabilityRunRef.model_validate(ref).model_dump(mode="json"))
            return self._set_terminal(RegistryStatus.COMPLETED, result_ref=result_ref)
        except GetTimeoutError:
            with self._lock:
                return self._state_locked()
        except TaskCancelledError:
            return self._set_terminal(RegistryStatus.CANCELLED)
        except Exception as exc:  # noqa: BLE001
            return self._set_terminal(RegistryStatus.FAILED, error=str(exc))

    def cancel(self, controller_token: str | None = None) -> bool:
        """Cancel this job if it is still running.

        Terminal jobs cannot be cancelled. If no worker task has been launched
        yet, the controller marks the job ``CANCELLED`` directly. Otherwise it
        records the cancellation request in the registry, asks Ray to cancel the
        worker task, and reports ``True`` when that request was accepted. The
        caller must present the controller owner token so arbitrary Ray clients
        cannot cancel a named detached controller by accident.
        """
        with self._lock:
            if not self._controller_token_matches(controller_token):
                return False
            if self._is_terminal(self._status):
                return False
            obj_ref = self._obj_ref
            token = self._controller_token

        if obj_ref is None:
            self._set_terminal(RegistryStatus.CANCELLED)
            return True

        registry = self._registry()
        if registry is not None and token is not None:
            with suppress(Exception):
                ray.get(
                    registry.request_cancellation.remote(
                        self._scope,
                        self._job_id,
                        self._actor_name,
                        token,
                    ),
                    timeout=self._registry_update_timeout_s,
                )

        ready, _ = ray.wait([obj_ref], timeout=0)
        if ready:
            self.reconcile()
            return False

        try:
            ray.cancel(obj_ref, force=True)
        except Exception:  # noqa: BLE001
            return False

        with self._lock:
            if not self._is_terminal(self._status):
                self._status = RegistryStatus.CANCELLING
        return True

    def get_state(self, controller_token: str | None = None, reconcile: bool = True) -> ControllerStatePayload:
        if not self._controller_token_matches(controller_token):
            raise PermissionError(f"Invalid controller token for job {self._job_id!r}")
        if reconcile:
            return self.reconcile()
        with self._lock:
            return self._state_locked()


JobControllerActor = ray.remote(max_restarts=0)(JobController)


def get_or_create_controller_actor(
    *,
    name: str,
    namespace: str,
    registry_name: str,
    registry_namespace: str,
    scope: str,
    job_id: str,
    registry_update_timeout_s: float,
    controller_heartbeat_interval_s: float,
    controller_terminal_retry_interval_s: float,
    controller_num_cpus: float,
    controller_memory: float | None,
    controller_resources: dict[str, float] | None,
    controller_max_pending_calls: int | None = None,
    controller_token: str | None = None,
) -> ActorHandle:
    """Get existing detached per-job controller actor or create it if absent."""
    try:
        return ray.get_actor(name, namespace=namespace)
    except ValueError:
        pass

    options: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
        "num_cpus": float(controller_num_cpus),
    }
    if controller_memory is not None:
        options["memory"] = float(controller_memory)
    if controller_resources is not None:
        options["resources"] = controller_resources
    if controller_max_pending_calls is not None:
        options["max_pending_calls"] = int(controller_max_pending_calls)

    try:
        return cast(
            ActorHandle,
            JobControllerActor.options(**options).remote(
                actor_name=name,
                registry_name=registry_name,
                registry_namespace=registry_namespace,
                scope=scope,
                job_id=job_id,
                registry_update_timeout_s=registry_update_timeout_s,
                controller_heartbeat_interval_s=controller_heartbeat_interval_s,
                controller_terminal_retry_interval_s=controller_terminal_retry_interval_s,
                controller_token=controller_token,
            ),
        )
    except ValueError:
        return ray.get_actor(name, namespace=namespace)
