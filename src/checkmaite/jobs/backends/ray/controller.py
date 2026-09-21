from __future__ import annotations

import enum
import logging
import secrets
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Protocol, SupportsFloat, SupportsIndex, TypedDict, cast

import ray
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError, TaskCancelledError
from typing_extensions import NotRequired

from checkmaite.jobs.protocol import BackpressureError, CapabilityRunRef, CapabilityRunRefPayload, CapabilityType

from .registry import (
    DEFAULT_CONTROLLER_RETENTION_S,
    CancellationRequestResult,
    HeartbeatResult,
    RegistryStatus,
    WorkerStartResult,
)

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_UPDATE_TIMEOUT_S = 5.0
DEFAULT_CONTROLLER_NUM_CPUS = 0.01
DEFAULT_CONTROLLER_STARTUP_TIMEOUT_S = 30.0
DEFAULT_SCHEDULING_TIMEOUT_S = 30 * 60.0
CONTROLLER_COMPATIBILITY_VERSION = 1
DEFAULT_CONTROLLER_HEARTBEAT_INTERVAL_S = 10.0
DEFAULT_CONTROLLER_TERMINAL_RETRY_INTERVAL_S = 1.0


class WorkerStartupUnavailableError(RuntimeError):
    """Signal a retryable worker startup handshake failure."""


@dataclass(frozen=True, slots=True)
class RayTaskResources:
    """Ray worker resources for one capability task.

    ``num_cpus`` and ``num_gpus`` are the standard Ray task options. ``resources``
    holds custom Ray resources, such as accelerator labels or node-affinity keys.
    All quantities are normalized to non-negative floats.
    """

    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory: float | None = None
    resources: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def normalize_quantity(name: str, value: object) -> float:
        """Convert one resource value to a non-negative float."""
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a non-negative numeric resource quantity")
        try:
            if isinstance(value, (str, bytes, bytearray, SupportsFloat)):
                quantity = float(value)
            elif isinstance(value, SupportsIndex):
                quantity = float(value.__index__())
            else:
                raise TypeError
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a non-negative numeric resource quantity") from exc
        if quantity < 0:
            raise ValueError(f"{name} must be a non-negative numeric resource quantity")
        return quantity

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_cpus", self.normalize_quantity("num_cpus", self.num_cpus))
        object.__setattr__(self, "num_gpus", self.normalize_quantity("num_gpus", self.num_gpus))
        if self.memory is not None:
            object.__setattr__(self, "memory", self.normalize_quantity("memory", self.memory))

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
            {
                str(key): value
                for key, value in resources.items()
                if key not in {"num_cpus", "num_gpus", "memory", "resources"}
            }
        )

        normalized_custom_resources = {
            name: cls.normalize_quantity(f"resources[{name!r}]", value) for name, value in custom_resources.items()
        }
        memory = resources.get("memory")
        return cls(
            num_cpus=cls.normalize_quantity("num_cpus", resources.get("num_cpus", 1.0)),
            num_gpus=cls.normalize_quantity("num_gpus", resources.get("num_gpus", 0.0)),
            memory=None if memory is None else cls.normalize_quantity("memory", memory),
            resources=normalized_custom_resources,
        )

    def as_dict(self) -> dict[str, float | dict[str, float]]:
        """Return normalized resource metadata suitable for persistence."""
        payload: dict[str, float | dict[str, float]] = {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
        }
        if self.memory is not None:
            payload["memory"] = self.memory
        if self.resources:
            payload["resources"] = dict(self.resources)
        return payload


class RayWorkerOptions(TypedDict):
    """Typed Ray options for one capability worker task."""

    num_cpus: float
    num_gpus: float
    max_retries: int
    retry_exceptions: list[type[BaseException]]
    memory: NotRequired[float]
    resources: dict[str, float]


class CapabilityWorkerRemote(Protocol):
    """Typed interface for the dynamically decorated Ray worker function."""

    def remote(
        self,
        capability: CapabilityType,
        run_kwargs: dict[str, Any],
        *,
        controller: ActorHandle,
        controller_token: str,
        startup_timeout_s: float,
    ) -> ray.ObjectRef[CapabilityRunRef]: ...


class ControllerConfiguration(TypedDict):
    """Immutable behavior settings for one controller actor."""

    registry_update_timeout_s: float
    controller_heartbeat_interval_s: float
    controller_terminal_retry_interval_s: float
    scheduling_timeout_s: float | None
    controller_retention_s: float


class ControllerDescriptor(TypedDict):
    """Compatibility and identity payload returned by a controller actor."""

    compatibility_version: int
    actor_name: str
    registry_name: str
    registry_namespace: str
    scope: str
    job_id: str
    configuration: ControllerConfiguration


class ControllerCompatibilityError(RuntimeError):
    """Raised when an existing controller has incompatible identity or settings."""


class ControllerStartupError(RuntimeError):
    """Raised when a controller does not become ready within its startup budget."""


def _controller_configuration(
    *,
    registry_update_timeout_s: float,
    controller_heartbeat_interval_s: float,
    controller_terminal_retry_interval_s: float,
    scheduling_timeout_s: float | None,
    controller_retention_s: float,
) -> ControllerConfiguration:
    return {
        "registry_update_timeout_s": float(registry_update_timeout_s),
        "controller_heartbeat_interval_s": float(controller_heartbeat_interval_s),
        "controller_terminal_retry_interval_s": float(controller_terminal_retry_interval_s),
        "scheduling_timeout_s": None if scheduling_timeout_s is None else float(scheduling_timeout_s),
        "controller_retention_s": float(controller_retention_s),
    }


def _controller_descriptor(
    *,
    actor_name: str,
    registry_name: str,
    registry_namespace: str,
    scope: str,
    job_id: str,
    configuration: ControllerConfiguration,
) -> ControllerDescriptor:
    return {
        "compatibility_version": CONTROLLER_COMPATIBILITY_VERSION,
        "actor_name": actor_name,
        "registry_name": registry_name,
        "registry_namespace": registry_namespace,
        "scope": scope,
        "job_id": job_id,
        "configuration": configuration.copy(),
    }


class ControllerStatePayload(TypedDict):
    """Serializable controller state returned to clients/backends."""

    job_id: str
    status: RegistryStatus
    result_ref: CapabilityRunRefPayload | None
    error: str | None
    terminal_at_ts: float | None
    terminal_authoritative: NotRequired[bool]


class _RegistryCallOutcome(enum.Enum):
    ACCEPTED = enum.auto()
    REJECTED = enum.auto()
    UNAVAILABLE = enum.auto()


@dataclass(frozen=True, slots=True)
class _RegistryCallResult:
    outcome: _RegistryCallOutcome
    status: RegistryStatus | None = None


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
) -> _RegistryCallResult:
    """Return an authoritative status, rejection, or unavailable outcome."""
    try:
        committed_status = ray.get(
            registry.commit_controller_terminal.remote(
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
        if committed_status is None:
            return _RegistryCallResult(_RegistryCallOutcome.REJECTED)
        return _RegistryCallResult(_RegistryCallOutcome.ACCEPTED, RegistryStatus(committed_status))
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
    return _RegistryCallResult(_RegistryCallOutcome.UNAVAILABLE)


def _heartbeat_registry_best_effort(
    registry: ActorHandle,
    *,
    scope: str,
    job_id: str,
    controller_actor_name: str,
    controller_token: str,
    timeout_s: float = DEFAULT_REGISTRY_UPDATE_TIMEOUT_S,
) -> _RegistryCallResult:
    """Refresh the controller lease and distinguish rejection from unavailability."""
    try:
        accepted, status = cast(
            HeartbeatResult,
            ray.get(
                registry.heartbeat_controller.remote(
                    scope,
                    job_id,
                    controller_actor_name,
                    controller_token,
                ),
                timeout=float(timeout_s),
            ),
        )
    except Exception:  # noqa: BLE001
        return _RegistryCallResult(_RegistryCallOutcome.UNAVAILABLE)
    outcome = _RegistryCallOutcome.ACCEPTED if accepted else _RegistryCallOutcome.REJECTED
    return _RegistryCallResult(outcome, RegistryStatus(status) if status is not None else None)


def _execute_capability_ref(
    capability: CapabilityType,
    run_kwargs: dict[str, Any],
    controller: ActorHandle | None = None,
    controller_token: str | None = None,
    startup_timeout_s: float = DEFAULT_REGISTRY_UPDATE_TIMEOUT_S,
) -> CapabilityRunRef:
    """Load worker-only dependencies lazily and execute one capability."""
    from .worker import execute_capability_ref

    return execute_capability_ref(
        capability,
        run_kwargs,
        controller=controller,
        controller_token=controller_token,
        startup_timeout_s=startup_timeout_s,
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
        scheduling_timeout_s: float | None = DEFAULT_SCHEDULING_TIMEOUT_S,
        controller_retention_s: float = DEFAULT_CONTROLLER_RETENTION_S,
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
        scheduling_timeout_s
            Maximum time the worker may wait for Ray resources before failure.
        controller_retention_s
            Delay before this terminal controller asks the registry to clean it up.
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
        self._scheduling_timeout_s = None if scheduling_timeout_s is None else float(scheduling_timeout_s)
        self._controller_retention_s = float(controller_retention_s)
        self._controller_token = controller_token

        self._lock = threading.RLock()
        self._registry_actor: ActorHandle | None = None
        self._registry_lookup_inflight: threading.Event | None = None
        self._obj_ref: ray.ObjectRef[CapabilityRunRef] | None = None
        self._status = RegistryStatus.SUBMITTING
        self._result_ref: CapabilityRunRefPayload | None = None
        self._error: str | None = None
        self._terminal_at_ts: float | None = None
        self._watcher_started = False
        self._heartbeat_started = False
        self._heartbeat_stop = threading.Event()
        self._terminal_committed = False
        self._terminal_authoritative = False
        self._terminal_retry_started = False
        self._registry_orphaned = False
        self._orphaned_obj_ref: ray.ObjectRef[CapabilityRunRef] | None = None
        self._worker_cancellation_retry_started = False
        self._worker_started = threading.Event()
        self._scheduling_timeout_started = False
        self._retirement_started = False

    def describe(self, controller_token: str | None = None) -> ControllerDescriptor:
        """Return bounded protocol and immutable controller configuration metadata."""
        if self._controller_token is not None and not self._controller_token_matches(controller_token):
            raise PermissionError(f"Invalid controller token for job {self._job_id!r}")
        return _controller_descriptor(
            actor_name=self._actor_name,
            registry_name=self._registry_name,
            registry_namespace=self._registry_namespace,
            scope=self._scope,
            job_id=self._job_id,
            configuration=_controller_configuration(
                registry_update_timeout_s=self._registry_update_timeout_s,
                controller_heartbeat_interval_s=self._controller_heartbeat_interval_s,
                controller_terminal_retry_interval_s=self._controller_terminal_retry_interval_s,
                scheduling_timeout_s=self._scheduling_timeout_s,
                controller_retention_s=self._controller_retention_s,
            ),
        )

    def _registry(self) -> ActorHandle | None:
        """Return a cached registry handle, bounding cache-miss discovery time."""
        with self._lock:
            if self._registry_actor is not None:
                return self._registry_actor
            lookup_done = self._registry_lookup_inflight
            start_lookup = lookup_done is None
            if lookup_done is None:
                lookup_done = threading.Event()
                self._registry_lookup_inflight = lookup_done

        if start_lookup:

            def lookup() -> None:
                registry: ActorHandle | None = None
                with suppress(Exception):
                    registry = ray.get_actor(self._registry_name, namespace=self._registry_namespace)
                with self._lock:
                    if registry is not None:
                        self._registry_actor = registry
                    if self._registry_lookup_inflight is lookup_done:
                        self._registry_lookup_inflight = None
                    lookup_done.set()

            threading.Thread(target=lookup, daemon=True).start()

        if not lookup_done.wait(timeout=self._registry_update_timeout_s):
            return None
        with self._lock:
            return self._registry_actor

    def _invalidate_registry(self, registry: ActorHandle) -> None:
        """Forget a failed cached handle so later background work may reconnect."""
        with self._lock:
            if self._registry_actor == registry:
                self._registry_actor = None

    def _state_locked(self) -> ControllerStatePayload:
        """Return the controller's current state while ``self._lock`` is held."""
        state: ControllerStatePayload = {
            "job_id": self._job_id,
            "status": self._status,
            "result_ref": self._result_ref,
            "error": self._error,
            "terminal_at_ts": self._terminal_at_ts,
        }
        if self._is_terminal(self._status):
            state["terminal_authoritative"] = self._terminal_authoritative
        return state

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

    def _push_terminal_best_effort(self) -> _RegistryCallResult:
        """Try to write this controller's terminal state to the registry."""
        registry = self._registry()
        if registry is None:
            return _RegistryCallResult(_RegistryCallOutcome.UNAVAILABLE)
        result = _update_registry_terminal_best_effort(
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
        if result.outcome is _RegistryCallOutcome.UNAVAILABLE:
            self._invalidate_registry(registry)
        return result

    def _heartbeat_loop(self) -> None:
        """Refresh the registry lease until accepted work stops or loses ownership."""
        while True:
            with self._lock:
                token = self._controller_token
            if token is not None:
                registry = self._registry()
                if registry is not None:
                    result = _heartbeat_registry_best_effort(
                        registry,
                        scope=self._scope,
                        job_id=self._job_id,
                        controller_actor_name=self._actor_name,
                        controller_token=token,
                        timeout_s=self._registry_update_timeout_s,
                    )
                    if result.outcome is _RegistryCallOutcome.REJECTED:
                        self._set_orphaned_live_terminal(cancel_worker=True)
                        return
                    if result.outcome is _RegistryCallOutcome.UNAVAILABLE:
                        self._invalidate_registry(registry)
                    elif result.status is RegistryStatus.CANCELLING:
                        self._accept_cancellation_intent()
            if self._heartbeat_stop.wait(self._controller_heartbeat_interval_s):
                return

    def _start_heartbeat_locked(self) -> None:
        """Start the heartbeat thread once while ``self._lock`` is held."""
        if self._heartbeat_started:
            return
        self._heartbeat_started = True
        thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        thread.start()

    def _resolve_scheduling_timeout(
        self,
        obj_ref: ray.ObjectRef[CapabilityRunRef],
    ) -> RegistryStatus | None:
        """Ask the registry once to order scheduling timeout against cancellation."""
        with self._lock:
            if self._obj_ref != obj_ref or self._status not in {
                RegistryStatus.SCHEDULING,
                RegistryStatus.CANCELLING,
            }:
                return self._status
            token = self._controller_token
        if token is None:
            return None

        deadline = time.monotonic() + self._registry_update_timeout_s
        retry_s = 0.1
        registry = self._registry()
        while registry is None:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                logger.error(
                    "Job registry remained unavailable while resolving scheduling timeout",
                    extra={"job_id": self._job_id, "scope": self._scope},
                )
                return None
            if self._heartbeat_stop.wait(min(remaining_s, self._jittered_retry_delay_s(retry_s))):
                return None
            retry_s = min(1.0, retry_s * 2.0)
            registry = self._registry()

        resolution_ref: ray.ObjectRef[RegistryStatus | None] | None = None
        try:
            resolution_ref = cast(
                ray.ObjectRef[RegistryStatus | None],
                registry.resolve_scheduling_timeout.remote(
                    self._scope,
                    self._job_id,
                    self._actor_name,
                    token,
                ),
            )
            outcome = ray.get(
                cast(ray.ObjectRef[RegistryStatus | None], resolution_ref),
                timeout=max(0.0, deadline - time.monotonic()),
            )
            return None if outcome is None else RegistryStatus(outcome)
        except GetTimeoutError:
            if resolution_ref is not None:
                with suppress(Exception):
                    ray.cancel(resolution_ref)
            logger.error(
                "Timed out resolving scheduling timeout against registry state",
                extra={"job_id": self._job_id, "scope": self._scope},
            )
        except Exception:
            logger.exception(
                "Could not resolve scheduling timeout against registry state",
                extra={"job_id": self._job_id, "scope": self._scope},
            )
        self._invalidate_registry(registry)
        return None

    def _cancel_worker_with_retry(
        self,
        obj_ref: ray.ObjectRef[CapabilityRunRef],
        *,
        failure_message: str,
    ) -> None:
        """Dispatch worker cancellation and retain the ref until dispatch succeeds."""
        with self._lock:
            self._orphaned_obj_ref = obj_ref
        try:
            ray.cancel(obj_ref, force=True)
        except Exception:  # noqa: BLE001
            logger.warning(failure_message, extra={"job_id": self._job_id}, exc_info=True)
            with self._lock:
                self._start_worker_cancellation_retry_locked()
        else:
            with self._lock:
                if self._orphaned_obj_ref == obj_ref:
                    self._orphaned_obj_ref = None

    def _accept_cancellation_intent(self) -> bool:
        """Mirror authoritative registry cancellation and stop live worker work."""
        with self._lock:
            if self._is_terminal(self._status):
                return False
            self._status = RegistryStatus.CANCELLING
            obj_ref = self._obj_ref
        if obj_ref is not None:
            self._cancel_worker_with_retry(obj_ref, failure_message="Could not cancel worker")
        return True

    def _set_orphaned_live_terminal(
        self,
        *,
        expected_obj_ref: ray.ObjectRef[CapabilityRunRef] | None = None,
        cancel_worker: bool = False,
    ) -> bool:
        """Stop controller-owned live work after registry state or ownership is lost."""
        with self._lock:
            if expected_obj_ref is not None and self._obj_ref != expected_obj_ref:
                return False
            obj_ref = self._obj_ref
            if self._status is RegistryStatus.CANCELLING:
                self._set_terminal_locked(RegistryStatus.CANCELLED)
            elif self._status in {RegistryStatus.SCHEDULING, RegistryStatus.RUNNING}:
                self._set_terminal_locked(
                    RegistryStatus.FAILED,
                    error="job registry state or controller ownership was unavailable",
                )
            else:
                return False
            self._registry_orphaned = True
            self._terminal_committed = True
        self._heartbeat_stop.set()
        if cancel_worker and obj_ref is not None:
            self._cancel_worker_with_retry(obj_ref, failure_message="Could not cancel orphaned worker")
        with self._lock:
            self._start_retirement_locked()
        return True

    def _set_terminal_and_cancel_worker(
        self,
        status: RegistryStatus,
        obj_ref: ray.ObjectRef[CapabilityRunRef],
        *,
        error: str | None = None,
    ) -> None:
        """Make a worker terminal and secure cancellation before retirement can start."""
        with self._lock:
            if self._obj_ref != obj_ref:
                return
            self._set_terminal_locked(status, error=error)
        self._heartbeat_stop.set()
        self._cancel_worker_with_retry(obj_ref, failure_message="Could not cancel terminal worker")
        self._publish_terminal()

    def _scheduling_timeout_loop(self, obj_ref: ray.ObjectRef[CapabilityRunRef]) -> None:
        timeout_s = self._scheduling_timeout_s
        if timeout_s is None or self._worker_started.wait(timeout_s):
            return

        outcome = self._resolve_scheduling_timeout(obj_ref)
        if outcome is RegistryStatus.FAILED:
            self._set_terminal_and_cancel_worker(
                RegistryStatus.FAILED,
                obj_ref,
                error=f"worker was not scheduled within {timeout_s:.3f}s",
            )
        elif outcome in {RegistryStatus.CANCELLING, RegistryStatus.CANCELLED}:
            self._set_terminal_and_cancel_worker(RegistryStatus.CANCELLED, obj_ref)
        elif outcome is None:
            self._set_orphaned_live_terminal(expected_obj_ref=obj_ref, cancel_worker=True)

    def _start_scheduling_timeout_locked(self, obj_ref: ray.ObjectRef[CapabilityRunRef]) -> None:
        if self._scheduling_timeout_s is None or self._scheduling_timeout_started:
            return
        self._scheduling_timeout_started = True
        thread = threading.Thread(target=self._scheduling_timeout_loop, args=(obj_ref,), daemon=True)
        thread.start()

    def _worker_cancellation_retry_loop(self) -> None:
        retry_s = max(0.1, self._controller_terminal_retry_interval_s)
        while True:
            with self._lock:
                obj_ref = self._orphaned_obj_ref
                if obj_ref is None:
                    self._worker_cancellation_retry_started = False
                    return
            try:
                ray.cancel(obj_ref, force=True)
            except Exception:
                logger.exception("Could not retry worker cancellation", extra={"job_id": self._job_id})
                time.sleep(retry_s)
                continue
            with self._lock:
                if self._orphaned_obj_ref == obj_ref:
                    self._orphaned_obj_ref = None

    def _start_worker_cancellation_retry_locked(self) -> None:
        if self._worker_cancellation_retry_started or self._orphaned_obj_ref is None:
            return
        self._worker_cancellation_retry_started = True
        thread = threading.Thread(target=self._worker_cancellation_retry_loop, daemon=True)
        thread.start()

    def _cancel_orphaned_worker_before_retirement(self, retry_s: float) -> None:
        while True:
            with self._lock:
                orphaned_obj_ref = self._orphaned_obj_ref
            if orphaned_obj_ref is None:
                return
            try:
                ray.cancel(orphaned_obj_ref, force=True)
            except Exception:
                logger.exception("Could not cancel orphaned worker", extra={"job_id": self._job_id})
                time.sleep(retry_s)
                continue
            with self._lock:
                if self._orphaned_obj_ref == orphaned_obj_ref:
                    self._orphaned_obj_ref = None

    def _retirement_loop(self) -> None:
        retry_s = max(0.1, self._controller_terminal_retry_interval_s)
        self._cancel_orphaned_worker_before_retirement(retry_s)
        time.sleep(max(0.0, self._controller_retention_s))
        # A late launch can publish its ObjectRef after this loop begins. Drain
        # retained cancellation work again before any controller cleanup.
        self._cancel_orphaned_worker_before_retirement(retry_s)
        while True:
            with self._lock:
                registry_orphaned = self._registry_orphaned
            if registry_orphaned:
                try:
                    controller = ray.get_actor(self._actor_name, namespace=self._registry_namespace)
                    ray.kill(controller, no_restart=True)
                except ValueError:
                    return
                except Exception:
                    logger.exception("Could not retire orphaned controller", extra={"job_id": self._job_id})
                    time.sleep(retry_s)
                    continue
                return

            registry = self._registry()
            if registry is not None:
                try:
                    registry.sweep_terminal_controllers.remote(now_ts=time.time())
                    return
                except Exception:
                    logger.exception("Could not schedule autonomous controller cleanup", extra={"job_id": self._job_id})
            else:
                logger.warning(
                    "Registry unavailable during autonomous controller cleanup",
                    extra={"job_id": self._job_id},
                )
            time.sleep(retry_s)

    def _start_retirement_locked(self) -> None:
        if self._retirement_started or not self._terminal_committed:
            return
        self._retirement_started = True
        thread = threading.Thread(target=self._retirement_loop, daemon=True)
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
            result = self._push_terminal_best_effort()
            if result.outcome is _RegistryCallOutcome.ACCEPTED and result.status is not None:
                self._accept_registry_terminal(result.status)
                return
            if result.outcome is _RegistryCallOutcome.REJECTED:
                self._accept_registry_terminal_rejection()
                return
            delay_s = min(10.0, delay_s * 2.0)

    def _start_terminal_retry_locked(self) -> None:
        """Start the terminal-write retry thread once while ``self._lock`` is held."""
        if self._terminal_retry_started or self._terminal_committed:
            return
        self._terminal_retry_started = True
        thread = threading.Thread(target=self._terminal_retry_loop, daemon=True)
        thread.start()

    def _set_terminal_locked(
        self,
        status: RegistryStatus,
        *,
        error: str | None = None,
        result_ref: CapabilityRunRefPayload | None = None,
    ) -> ControllerStatePayload:
        """Move to terminal state while the caller holds ``self._lock``."""
        if self._is_terminal(self._status):
            return self._state_locked()
        if self._status is RegistryStatus.CANCELLING:
            status = RegistryStatus.CANCELLED
            error = None
            result_ref = None
        elif status == RegistryStatus.COMPLETED and result_ref is None:
            status = RegistryStatus.FAILED
            error = error or "completed job missing result_ref"
        self._status = status
        self._error = error
        self._result_ref = result_ref
        self._terminal_at_ts = time.time()
        self._obj_ref = None
        return self._state_locked()

    def _accept_registry_terminal(self, status: RegistryStatus) -> None:
        """Align local terminal state with the registry's authoritative result."""
        with self._lock:
            if self._status is not status:
                self._status = status
                if status is RegistryStatus.CANCELLED:
                    self._error = None
                    self._result_ref = None
                elif status is RegistryStatus.COMPLETED:
                    self._error = None
                else:
                    self._result_ref = None
                self._obj_ref = None
                self._terminal_at_ts = self._terminal_at_ts or time.time()
            self._terminal_committed = True
            self._terminal_authoritative = True
            self._start_retirement_locked()
        self._heartbeat_stop.set()

    def _accept_registry_terminal_rejection(self) -> None:
        """Retire terminal work after definitive registry ownership loss."""
        with self._lock:
            self._registry_orphaned = True
            self._terminal_committed = True
            self._start_retirement_locked()
        self._heartbeat_stop.set()

    def _publish_terminal(self) -> None:
        """Publish the already-committed local terminal state without holding the lock."""
        result = self._push_terminal_best_effort()
        if result.outcome is _RegistryCallOutcome.ACCEPTED and result.status is not None:
            self._accept_registry_terminal(result.status)
        elif result.outcome is _RegistryCallOutcome.REJECTED:
            self._accept_registry_terminal_rejection()
        else:
            with self._lock:
                self._start_terminal_retry_locked()

    def _set_terminal(
        self,
        status: RegistryStatus,
        *,
        error: str | None = None,
        result_ref: CapabilityRunRefPayload | None = None,
    ) -> ControllerStatePayload:
        """Move the controller to a terminal state and publish it to the registry."""
        with self._lock:
            self._set_terminal_locked(status, error=error, result_ref=result_ref)
        self._publish_terminal()
        with self._lock:
            return self._state_locked()

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

    def _worker_start_local_decision(self) -> bool | None:
        """Resolve worker startup from local state, or defer to the registry."""
        cancellation_observed = False
        with self._lock:
            if self._is_terminal(self._status):
                return False
            if self._status is RegistryStatus.CANCELLING:
                self._set_terminal_locked(RegistryStatus.CANCELLED)
                cancellation_observed = True
            elif self._status not in {RegistryStatus.SCHEDULING, RegistryStatus.RUNNING}:
                return False

        if cancellation_observed:
            self._publish_terminal()
            return False
        return None

    def _mark_worker_running_best_effort(
        self,
        controller_token: str,
        worker_info: dict[str, Any],
    ) -> _RegistryCallResult:
        """Ask the registry to admit a worker without conflating rejection and transport failure."""
        registry = self._registry()
        if registry is None:
            return _RegistryCallResult(_RegistryCallOutcome.UNAVAILABLE)
        try:
            accepted, raw_status = cast(
                WorkerStartResult,
                ray.get(
                    registry.mark_worker_running.remote(
                        self._scope,
                        self._job_id,
                        self._actor_name,
                        controller_token,
                        worker_info,
                    ),
                    timeout=self._registry_update_timeout_s,
                ),
            )
            status = None if raw_status is None else RegistryStatus(raw_status)
        except Exception:
            logger.exception("Could not validate worker startup with registry", extra={"job_id": self._job_id})
            self._invalidate_registry(registry)
            return _RegistryCallResult(_RegistryCallOutcome.UNAVAILABLE)

        if accepted and status is RegistryStatus.RUNNING:
            return _RegistryCallResult(_RegistryCallOutcome.ACCEPTED, status)
        if not accepted and status is RegistryStatus.CANCELLING:
            return _RegistryCallResult(_RegistryCallOutcome.ACCEPTED, status)
        return _RegistryCallResult(_RegistryCallOutcome.REJECTED)

    def worker_started(self, controller_token: str, worker_info: dict[str, Any]) -> bool | None:
        """Accept the worker's startup handshake and publish actual execution."""
        if not self._controller_token_matches(controller_token):
            return False
        local_decision = self._worker_start_local_decision()
        if local_decision is not None:
            return local_decision

        result = self._mark_worker_running_best_effort(controller_token, worker_info)
        if result.outcome is _RegistryCallOutcome.UNAVAILABLE:
            return None
        if result.outcome is _RegistryCallOutcome.REJECTED:
            self._set_orphaned_live_terminal(cancel_worker=True)
            return False
        if result.status is RegistryStatus.CANCELLING:
            with self._lock:
                obj_ref = self._obj_ref
            if obj_ref is None:
                self._set_terminal(RegistryStatus.CANCELLED)
            else:
                self._set_terminal_and_cancel_worker(RegistryStatus.CANCELLED, obj_ref)
            return False

        with self._lock:
            if self._status is RegistryStatus.CANCELLING:
                self._set_terminal_locked(RegistryStatus.CANCELLED)
                cancelled = True
            elif self._is_terminal(self._status):
                return False
            else:
                self._status = RegistryStatus.RUNNING
                self._worker_started.set()
                return True
        if cancelled:
            self._publish_terminal()
        return False

    def _mark_scheduling(self, reservation_token: str) -> bool:
        registry = self._registry()
        if registry is None:
            raise RuntimeError("job registry unavailable")
        return bool(
            ray.get(
                registry.mark_scheduling.remote(
                    self._scope,
                    self._job_id,
                    reservation_token,
                    self._actor_name,
                    self._registry_namespace,
                    self._scheduling_timeout_s,
                ),
                timeout=self._registry_update_timeout_s,
            )
        )

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
        ``SCHEDULING``. The worker changes it to ``RUNNING`` through a startup
        handshake only after Ray has assigned resources and begun execution.

        Once the registry accepts scheduling, the controller builds the Ray remote
        options from ``resources``, launches the worker task, stores the task
        ``ObjectRef``, starts heartbeats, and starts a watcher thread that will
        publish the final result or error. Calling ``start`` again is safe: if
        work is already running or the job is terminal, the current state is
        returned instead of launching a second task.
        """
        started_scheduling = False
        with self._lock:
            if self._controller_token is None:
                self._controller_token = reservation_token
            elif self._controller_token != reservation_token:
                raise ValueError(f"Invalid controller token for job {self._job_id!r}")
            if self._obj_ref is not None or self._is_terminal(self._status):
                return self._state_locked()

        try:
            if not self._mark_scheduling(reservation_token):
                with self._lock:
                    return self._state_locked()

            with self._lock:
                self._status = RegistryStatus.SCHEDULING
            started_scheduling = True

            resolved_resources = RayTaskResources.from_mapping(resources)
            worker_options = RayWorkerOptions(
                num_cpus=resolved_resources.num_cpus,
                num_gpus=resolved_resources.num_gpus,
                max_retries=max(1, max_retries),
                retry_exceptions=[WorkerStartupUnavailableError],
                resources=dict(resolved_resources.resources),
            )
            if resolved_resources.memory is not None:
                worker_options["memory"] = resolved_resources.memory
            # Ray accepts an exception allowlist at runtime, but its type stub
            # currently declares ``retry_exceptions`` as bool-only.
            remote_fn = cast(Any, ray.remote)(**worker_options)(_execute_capability_ref)
            worker_remote = cast(CapabilityWorkerRemote, remote_fn)
            current_actor = cast(ActorHandle, ray.get_runtime_context().current_actor)

            obj_ref = worker_remote.remote(
                capability,
                dict(run_kwargs),
                controller=current_actor,
                controller_token=reservation_token,
                startup_timeout_s=self._registry_update_timeout_s * 3.0,
            )
            with self._lock:
                if self._is_terminal(self._status):
                    terminal_state = self._state_locked()
                else:
                    self._obj_ref = obj_ref
                    self._status = RegistryStatus.SCHEDULING
                    self._start_heartbeat_locked()
                    self._start_scheduling_timeout_locked(obj_ref)
                    self._start_watcher_locked(obj_ref)
                    return self._state_locked()
            self._cancel_worker_with_retry(
                obj_ref,
                failure_message="Could not cancel worker launched after terminal state",
            )
            return terminal_state
        except BackpressureError:
            # Scheduling admission was rejected before the registry mutated the
            # SUBMITTING reservation. The submitter owns reservation cleanup.
            raise
        except Exception as exc:
            state = self._set_terminal(RegistryStatus.FAILED, error=str(exc))
            if started_scheduling:
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

    def _request_registry_cancellation_best_effort(self, token: str) -> CancellationRequestResult | None:
        registry = self._registry()
        if registry is None:
            return None
        try:
            accepted, _record = cast(
                CancellationRequestResult,
                ray.get(
                    registry.request_cancellation.remote(
                        self._scope,
                        self._job_id,
                        self._actor_name,
                        token,
                    ),
                    timeout=self._registry_update_timeout_s,
                ),
            )
        except Exception:  # noqa: BLE001
            self._invalidate_registry(registry)
            return None
        return accepted, _record

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

        decision = self._request_registry_cancellation_best_effort(token) if token is not None else None
        if decision is not None:
            accepted, record = decision
            if not accepted:
                if record is not None and RegistryStatus(record["status"]) is RegistryStatus.CANCELLED:
                    with self._lock:
                        obj_ref = self._obj_ref
                    if obj_ref is None:
                        self._set_terminal(RegistryStatus.CANCELLED)
                    else:
                        self._set_terminal_and_cancel_worker(RegistryStatus.CANCELLED, obj_ref)
                else:
                    self._set_orphaned_live_terminal(cancel_worker=True)
                return False
            self._accept_cancellation_intent()
            return True

        ready, _ = ray.wait([obj_ref], timeout=0)
        if ready:
            self.reconcile()
            return False

        return self._accept_cancellation_intent()

    def get_state(self, controller_token: str | None = None, reconcile: bool = True) -> ControllerStatePayload:
        if not self._controller_token_matches(controller_token):
            raise PermissionError(f"Invalid controller token for job {self._job_id!r}")
        if reconcile:
            return self.reconcile()
        with self._lock:
            return self._state_locked()


JobControllerActor = ray.remote(max_restarts=0)(JobController)


def _kill_created_controller_after_failed_startup(
    controller: ActorHandle,
    name: str,
    *,
    created_here: bool,
) -> None:
    """Best-effort cleanup for a controller this client created but could not use."""
    if not created_here:
        return
    try:
        ray.kill(controller, no_restart=True)
    except Exception:  # noqa: BLE001 - cleanup must not mask the startup failure
        logger.warning(
            "Could not clean up newly created Ray controller after failed startup",
            extra={"controller_actor_name": name},
            exc_info=True,
        )


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
    scheduling_timeout_s: float | None = DEFAULT_SCHEDULING_TIMEOUT_S,
    controller_retention_s: float = DEFAULT_CONTROLLER_RETENTION_S,
    controller_max_pending_calls: int | None = None,
    controller_token: str | None = None,
    startup_timeout_s: float = DEFAULT_CONTROLLER_STARTUP_TIMEOUT_S,
) -> ActorHandle:
    """Get or create a controller and validate its protocol, identity, and settings."""
    created_here = False
    expected_configuration = _controller_configuration(
        registry_update_timeout_s=registry_update_timeout_s,
        controller_heartbeat_interval_s=controller_heartbeat_interval_s,
        controller_terminal_retry_interval_s=controller_terminal_retry_interval_s,
        scheduling_timeout_s=scheduling_timeout_s,
        controller_retention_s=controller_retention_s,
    )
    expected_descriptor = _controller_descriptor(
        actor_name=name,
        registry_name=registry_name,
        registry_namespace=registry_namespace,
        scope=scope,
        job_id=job_id,
        configuration=expected_configuration,
    )
    try:
        controller = ray.get_actor(name, namespace=namespace)
    except ValueError:
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
            controller = cast(
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
                    scheduling_timeout_s=scheduling_timeout_s,
                    controller_retention_s=controller_retention_s,
                    controller_token=controller_token,
                ),
            )
            created_here = True
        except ValueError:
            controller = ray.get_actor(name, namespace=namespace)

    try:
        descriptor = ray.get(controller.describe.remote(controller_token), timeout=float(startup_timeout_s))
    except GetTimeoutError as exc:
        _kill_created_controller_after_failed_startup(controller, name, created_here=created_here)
        raise ControllerStartupError(
            f"Ray controller actor {name!r} did not become ready within {float(startup_timeout_s):.3f}s"
        ) from exc
    except Exception as exc:
        _kill_created_controller_after_failed_startup(controller, name, created_here=created_here)
        raise ControllerCompatibilityError(
            f"Ray controller actor {name!r} does not support the required Checkmaite controller handshake"
        ) from exc
    if descriptor != expected_descriptor:
        _kill_created_controller_after_failed_startup(controller, name, created_here=created_here)
        raise ControllerCompatibilityError(
            f"Existing Ray controller actor {name!r} is incompatible: "
            f"expected {expected_descriptor!r}, got {descriptor!r}"
        )
    return controller
