from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, NoReturn, TypeVar, cast

import ray
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError

try:
    from ray.exceptions import PendingCallsLimitExceeded
except ImportError:  # pragma: no cover - compatibility with Ray versions lacking this exception
    PendingCallsLimitExceeded = type("PendingCallsLimitExceeded", (Exception,), {})

from checkmaite.jobs._store import AnalyticsStoreConfig
from checkmaite.jobs._submission import prepare_job_submission_run_kwargs
from checkmaite.jobs.protocol import (
    BackpressureError,
    CapabilityRunRef,
    CapabilityRunRefPayload,
    CapabilityType,
    Job,
    JobCancelledError,
    JobFailedError,
    JobStatus,
    JobTimeoutError,
)

from .controller import (
    DEFAULT_CONTROLLER_HEARTBEAT_INTERVAL_S,
    DEFAULT_CONTROLLER_NUM_CPUS,
    DEFAULT_CONTROLLER_TERMINAL_RETRY_INTERVAL_S,
    DEFAULT_REGISTRY_UPDATE_TIMEOUT_S,
    ControllerStatePayload,
    RayTaskResources,
    get_or_create_controller_actor,
)
from .registry import (
    DEFAULT_CONTROLLER_HEARTBEAT_TTL_S,
    DEFAULT_CONTROLLER_RETENTION_S,
    DEFAULT_LIST_JOBS_LIMIT,
    DEFAULT_MAX_RETAINED_TERMINAL_CONTROLLERS,
    DEFAULT_MAX_RETAINED_TERMINAL_JOBS_PER_SCOPE,
    DEFAULT_REGISTRY_ACTOR_NAME,
    DEFAULT_REGISTRY_NAMESPACE,
    DEFAULT_REGISTRY_SWEEP_BATCH_LIMIT,
    DEFAULT_REGISTRY_SWEEP_INTERVAL_S,
    DEFAULT_RESERVATION_TTL_S,
    DEFAULT_TERMINAL_JOB_RETENTION_S,
    REGISTRY_TERMINAL_STATUSES,
    JobRecord,
    JobRegistrationRecord,
    NewJobRegistrationRecord,
    RegistryStatus,
    get_or_create_registry_actor,
)

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_MAX_PENDING_CALLS = 1024
DEFAULT_CONTROLLER_MAX_PENDING_CALLS = 64
_JOB_POLL_BACKOFF_DELAYS_S = (0.05, 0.25, 1.0, 5.0)

_RayGetResultT = TypeVar("_RayGetResultT")


def _is_pending_calls_limit_exceeded(exc: BaseException) -> bool:
    return isinstance(exc, PendingCallsLimitExceeded)


def _backpressure_error(action: str) -> BackpressureError:
    return BackpressureError(
        "Ray job backend control plane is overloaded while "
        f"{action}; actor pending-call limit was exceeded. Retry with exponential backoff, reduce "
        "client concurrency, or tune registry_max_pending_calls/controller_max_pending_calls for expected bursts."
    )


def _ray_get_with_backpressure(
    ref_factory: Callable[[], ray.ObjectRef[_RayGetResultT]],
    *,
    timeout_s: float | None,
    action: str,
) -> _RayGetResultT:
    try:
        return ray.get(ref_factory(), timeout=timeout_s)
    except Exception as exc:
        if _is_pending_calls_limit_exceeded(exc):
            raise _backpressure_error(action) from exc
        raise


class _ControllerJobSnapshot(JobRecord):
    """Job snapshot that includes fresh state from the controller.

    ``terminal_at_ts`` is the controller's local Unix timestamp for when it saw
    the job finish. It may be ``None`` while the job is still running or when the
    snapshot came from registry state only.
    """

    terminal_at_ts: float | None


@dataclass(frozen=True)
class ControllerState:
    """Current state read directly from the job's controller actor.

    The controller is the Ray actor that owns the live worker task. This state is
    useful while the registry still says the job is active, because the controller
    may have already seen the task finish and not yet written that final state to
    the registry.
    """

    job_id: str
    status: RegistryStatus
    result_ref: CapabilityRunRefPayload | None
    error: str | None
    terminal_at_ts: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "result_ref": self.result_ref,
            "error": self.error,
            "terminal_at_ts": self.terminal_at_ts,
        }


# JobRecord is a TypedDict, so at runtime it is only a plain dict and cannot be
# used in isinstance() checks. Wrap it in a small dataclass so we can use
# isinstance() to check if a snapshot came from the registry or the controller.
@dataclass(frozen=True)
class RegistrySnapshot:
    record: JobRecord


def _created_at_from_submitted_ts(submitted_at_ts: float) -> datetime:
    return datetime.fromtimestamp(float(submitted_at_ts), tz=timezone.utc)


def _registry_status_from_raw(status: RegistryStatus | str) -> RegistryStatus:
    if isinstance(status, RegistryStatus):
        return status
    try:
        return RegistryStatus(status.upper())
    except ValueError as exc:
        raise ValueError(f"Unknown registry status {status!r}") from exc


def _job_status_from_fields(
    status: RegistryStatus | str,
    result_ref: CapabilityRunRefPayload | None = None,
) -> JobStatus:
    """Convert the registry's internal status into the public job status.

    The registry has a few states that users do not see directly. For example,
    ``SUBMITTING`` is shown as ``PENDING``, and ``CANCELLING`` is still shown as
    ``RUNNING`` because the public API has no separate cancelling state.

    A completed job is only public ``COMPLETED`` when it also has a valid
    ``CapabilityRunRef`` payload. If that payload is missing or invalid, the job
    is reported as ``FAILED`` so callers do not receive a broken result.
    """
    status = _registry_status_from_raw(status)

    if status is RegistryStatus.SUBMITTING:
        return JobStatus.PENDING

    if status in {RegistryStatus.RUNNING, RegistryStatus.CANCELLING}:
        return JobStatus.RUNNING

    if status is RegistryStatus.COMPLETED:
        if result_ref is None:
            return JobStatus.FAILED
        try:
            CapabilityRunRef.model_validate(result_ref)
        except Exception:  # noqa: BLE001
            return JobStatus.FAILED
        return JobStatus.COMPLETED

    if status is RegistryStatus.FAILED:
        return JobStatus.FAILED

    if status is RegistryStatus.CANCELLED:
        return JobStatus.CANCELLED

    raise ValueError(f"Unknown registry status {status.value!r}")


def _controller_state_from_raw(raw_state: ControllerStatePayload | Mapping[str, Any]) -> ControllerState:
    """Build typed controller state from the raw dict returned by the actor.

    Ray actor calls return plain Python objects. The controller is expected to
    return a small dict with job id, status, result payload, error text, and the
    time it observed terminal state. This helper normalizes those fields into a
    ``ControllerState`` and fails early if the result payload has the wrong basic
    shape.
    """
    raw_result_ref = raw_state.get("result_ref")
    if raw_result_ref is not None and not isinstance(raw_result_ref, dict):
        raise TypeError("controller result_ref must be a mapping or None")

    raw_error = raw_state.get("error")
    raw_terminal_at_ts = raw_state.get("terminal_at_ts")
    return ControllerState(
        job_id=str(raw_state["job_id"]),
        status=_registry_status_from_raw(raw_state["status"]),
        result_ref=cast(CapabilityRunRefPayload | None, raw_result_ref),
        error=None if raw_error is None else str(raw_error),
        terminal_at_ts=None if raw_terminal_at_ts is None else float(raw_terminal_at_ts),
    )


def _snapshot_from_controller_state(record: JobRecord, state: ControllerState) -> _ControllerJobSnapshot:
    """Merge a registry record with the latest state read from the controller.

    The registry record has stable job metadata such as scope, submit time, and
    controller identity. The controller state may be newer for live fields such as
    status, result, error, and ``terminal_at_ts``. This helper keeps the registry
    metadata and overlays those live controller fields.
    """
    if state.job_id != record["job_id"]:
        raise ValueError(f"controller state job_id {state.job_id!r} does not match record job_id {record['job_id']!r}")

    snapshot = cast(_ControllerJobSnapshot, dict(record))
    snapshot["status"] = state.status
    if state.result_ref is not None:
        snapshot["result_ref"] = state.result_ref
    if state.error is not None:
        snapshot["error"] = state.error
    snapshot["terminal_at_ts"] = state.terminal_at_ts
    return snapshot


class RayJob(Job[CapabilityRunRef]):
    """Handle returned for a Ray-backed capability run.

    A ``RayJob`` is the user's receipt for submitted work. It lets callers check
    status, wait for completion, get the final ``CapabilityRunRef``, inspect a
    failure, or ask Ray to cancel the run.

    The actual capability run happens elsewhere in Ray. This handle only knows
    how to find and observe that work. It first reads a small shared registry
    entry for ``(scope, job_id)``. The registry acts like a cluster-local job
    directory: it stores information like status, timestamps and the final result
    reference. It should not store datasets, models, logs, full run objects, or
    other large data. See ``registry.py`` for the registry implementation.

    Each active job also has a small Ray actor called a controller. The
    controller is the job's long-lived caretaker inside the Ray cluster: it starts
    the worker task, keeps the task reference, watches it finish, updates the
    registry, sends heartbeats, and handles cancellation. This handle may contact
    the controller when the registry says the job is still active. The handle
    lives in the submitting Python process, but the controller does not. Because
    the controller is detached, the run can keep going after the process that
    created this handle exits, as long as the Ray cluster stays alive. See
    ``controller.py`` for the controller implementation.

    Parameters
    ----------
    registry
        Handle to the shared ``JobRegistryActor`` used to read and update job
        metadata.
    scope
        User-provided namespace for job ids and duplicate-submission keys. This
        is required so different projects or workspaces do not accidentally
        deduplicate against each other while still allowing clients in the same
        scope to reattach to the same job.
    job_id
        Registry job id for this submitted capability run.
    created_at
        UTC time when the job was submitted.
    control_plane_timeout_s
        Maximum seconds to wait for one registry or controller Ray call made by
        this handle. This is separate from the total timeout passed to
        ``result()`` or ``wait()``.
    initial_status
        Best status known when this handle was created. Used as a fallback if a
        later best-effort status check cannot reach Ray.
    """

    def __init__(
        self,
        *,
        registry: ActorHandle,
        scope: str,
        job_id: str,
        created_at: datetime,
        control_plane_timeout_s: float = DEFAULT_REGISTRY_UPDATE_TIMEOUT_S,
        initial_status: JobStatus = JobStatus.PENDING,
    ) -> None:
        self._registry = registry
        self._scope = scope
        self._job_id = job_id
        self._created_at = created_at
        self._control_plane_timeout_s = float(control_plane_timeout_s)
        self._last_status = initial_status

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @staticmethod
    def _remaining_s(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.time())

    def _control_call_timeout_s(self, deadline: float | None) -> float:
        """Bound one registry/controller call without changing the user's total timeout."""
        remaining = self._remaining_s(deadline)
        if remaining is None:
            return self._control_plane_timeout_s
        return min(remaining, self._control_plane_timeout_s)

    @staticmethod
    def _next_poll_delay_s(current_delay_s: float) -> float:
        for delay_s in _JOB_POLL_BACKOFF_DELAYS_S:
            if delay_s > current_delay_s:
                return delay_s
        return _JOB_POLL_BACKOFF_DELAYS_S[-1]

    def _sleep_before_next_poll(self, deadline: float | None, delay_s: float) -> None:
        remaining = self._remaining_s(deadline)
        time.sleep(delay_s if remaining is None else min(delay_s, remaining))

    def _remember_status(self, snapshot: JobRecord | _ControllerJobSnapshot) -> JobStatus:
        """Update and return the last public status seen by this handle.

        ``status`` is allowed to be best-effort. If a later Ray call fails, the
        public ``status`` property can return this cached value instead of
        raising an error for normal polling code.
        """
        status = _job_status_from_fields(snapshot["status"], snapshot["result_ref"])
        self._last_status = status
        return status

    def _raise_timeout(self, timeout: float | None) -> NoReturn:
        raise JobTimeoutError(self._job_id, timeout or 0.0)

    def _fetch_record(self, timeout_s: float | None = None) -> JobRecord:
        """Read this job's metadata record from the registry actor.

        The record is created and stored by ``JobRegistryActor`` in
        ``registry.py``. It is keyed by this handle's ``scope`` and ``job_id`` and
        contains small job metadata such as status, timestamps, controller
        identity, error text, and final result reference.
        """
        record = cast(
            JobRecord | None,
            _ray_get_with_backpressure(
                lambda: self._registry.get_job.remote(self._scope, self._job_id),
                timeout_s=timeout_s,
                action=f"reading job {self._job_id!r}",
            ),
        )
        if record is None:
            raise KeyError(f"No job with ID {self._job_id!r} in scope {self._scope!r}")
        return record

    def _fail_unavailable_controller(
        self,
        record: JobRecord,
        error: str,
        timeout_s: float | None = None,
    ) -> JobRecord:
        """Ask the registry to fail a job whose controller cannot be used.

        The registry requires the recorded controller name and token before it
        accepts this terminal transition. If the local record does not include a
        token, avoid force-failing the job and fall back to the heartbeat-staleness
        sweep instead.
        """
        update_timeout_s = (
            self._control_plane_timeout_s
            if timeout_s is None
            else min(max(0.0, timeout_s), self._control_plane_timeout_s)
        )
        controller_token = record.get("controller_token")
        if controller_token is None:
            _ray_get_with_backpressure(
                lambda: self._registry.sweep_stale_running_jobs.remote(self._scope, self._job_id),
                timeout_s=update_timeout_s,
                action=f"sweeping stale job {self._job_id!r}",
            )
            refreshed = cast(
                JobRecord | None,
                _ray_get_with_backpressure(
                    lambda: self._registry.get_job.remote(self._scope, self._job_id),
                    timeout_s=update_timeout_s,
                    action=f"reading job {self._job_id!r}",
                ),
            )
            return record if refreshed is None else refreshed

        failed = cast(
            JobRecord | None,
            _ray_get_with_backpressure(
                lambda: self._registry.mark_controller_unavailable.remote(
                    self._scope,
                    self._job_id,
                    record.get("controller_actor_name"),
                    controller_token,
                    error,
                ),
                timeout_s=update_timeout_s,
                action=f"failing unavailable controller for job {self._job_id!r}",
            ),
        )
        if failed is None:
            raise RuntimeError(f"Registry could not mark controller unavailable for job {self._job_id!r}")
        return failed

    @staticmethod
    def _get_controller_actor(
        actor_name: str | None,
        namespace: str | None,
    ) -> ActorHandle | None:
        if not actor_name:
            return None

        try:
            return ray.get_actor(actor_name, namespace=namespace)
        except Exception:  # noqa: BLE001
            return None

    def _sweep_stale_running_job(self, deadline: float | None) -> JobRecord | None:
        """Let the registry fail this job if the controller stopped checking in.

        Controllers send periodic heartbeats while a job is running. A missing or
        slow controller is not failed immediately, because Ray may only be slow or
        temporarily unavailable. This asks the registry to check whether the last
        heartbeat is too old and then reads back the job record.
        """
        _ray_get_with_backpressure(
            lambda: self._registry.sweep_stale_running_jobs.remote(self._scope, self._job_id),
            timeout_s=self._control_call_timeout_s(deadline),
            action=f"sweeping stale job {self._job_id!r}",
        )
        return cast(
            JobRecord | None,
            _ray_get_with_backpressure(
                lambda: self._registry.get_job.remote(self._scope, self._job_id),
                timeout_s=self._control_call_timeout_s(deadline),
                action=f"reading job {self._job_id!r}",
            ),
        )

    def _commit_controller_terminal(
        self,
        *,
        state: ControllerState,
        controller_actor_name: str | None,
        controller_token: str | None,
        deadline: float | None,
    ) -> JobRecord:
        """Write a terminal controller state to the registry before returning it.

        The controller can see the worker finish before the registry has the
        final result. Before this handle returns success or failure to a caller,
        it commits that terminal state to the registry so other clients can see
        the same answer.

        ``controller_token`` is the registry's owner token for this controller.
        The registry checks it with ``controller_actor_name`` before accepting a
        terminal update. This prevents a stale or wrong controller from updating
        a job it does not own, so the token is needed for ownership safety.
        """
        _ray_get_with_backpressure(
            lambda: self._registry.update_terminal.remote(
                self._scope,
                self._job_id,
                state.status,
                state.error,
                state.result_ref,
                controller_actor_name,
                controller_token,
            ),
            timeout_s=self._control_call_timeout_s(deadline),
            action=f"committing terminal state for job {self._job_id!r}",
        )
        return self._fetch_record(timeout_s=self._control_call_timeout_s(deadline))

    def _record_with_missing_controller(
        self,
        record: JobRecord,
        record_status: JobStatus,
        deadline: float | None,
        fail_unavailable_controller: bool,
    ) -> JobRecord:
        """Handle a non-terminal job when its controller actor cannot be found.

        A pending job may not have a controller yet, so the record is returned as
        it is. For a running job, the caller chooses the behavior: fail the job
        right away, or first ask the registry whether the controller has stopped
        sending heartbeats for too long.
        """
        if record_status is not JobStatus.RUNNING:
            return record
        if fail_unavailable_controller:
            return self._fail_unavailable_controller(
                record,
                "job controller unavailable",
                timeout_s=self._remaining_s(deadline),
            )
        swept = self._sweep_stale_running_job(deadline)
        return record if swept is None else swept

    def _record_after_controller_error(
        self,
        record: JobRecord,
        record_status: JobStatus,
        deadline: float | None,
        exc: Exception,
        fail_unavailable_controller: bool,
    ) -> JobRecord:
        """Resolve a controller read error without hiding registry truth.

        For running jobs, callers can choose between failing the job immediately
        or asking the registry whether the controller heartbeat has actually
        expired. Non-running records are returned unchanged.
        """
        if record_status is not JobStatus.RUNNING:
            return record
        if fail_unavailable_controller:
            return self._fail_unavailable_controller(
                record,
                f"job controller unavailable: {exc}",
                timeout_s=self._remaining_s(deadline),
            )
        swept = self._sweep_stale_running_job(deadline)
        return record if swept is None else swept

    def _fetch_controller_state(
        self,
        controller: ActorHandle,
        record: JobRecord,
        record_status: JobStatus,
        deadline: float | None,
        fail_unavailable_controller: bool,
    ) -> ControllerState | RegistrySnapshot:
        """Read live state from the controller or return a registry fallback.

        A successful controller read returns ``ControllerState``. If the
        controller call fails and the caller does not want an immediate failure,
        this may instead return ``RegistrySnapshot`` after checking the registry's
        stale-heartbeat rule.
        """
        try:
            raw_state = cast(
                ControllerStatePayload,
                _ray_get_with_backpressure(
                    lambda: controller.get_state.remote(record.get("controller_token")),
                    timeout_s=self._control_call_timeout_s(deadline),
                    action=f"reading controller state for job {self._job_id!r}",
                ),
            )
            return _controller_state_from_raw(raw_state)
        except GetTimeoutError:
            if record_status is JobStatus.RUNNING and not fail_unavailable_controller:
                swept = self._sweep_stale_running_job(deadline)
                if swept is not None and _job_status_from_fields(swept["status"], swept["result_ref"]).is_terminal:
                    return RegistrySnapshot(swept)
            raise
        except BackpressureError:
            raise
        except Exception as exc:  # noqa: BLE001
            return RegistrySnapshot(
                self._record_after_controller_error(record, record_status, deadline, exc, fail_unavailable_controller)
            )

    def _fetch_snapshot(
        self,
        timeout_s: float | None = None,
        *,
        commit_controller_terminal: bool = True,
        fail_unavailable_controller: bool = True,
    ) -> JobRecord | _ControllerJobSnapshot:
        """Return the best job state this handle can safely observe.

        The registry is checked first because terminal registry records are shared
        truth for all clients. For active jobs, the controller may have newer live
        state. If the controller reports a terminal state, this method normally
        commits it to the registry before returning it.
        """
        deadline = None if timeout_s is None else time.time() + timeout_s
        record = self._fetch_record(timeout_s=self._control_call_timeout_s(deadline))
        record_status = _job_status_from_fields(record["status"], record["result_ref"])
        if record_status.is_terminal:
            return record

        controller = self._get_controller_actor(record["controller_actor_name"], record["controller_namespace"])
        if controller is None:
            return self._record_with_missing_controller(record, record_status, deadline, fail_unavailable_controller)

        fetched_state = self._fetch_controller_state(
            controller,
            record,
            record_status,
            deadline,
            fail_unavailable_controller,
        )
        if isinstance(fetched_state, RegistrySnapshot):
            return fetched_state.record

        state = fetched_state
        if state.job_id != record["job_id"]:
            return self._record_after_controller_error(
                record,
                record_status,
                deadline,
                RuntimeError(f"job controller reported state for {state.job_id!r}, expected {record['job_id']!r}"),
                fail_unavailable_controller,
            )

        if state.status in REGISTRY_TERMINAL_STATUSES:
            if not commit_controller_terminal:
                return record
            return self._commit_controller_terminal(
                state=state,
                controller_actor_name=record["controller_actor_name"],
                controller_token=record["controller_token"],
                deadline=deadline,
            )

        if record_status is JobStatus.RUNNING and state.status is RegistryStatus.SUBMITTING:
            if fail_unavailable_controller:
                return self._fail_unavailable_controller(
                    record,
                    "job controller lost task ownership",
                    timeout_s=self._remaining_s(deadline),
                )
            return record

        return _snapshot_from_controller_state(record, state)

    def _ref_from_snapshot(self, snapshot: JobRecord | _ControllerJobSnapshot) -> CapabilityRunRef:
        """Validate and return the result reference stored in a successful snapshot.

        A completed job must contain a usable ``CapabilityRunRef`` payload. If it
        does not, the job is treated as failed rather than returning a broken
        result object to the caller.
        """
        result_ref = snapshot["result_ref"]
        if result_ref is None:
            raise JobFailedError(self._job_id, "completed without result payload")
        try:
            return CapabilityRunRef.model_validate(result_ref)
        except Exception as exc:
            raise JobFailedError(self._job_id, f"invalid result payload: {exc}") from exc

    def _raise_for_snapshot(self, snapshot: JobRecord | _ControllerJobSnapshot) -> None:
        """Raise the public job exception represented by a terminal snapshot.

        ``result()`` uses this after each poll. Failed jobs become
        ``JobFailedError``, cancelled jobs become ``JobCancelledError``, and
        pending/running/completed snapshots do not raise here.
        """
        status = _job_status_from_fields(snapshot["status"], snapshot["result_ref"])
        if status is JobStatus.CANCELLED:
            raise JobCancelledError(self._job_id)
        if status is JobStatus.FAILED:
            raise JobFailedError(self._job_id, snapshot.get("error") or "failed")

    @property
    def status(self) -> JobStatus:
        """Return the latest status this handle can observe without blocking long.

        This property is meant for lightweight polling in notebooks and user
        interfaces. It tries to read the registry and, for active jobs, may also
        ask the controller for fresher state. If those Ray calls are slow or fail
        temporarily, it returns the last status this handle saw instead of
        raising. That means ``status`` is best-effort: it may be slightly stale,
        but it should stay safe and responsive.

        Use ``wait()`` or ``result()`` when the caller needs to wait for a final
        state or receive a result/error.
        """
        try:
            return self._remember_status(
                self._fetch_snapshot(
                    timeout_s=self._control_plane_timeout_s,
                    commit_controller_terminal=True,
                    fail_unavailable_controller=False,
                )
            )
        except GetTimeoutError:
            return self._last_status
        except BackpressureError:
            raise
        except Exception:  # noqa: BLE001
            # Status is a best-effort observation. If the live controller cannot
            # be reached, fall back to the registry record before surfacing Ray
            # control-plane errors to notebook users.
            try:
                return self._remember_status(self._fetch_record(timeout_s=self._control_plane_timeout_s))
            except BackpressureError:
                raise
            except Exception:  # noqa: BLE001
                return self._last_status

    def result(self, timeout: float | None = None) -> CapabilityRunRef:
        """Wait for the job to succeed and return its ``CapabilityRunRef``.

        This is the strict result API. It keeps polling until the job completes,
        fails, is cancelled, or the caller's timeout expires. Successful jobs
        return the small stored reference to the run; failed and cancelled jobs
        raise the public job exceptions. If the controller reports a final state
        before the registry has it, this method commits that state to the
        registry before returning so other clients can see the same result.
        """
        deadline = None if timeout is None else time.time() + timeout
        first_poll = True
        poll_delay_s = _JOB_POLL_BACKOFF_DELAYS_S[0]

        while True:
            remaining = self._remaining_s(deadline)
            fetch_timeout = remaining
            if remaining is not None and remaining <= 0:
                if not first_poll:
                    self._raise_timeout(timeout)
                fetch_timeout = 0.0

            try:
                snapshot = self._fetch_snapshot(timeout_s=fetch_timeout, fail_unavailable_controller=False)
            except GetTimeoutError:
                remaining_after_timeout = self._remaining_s(deadline)
                if remaining_after_timeout is not None and remaining_after_timeout <= 0:
                    self._raise_timeout(timeout)
                first_poll = False
                self._sleep_before_next_poll(deadline, poll_delay_s)
                poll_delay_s = self._next_poll_delay_s(poll_delay_s)
                continue
            first_poll = False

            status = self._remember_status(snapshot)
            if status is JobStatus.COMPLETED:
                return self._ref_from_snapshot(snapshot)
            self._raise_for_snapshot(snapshot)

            remaining = self._remaining_s(deadline)
            if remaining is not None and remaining <= 0:
                self._raise_timeout(timeout)
            self._sleep_before_next_poll(deadline, poll_delay_s)
            poll_delay_s = self._next_poll_delay_s(poll_delay_s)

    def wait(self, timeout: float | None = None) -> JobStatus:
        """Wait until the job is terminal or the timeout expires.

        Unlike ``result()``, this method does not return a result payload and does
        not raise for failed or cancelled jobs. It returns the terminal status when
        one is reached. If the timeout expires first, it returns the latest status
        this handle has observed.
        """
        deadline = None if timeout is None else time.time() + timeout
        last_status = self._last_status
        first_poll = True
        poll_delay_s = _JOB_POLL_BACKOFF_DELAYS_S[0]

        while True:
            remaining = self._remaining_s(deadline)
            fetch_timeout = remaining
            if remaining is not None and remaining <= 0:
                if not first_poll:
                    return last_status
                fetch_timeout = 0.0

            try:
                snapshot = self._fetch_snapshot(timeout_s=fetch_timeout, fail_unavailable_controller=False)
            except GetTimeoutError:
                remaining_after_timeout = self._remaining_s(deadline)
                if remaining_after_timeout is not None and remaining_after_timeout <= 0:
                    return last_status
                first_poll = False
                self._sleep_before_next_poll(deadline, poll_delay_s)
                poll_delay_s = self._next_poll_delay_s(poll_delay_s)
                continue
            first_poll = False

            last_status = self._remember_status(snapshot)
            if last_status.is_terminal:
                return last_status

            remaining = self._remaining_s(deadline)
            if remaining is not None and remaining <= 0:
                return last_status
            self._sleep_before_next_poll(deadline, poll_delay_s)
            poll_delay_s = self._next_poll_delay_s(poll_delay_s)

    def exception(self) -> BaseException | None:
        """Return a lightweight failure object if the job is known to have failed.

        This is a convenience method for inspection. It returns ``RuntimeError``
        with the stored error text for failed jobs, and ``None`` for running,
        pending, completed, cancelled, or temporarily unreadable jobs. Use
        ``result()`` when the caller needs the protocol-specific exception.
        """
        try:
            snapshot = self._fetch_snapshot(
                timeout_s=self._control_plane_timeout_s,
                commit_controller_terminal=True,
                fail_unavailable_controller=False,
            )
        except BackpressureError:
            raise
        except Exception:  # noqa: BLE001
            return None

        if self._remember_status(snapshot) is JobStatus.FAILED:
            return RuntimeError(str(snapshot.get("error") or "failed"))
        return None

    def _request_registry_cancellation(self, snapshot: JobRecord) -> JobRecord | None:
        return cast(
            JobRecord | None,
            _ray_get_with_backpressure(
                lambda: self._registry.request_cancellation.remote(
                    self._scope,
                    self._job_id,
                    snapshot.get("controller_actor_name"),
                    snapshot.get("controller_token"),
                ),
                timeout_s=self._control_plane_timeout_s,
                action=f"requesting cancellation for job {self._job_id!r}",
            ),
        )

    def _request_registry_cancellation_best_effort(self, snapshot: JobRecord) -> tuple[bool, JobRecord | None]:
        try:
            requested = self._request_registry_cancellation(snapshot)
        except BackpressureError:
            raise
        except Exception:  # noqa: BLE001
            logger.debug(
                "Registry cancellation request failed",
                extra={"job_id": self._job_id, "scope": self._scope},
                exc_info=True,
            )
            return False, None
        return requested is not None and requested["status"] in {
            RegistryStatus.CANCELLING,
            RegistryStatus.CANCELLED,
        }, requested

    def cancel(self) -> bool:
        """Ask Ray to cancel this job.

        Cancellation is best-effort. For jobs that have not started yet, the
        registry can mark the submission cancelled directly. For running jobs,
        this records cancellation intent in the registry and asks the controller
        to cancel the worker task. A return value of ``True`` means a cancellation
        request was issued or may still be in flight; it does not guarantee that
        the final state will be ``CANCELLED`` if the task finishes at the same
        time.
        """
        try:
            snapshot = self._fetch_record(timeout_s=self._control_plane_timeout_s)
        except BackpressureError:
            raise
        except Exception:  # noqa: BLE001
            return False
        if _job_status_from_fields(snapshot["status"], snapshot["result_ref"]).is_terminal:
            return False

        controller = self._get_controller_actor(snapshot["controller_actor_name"], snapshot["controller_namespace"])
        if controller is None:
            if _registry_status_from_raw(snapshot["status"]) is not RegistryStatus.SUBMITTING:
                return False
            _cancel_requested, cancelled = self._request_registry_cancellation_best_effort(snapshot)
            if cancelled is None:
                return False
            self._remember_status(cancelled)
            return _job_status_from_fields(cancelled["status"], cancelled["result_ref"]) is JobStatus.CANCELLED

        cancel_requested, _requested = self._request_registry_cancellation_best_effort(snapshot)
        try:
            controller_cancelled = bool(
                _ray_get_with_backpressure(
                    lambda: controller.cancel.remote(snapshot.get("controller_token")),
                    timeout_s=self._control_plane_timeout_s,
                    action=f"cancelling controller for job {self._job_id!r}",
                )
            )
            return controller_cancelled or cancel_requested
        except GetTimeoutError:
            return True
        except Exception:  # noqa: BLE001
            return cancel_requested


class RayJobBackend:
    """Submit and reconnect to capability jobs running on Ray.

    A ``RayJobBackend`` is a client-side object. User code may create it directly,
    or ``checkmaite.jobs.configure_job_backend(kind="ray", ...)`` creates one and
    stores it as the active job backend in ``src/checkmaite/jobs/_api.py``. Calls like
    ``checkmaite.jobs.submit_capability(...)`` then use that active job backend.

    The job backend stores small job records in a shared registry actor and starts
    one controller actor for each new job. The registry makes job IDs, duplicate
    submissions, and final states visible to other job backend instances that use the
    same scope. Controllers own the actual Ray worker task and report progress
    back to the registry.

    If this Python object or its process dies, only its local cache of ``RayJob``
    handles is lost. The detached registry and controller actors can keep running
    in the Ray cluster, so the capability run can continue. A later backend with
    the same scope and registry settings can list jobs or get a job by ID to
    reattach. This recovery depends on the Ray cluster and detached actors still
    being alive.
    """

    @staticmethod
    def _validate_job_tracking_settings(
        registry_update_timeout_s: float,
        registry_controller_heartbeat_ttl_s: float,
        controller_heartbeat_interval_s: float,
        controller_terminal_retry_interval_s: float,
        registry_sweep_interval_s: float,
        registry_sweep_batch_limit: int | None,
    ) -> tuple[float, float, float, float, float, int | None]:
        """Validate settings that keep job tracking responsive.

        These values control how long registry/controller calls may take, how
        often controllers check in, and how often old records are cleaned up. The
        checks reject impossible timings and make sure a running job is not
        marked stale before it has had several chances to send a heartbeat.
        """
        registry_update_timeout_s = float(registry_update_timeout_s)
        registry_controller_heartbeat_ttl_s = float(registry_controller_heartbeat_ttl_s)
        controller_heartbeat_interval_s = float(controller_heartbeat_interval_s)
        controller_terminal_retry_interval_s = float(controller_terminal_retry_interval_s)
        registry_sweep_interval_s = float(registry_sweep_interval_s)

        if registry_update_timeout_s <= 0:
            raise ValueError("registry_update_timeout_s must be > 0")
        if registry_controller_heartbeat_ttl_s <= 0:
            raise ValueError("registry_controller_heartbeat_ttl_s must be > 0")
        if controller_heartbeat_interval_s <= 0:
            raise ValueError("controller_heartbeat_interval_s must be > 0")
        min_heartbeat_ttl_s = 3 * controller_heartbeat_interval_s + registry_update_timeout_s
        if registry_controller_heartbeat_ttl_s < min_heartbeat_ttl_s:
            raise ValueError(
                "registry_controller_heartbeat_ttl_s must be at least "
                "3 * controller_heartbeat_interval_s + registry_update_timeout_s"
            )
        if controller_terminal_retry_interval_s <= 0:
            raise ValueError("controller_terminal_retry_interval_s must be > 0")
        if registry_sweep_interval_s < 0:
            raise ValueError("registry_sweep_interval_s must be >= 0")
        if registry_sweep_batch_limit is not None:
            if isinstance(registry_sweep_batch_limit, bool) or not isinstance(registry_sweep_batch_limit, int):
                raise TypeError("registry_sweep_batch_limit must be a non-negative integer or None")
            if registry_sweep_batch_limit < 0:
                raise ValueError("registry_sweep_batch_limit must be >= 0 or None")

        return (
            registry_update_timeout_s,
            registry_controller_heartbeat_ttl_s,
            controller_heartbeat_interval_s,
            controller_terminal_retry_interval_s,
            registry_sweep_interval_s,
            registry_sweep_batch_limit,
        )

    @staticmethod
    def _validate_max_pending_calls(name: str, value: int | None) -> int | None:
        """Validate an optional Ray actor pending-call limit."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer or None")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer or None")
        return value

    @staticmethod
    def _default_registry_actor_name(idempotency_scope: str) -> str:
        scope_hash = hashlib.sha256(idempotency_scope.encode("utf-8")).hexdigest()[:16]
        return f"{DEFAULT_REGISTRY_ACTOR_NAME}_{scope_hash}"

    def __init__(
        self,
        analytics_store: AnalyticsStoreConfig | dict[str, Any],
        address: str | None = None,
        runtime_env: dict[str, Any] | None = None,
        max_retries: int = 0,
        force_reinit: bool = False,
        idempotency_scope: str | None = None,
        registry_actor_name: str | None = None,
        registry_namespace: str = DEFAULT_REGISTRY_NAMESPACE,
        registry_num_cpus: float | None = None,
        registry_memory: float | None = None,
        registry_resources: dict[str, float] | None = None,
        registry_max_pending_calls: int | None = DEFAULT_REGISTRY_MAX_PENDING_CALLS,
        registry_reservation_ttl_s: float = DEFAULT_RESERVATION_TTL_S,
        registry_controller_heartbeat_ttl_s: float = DEFAULT_CONTROLLER_HEARTBEAT_TTL_S,
        registry_sweep_on_submit: bool = True,
        registry_sweep_interval_s: float = DEFAULT_REGISTRY_SWEEP_INTERVAL_S,
        registry_sweep_batch_limit: int | None = DEFAULT_REGISTRY_SWEEP_BATCH_LIMIT,
        registry_update_timeout_s: float = DEFAULT_REGISTRY_UPDATE_TIMEOUT_S,
        controller_actor_prefix: str | None = None,
        controller_heartbeat_interval_s: float = DEFAULT_CONTROLLER_HEARTBEAT_INTERVAL_S,
        controller_terminal_retry_interval_s: float = DEFAULT_CONTROLLER_TERMINAL_RETRY_INTERVAL_S,
        controller_retention_s: float | None = None,
        max_retained_terminal_controllers: int | None = None,
        terminal_job_retention_s: float | None = DEFAULT_TERMINAL_JOB_RETENTION_S,
        max_retained_terminal_jobs_per_scope: int | None = DEFAULT_MAX_RETAINED_TERMINAL_JOBS_PER_SCOPE,
        controller_num_cpus: float | None = None,
        controller_memory: float | None = None,
        controller_resources: dict[str, float] | None = None,
        controller_max_pending_calls: int | None = DEFAULT_CONTROLLER_MAX_PENDING_CALLS,
    ) -> None:
        """Create a Ray job backend and connect it to the shared job registry.

        ``idempotency_scope`` is required so duplicate submissions and reconnects
        are explicit. The constructor initializes Ray when needed, stores backend
        settings, and gets or creates the registry actor used by job handles from
        this job backend.

        Parameters
        ----------
        analytics_store
            Storage config used by worker tasks when they persist completed runs.
            Accepts an ``AnalyticsStoreConfig`` or a matching dict.
        address
            Ray address passed to ``ray.init`` when this job backend initializes Ray.
        runtime_env
            Ray runtime environment passed to ``ray.init`` when this job backend
            initializes Ray.
        max_retries
            Number of retries Ray may use for each capability worker task.
        force_reinit
            Shut down an already initialized local Ray runtime before connecting.
        idempotency_scope
            Required workspace/project namespace for duplicate-submission keys
            and job lookup.
        registry_actor_name
            Name of the shared registry actor. ``None`` derives a stable,
            scope-specific name from ``idempotency_scope`` to avoid accidental
            collisions between unrelated clients on the same Ray cluster.
        registry_namespace
            Ray namespace that contains the registry and controller actors.
        registry_num_cpus
            CPU reservation for the registry actor; ``None`` leaves Ray's actor
            default unchanged.
        registry_memory
            Heap memory reservation for the registry actor; ``None`` leaves it
            unset.
        registry_resources
            Custom Ray resources reserved for the registry actor.
        registry_max_pending_calls
            Maximum queued calls allowed on each registry actor handle. The
            default bounds registry queue growth; ``None`` opts back into Ray's
            unbounded behavior.
        registry_reservation_ttl_s
            How long a new reservation may stay unattached before cleanup can
            fail it.
        registry_controller_heartbeat_ttl_s
            How long a running controller may go without a heartbeat before the
            job can be treated as stale.
        registry_sweep_on_submit
            Whether submit calls should also ask the registry to clean old
            records.
        registry_sweep_interval_s
            Minimum seconds between submit-triggered cleanup attempts.
        registry_sweep_batch_limit
            Maximum records to clean per cleanup call; ``None`` means no explicit
            limit.
        registry_update_timeout_s
            Maximum seconds to wait for one registry or controller bookkeeping
            call.
        controller_actor_prefix
            Prefix for per-job controller actor names.
        controller_heartbeat_interval_s
            How often controllers send heartbeats while jobs are running.
        controller_terminal_retry_interval_s
            How often controllers retry writing a final state to the registry.
        controller_retention_s
            How long terminal controller actors may be kept; ``None`` uses the
            default.
        max_retained_terminal_controllers
            Maximum terminal controller actors kept; ``None`` uses the default.
        terminal_job_retention_s
            How long terminal job records may remain in the registry by age;
            ``None`` disables age-based removal.
        max_retained_terminal_jobs_per_scope
            Maximum terminal job records kept per scope; ``None`` disables this
            cap.
        controller_num_cpus
            CPU reservation for each controller actor; ``None`` uses the default.
        controller_memory
            Memory reservation for each controller actor; ``None`` leaves it
            unset.
        controller_resources
            Custom Ray resources reserved for each controller actor.
        controller_max_pending_calls
            Maximum queued calls allowed on each controller actor handle. The
            default bounds per-controller queue growth; ``None`` opts back into
            Ray's unbounded behavior.
        """
        if not idempotency_scope:
            raise ValueError(
                "idempotency_scope is required for Ray job submission. "
                "Pass a stable workspace/project scope so dedupe and reattach semantics are explicit."
            )
        if registry_actor_name is None:
            registry_actor_name = self._default_registry_actor_name(idempotency_scope)
        elif not registry_actor_name:
            raise ValueError("registry_actor_name must be non-empty when provided")

        (
            registry_update_timeout_s,
            registry_controller_heartbeat_ttl_s,
            controller_heartbeat_interval_s,
            controller_terminal_retry_interval_s,
            registry_sweep_interval_s,
            registry_sweep_batch_limit,
        ) = self._validate_job_tracking_settings(
            registry_update_timeout_s,
            registry_controller_heartbeat_ttl_s,
            controller_heartbeat_interval_s,
            controller_terminal_retry_interval_s,
            registry_sweep_interval_s,
            registry_sweep_batch_limit,
        )

        if force_reinit and ray.is_initialized():
            ray.shutdown()

        if not ray.is_initialized():
            ray.init(address=address, runtime_env=runtime_env)
        elif address is not None or runtime_env is not None:
            logger.warning(
                "Ray runtime already initialized; new address/runtime_env were ignored. "
                "Pass force_reinit=True to reconnect (may interrupt in-flight jobs)."
            )

        if controller_retention_s is None:
            controller_retention_s = DEFAULT_CONTROLLER_RETENTION_S
        if max_retained_terminal_controllers is None:
            max_retained_terminal_controllers = DEFAULT_MAX_RETAINED_TERMINAL_CONTROLLERS
        if controller_num_cpus is None:
            controller_num_cpus = DEFAULT_CONTROLLER_NUM_CPUS

        self._analytics_store = AnalyticsStoreConfig.model_validate(analytics_store)
        self._max_retries = max_retries

        self._idempotency_scope = idempotency_scope
        self._registry_actor_name = registry_actor_name
        self._registry_namespace = registry_namespace
        self._registry_num_cpus = None if registry_num_cpus is None else float(registry_num_cpus)
        self._registry_memory = None if registry_memory is None else float(registry_memory)
        self._registry_resources = registry_resources
        self._registry_max_pending_calls = self._validate_max_pending_calls(
            "registry_max_pending_calls",
            registry_max_pending_calls,
        )
        self._registry_reservation_ttl_s = float(registry_reservation_ttl_s)
        self._registry_controller_heartbeat_ttl_s = registry_controller_heartbeat_ttl_s
        self._registry_sweep_on_submit = bool(registry_sweep_on_submit)
        self._registry_sweep_interval_s = registry_sweep_interval_s
        self._registry_sweep_batch_limit = registry_sweep_batch_limit
        self._last_registry_sweep_ts = 0.0
        self._registry_update_timeout_s = registry_update_timeout_s
        self._controller_actor_prefix = controller_actor_prefix or f"{self._registry_actor_name}_controller"
        self._controller_heartbeat_interval_s = controller_heartbeat_interval_s
        self._controller_terminal_retry_interval_s = controller_terminal_retry_interval_s
        self._controller_retention_s = float(controller_retention_s)
        self._max_retained_terminal_controllers = max_retained_terminal_controllers
        self._terminal_job_retention_s = terminal_job_retention_s
        self._max_retained_terminal_jobs_per_scope = max_retained_terminal_jobs_per_scope
        self._controller_num_cpus = float(controller_num_cpus)
        self._controller_memory = None if controller_memory is None else float(controller_memory)
        self._controller_resources = controller_resources
        self._controller_max_pending_calls = self._validate_max_pending_calls(
            "controller_max_pending_calls",
            controller_max_pending_calls,
        )

        self._registry = get_or_create_registry_actor(
            name=self._registry_actor_name,
            namespace=self._registry_namespace,
            reservation_ttl_s=self._registry_reservation_ttl_s,
            registry_num_cpus=self._registry_num_cpus,
            registry_memory=self._registry_memory,
            registry_resources=self._registry_resources,
            registry_max_pending_calls=self._registry_max_pending_calls,
            controller_heartbeat_ttl_s=self._registry_controller_heartbeat_ttl_s,
            controller_retention_s=self._controller_retention_s,
            max_retained_terminal_controllers=self._max_retained_terminal_controllers,
            terminal_job_retention_s=self._terminal_job_retention_s,
            max_retained_terminal_jobs_per_scope=self._max_retained_terminal_jobs_per_scope,
        )
        self._jobs: dict[str, RayJob] = {}

    @staticmethod
    def _resource_hint(
        *,
        name: str,
        resources: Mapping[str, object],
        config: object | None,
        capability: CapabilityType,
        capability_attr: str,
        default: object,
    ) -> object:
        """Choose one Ray resource quantity from the supported hint locations.

        This lets callers tune worker placement at the most convenient level:
        per-submission ``resources`` for one-off overrides, matching attributes
        on the run config for configuration-driven defaults, or capability-level
        ``default_num_*`` attributes for reusable capability defaults. Explicit
        submit resources win first, then a matching config attribute, then the
        capability's default value, and finally the job backend fallback.
        """
        value = resources.get(name)
        if value is None and config is not None:
            value = getattr(config, name, None)
        if value is None:
            value = getattr(capability, capability_attr, default)
        return value

    @staticmethod
    def _resolve_resources(capability: CapabilityType, run_kwargs: dict[str, Any]) -> RayTaskResources:
        """Build Ray task resources for one capability run.

        CPU and GPU counts come from submit-time resources, config hints, or the
        capability defaults. Custom Ray resources are included when the caller
        provides them.
        """
        resources_raw = run_kwargs.get("resources") or {}
        if not isinstance(resources_raw, Mapping):
            raise TypeError("resources must be a mapping")
        resources = cast(Mapping[str, object], resources_raw)
        config = run_kwargs.get("config")

        resolved_resources: dict[str, object] = dict(resources)
        resolved_resources["num_gpus"] = RayJobBackend._resource_hint(
            name="num_gpus",
            resources=resources,
            config=config,
            capability=capability,
            capability_attr="default_num_gpus",
            default=0.0,
        )
        resolved_resources["num_cpus"] = RayJobBackend._resource_hint(
            name="num_cpus",
            resources=resources,
            config=config,
            capability=capability,
            capability_attr="default_num_cpus",
            default=1.0,
        )
        return RayTaskResources.from_mapping(resolved_resources)

    @staticmethod
    def _compute_scoped_run_key(capability: CapabilityType, run_kwargs: dict[str, Any]) -> str:
        config = run_kwargs.get("config")
        if config is None:
            create_config = getattr(capability, "_create_config", None)
            if create_config is None:
                raise ValueError("Capability must provide config for dedupe key computation")
            config = create_config()

        run_type = getattr(capability, "_RUN_TYPE", None)
        if run_type is None:
            raise ValueError("Capability must define _RUN_TYPE for dedupe key computation")

        return run_type.compute_uid(
            capability_id=capability.id,
            config=config,
            dataset_metadata=[dataset.metadata for dataset in (run_kwargs.get("datasets") or [])],
            model_metadata=[model.metadata for model in (run_kwargs.get("models") or [])],
            metric_metadata=[metric.metadata for metric in (run_kwargs.get("metrics") or [])],
        )

    @staticmethod
    def _format_controller_actor_name(prefix: str, job_id: str) -> str:
        return f"{prefix}_{job_id}"

    def _controller_actor_name(self, job_id: str) -> str:
        return self._format_controller_actor_name(self._controller_actor_prefix, job_id)

    def _job_from_record(self, record: JobRecord) -> RayJob:
        """Build a ``RayJob`` handle from a registry record.

        This does not start work. It creates a local handle that points at the
        existing registry and controller state so callers can poll, cancel, or
        retrieve the result.
        """
        return RayJob(
            registry=self._registry,
            scope=self._idempotency_scope,
            job_id=record["job_id"],
            created_at=_created_at_from_submitted_ts(record["submitted_at_ts"]),
            control_plane_timeout_s=self._registry_update_timeout_s,
            initial_status=_job_status_from_fields(record["status"], record["result_ref"]),
        )

    def _attach_job(self, record: JobRecord) -> RayJob:
        """Return the cached handle for a record, creating it if needed.

        Reusing the same handle keeps the last observed status for this job backend
        instance and avoids creating multiple Python objects for the same job.
        """
        job_id = record["job_id"]
        existing = self._jobs.get(job_id)
        if existing is not None:
            return existing

        job = self._job_from_record(record)
        self._jobs[job_id] = job
        return job

    def _sweep_registry_on_submit(self) -> None:
        """Occasionally ask the registry to clean up old records before submit.

        Cleanup is best-effort and throttled by ``registry_sweep_interval_s``.
        Failures are ignored so stale-record cleanup does not block a user's new
        submission.
        """
        if not self._registry_sweep_on_submit:
            return
        now = time.time()
        if now - self._last_registry_sweep_ts < self._registry_sweep_interval_s:
            return
        self._last_registry_sweep_ts = now
        limit = self._registry_sweep_batch_limit

        with suppress(Exception):
            ray.get(
                self._registry.sweep_expired_submissions.remote(limit=limit),
                timeout=self._registry_update_timeout_s,
            )
        with suppress(Exception):
            ray.get(
                self._registry.sweep_stale_running_jobs.remote(limit=limit),
                timeout=self._registry_update_timeout_s,
            )
        with suppress(Exception):
            ray.get(
                self._registry.sweep_retained_job_records.remote(limit=limit),
                timeout=self._registry_update_timeout_s,
            )
        with suppress(Exception):
            ray.get(
                self._registry.sweep_terminal_controllers.remote(limit=limit),
                timeout=self._registry_update_timeout_s,
            )

    def sweep_registry(self, limit: int | None = DEFAULT_REGISTRY_SWEEP_BATCH_LIMIT) -> dict[str, int]:
        """Run one registry cleanup pass and return per-sweep counts.

        This is a public wrapper around the registry's bounded cleanup methods
        so operators can trigger cleanup from notebooks, admin services, or a
        Kubernetes CronJob without reaching into private backend attributes.
        ``limit`` is applied independently to each sweep. Pass ``None`` to let a
        sweep scan without an explicit batch limit.
        """
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0 or None")

        return {
            "expired_submissions": int(
                _ray_get_with_backpressure(
                    lambda: self._registry.sweep_expired_submissions.remote(limit=limit),
                    timeout_s=self._registry_update_timeout_s,
                    action="sweeping expired submissions",
                )
            ),
            "stale_running_jobs": int(
                _ray_get_with_backpressure(
                    lambda: self._registry.sweep_stale_running_jobs.remote(limit=limit),
                    timeout_s=self._registry_update_timeout_s,
                    action="sweeping stale running jobs",
                )
            ),
            "retained_job_records": int(
                _ray_get_with_backpressure(
                    lambda: self._registry.sweep_retained_job_records.remote(limit=limit),
                    timeout_s=self._registry_update_timeout_s,
                    action="sweeping retained job records",
                )
            ),
            "terminal_controllers": int(
                _ray_get_with_backpressure(
                    lambda: self._registry.sweep_terminal_controllers.remote(limit=limit),
                    timeout_s=self._registry_update_timeout_s,
                    action="sweeping terminal controllers",
                )
            ),
        }

    def _record_for_unattached_controller(self, controller: ActorHandle, job_id: str) -> RayJob:
        """Handle a controller that was created but not attached to the job record.

        If the registry rejects the attach, the reservation may have changed or
        expired. This kills the unused controller and returns a handle for the
        current registry record instead.
        """
        with suppress(Exception):
            ray.kill(controller, no_restart=True)
        record = cast(
            JobRecord | None,
            _ray_get_with_backpressure(
                lambda: self._registry.get_job.remote(self._idempotency_scope, job_id),
                timeout_s=self._registry_update_timeout_s,
                action=f"reading job {job_id!r}",
            ),
        )
        if record is None:
            raise RuntimeError(f"Registry lost job metadata for {job_id}")
        return self._attach_job(record)

    def _start_controller(
        self,
        *,
        controller: ActorHandle,
        capability: CapabilityType,
        run_kwargs: dict[str, Any],
        resources: RayTaskResources,
        job_id: str,
        reservation_token: str,
    ) -> ControllerStatePayload | None:
        """Start the controller and wait briefly for its first state.

        If the controller does not acknowledge the start before the timeout, the
        job may still continue in Ray. In that case this returns ``None`` so the
        caller can still hand back a reattachable job handle.
        """
        try:
            return cast(
                ControllerStatePayload,
                _ray_get_with_backpressure(
                    lambda: controller.start.remote(
                        capability,
                        run_kwargs,
                        resources,
                        self._max_retries,
                        reservation_token,
                    ),
                    timeout_s=self._registry_update_timeout_s,
                    action=f"starting controller for job {job_id!r}",
                ),
            )
        except GetTimeoutError:
            logger.warning(
                "Timed out waiting for Ray job controller start acknowledgement; returning a job handle",
                extra={"job_id": job_id, "scope": self._idempotency_scope},
            )
            return None

    def _close_failed_submission(
        self,
        *,
        attached: bool,
        job_id: str,
        reservation_token: str,
        controller_name: str,
        error: str,
    ) -> None:
        """Mark a failed submit attempt in the registry, best-effort.

        If a controller was already attached, this writes a terminal failure for
        that controller. Otherwise it fails the original reservation. Errors are
        suppressed because this runs while another submission error is already
        being reported.
        """
        with suppress(Exception):
            if attached:
                ray.get(
                    self._registry.update_terminal.remote(
                        self._idempotency_scope,
                        job_id,
                        RegistryStatus.FAILED,
                        error,
                        None,
                        controller_name,
                        reservation_token,
                    ),
                    timeout=self._registry_update_timeout_s,
                )
                return
            ray.get(
                self._registry.fail_submission.remote(
                    self._idempotency_scope,
                    job_id,
                    reservation_token,
                    error,
                ),
                timeout=self._registry_update_timeout_s,
            )

    def _fallback_record_after_submit_timeout(
        self,
        registration: NewJobRegistrationRecord,
        *,
        controller_name: str,
        reservation_token: str,
        start_state: ControllerStatePayload | None,
    ) -> JobRecord:
        """Build a local record when submit succeeded but the registry read timed out.

        The fallback includes the controller name, namespace, and token from the
        reservation so the returned handle can still find the job. The registry
        remains the shared source of truth when it becomes readable again.
        """
        fallback_record = cast(JobRecord, {k: v for k, v in registration.items() if k != "decision"})
        fallback_record["controller_actor_name"] = controller_name
        fallback_record["controller_namespace"] = self._registry_namespace
        fallback_record["controller_token"] = reservation_token
        if start_state is not None:
            fallback_record["status"] = _registry_status_from_raw(start_state["status"])
            fallback_record["result_ref"] = start_state["result_ref"]
            fallback_record["error"] = start_state["error"]
            if start_state["terminal_at_ts"] is not None:
                fallback_record["completed_at_ts"] = float(start_state["terminal_at_ts"])
        return fallback_record

    def _record_after_submit_acceptance(
        self,
        registration: NewJobRegistrationRecord,
        *,
        job_id: str,
        controller_name: str,
        reservation_token: str,
        start_state: ControllerStatePayload | None,
    ) -> JobRecord:
        """Read the registry record after the controller accepts a new job.

        If the registry read times out, this returns a fallback record with enough
        controller information for a useful handle. If the registry has lost the
        record, this raises because there is no shared metadata to attach to.
        """
        try:
            record = cast(
                JobRecord | None,
                _ray_get_with_backpressure(
                    lambda: self._registry.get_job.remote(self._idempotency_scope, job_id),
                    timeout_s=self._registry_update_timeout_s,
                    action=f"reading job {job_id!r}",
                ),
            )
        except GetTimeoutError:
            logger.warning(
                "Timed out reading Ray job registry after controller accepted the job; returning a reattachable handle",
                extra={"job_id": job_id, "scope": self._idempotency_scope},
            )
            return self._fallback_record_after_submit_timeout(
                registration,
                controller_name=controller_name,
                reservation_token=reservation_token,
                start_state=start_state,
            )

        if record is None:
            raise RuntimeError(f"Registry lost job metadata for {job_id}")
        return record

    def submit_capability(self, capability: CapabilityType, **kwargs: Any) -> RayJob:
        """Submit a capability run to Ray and return a job handle.

        The job backend first computes the run UID and asks the shared registry if a
        job for that run already exists in this idempotency scope. If it does,
        no new Ray work is started; the returned ``RayJob`` points at the
        existing job.

        For a new run, the job backend reserves a job ID, resolves worker resources,
        removes backend-only kwargs such as ``resources``, injects the analytics
        store config for the worker, creates the per-job controller actor, and
        asks that controller to start the Ray task. The controller owns the task,
        so the returned handle can be used to poll, cancel, or fetch the result
        even if this submitter process exits.
        """
        run_kwargs = prepare_job_submission_run_kwargs(kwargs)

        self._sweep_registry_on_submit()

        scoped_run_key = self._compute_scoped_run_key(capability, run_kwargs)
        registration = cast(
            JobRegistrationRecord,
            _ray_get_with_backpressure(
                lambda: self._registry.register_or_get.remote(self._idempotency_scope, scoped_run_key),
                timeout_s=self._registry_update_timeout_s,
                action="registering job submission",
            ),
        )

        decision = registration["decision"]
        if decision == "existing":
            return self._attach_job(registration)
        if decision != "new":
            raise RuntimeError(f"Unexpected registry decision {decision!r}")

        new_registration = cast(NewJobRegistrationRecord, registration)
        job_id = new_registration["job_id"]
        reservation_token = new_registration["reservation_token"]
        if reservation_token is None:
            raise RuntimeError(f"Registry returned a new job without reservation token for {job_id}")
        controller: ActorHandle | None = None
        controller_name = self._controller_actor_name(job_id)
        attached = False
        start_state: ControllerStatePayload | None = None

        try:
            resources: RayTaskResources = self._resolve_resources(capability, run_kwargs)
            run_kwargs.pop("resources", None)
            run_kwargs["_analytics_store"] = self._analytics_store.model_dump(mode="python")

            controller = get_or_create_controller_actor(
                name=controller_name,
                namespace=self._registry_namespace,
                registry_name=self._registry_actor_name,
                registry_namespace=self._registry_namespace,
                scope=self._idempotency_scope,
                job_id=job_id,
                registry_update_timeout_s=self._registry_update_timeout_s,
                controller_heartbeat_interval_s=self._controller_heartbeat_interval_s,
                controller_terminal_retry_interval_s=self._controller_terminal_retry_interval_s,
                controller_num_cpus=self._controller_num_cpus,
                controller_memory=self._controller_memory,
                controller_resources=self._controller_resources,
                controller_max_pending_calls=self._controller_max_pending_calls,
                controller_token=reservation_token,
            )

            attached = bool(
                _ray_get_with_backpressure(
                    lambda: self._registry.attach_controller.remote(
                        self._idempotency_scope,
                        job_id,
                        reservation_token,
                        controller_name,
                        self._registry_namespace,
                    ),
                    timeout_s=self._registry_update_timeout_s,
                    action=f"attaching controller for job {job_id!r}",
                )
            )
            if not attached:
                return self._record_for_unattached_controller(controller, job_id)

            start_state = self._start_controller(
                controller=controller,
                capability=capability,
                run_kwargs=run_kwargs,
                resources=resources,
                job_id=job_id,
                reservation_token=reservation_token,
            )
        except BackpressureError:
            if controller is not None and not attached:
                with suppress(Exception):
                    ray.kill(controller, no_restart=True)
            raise
        except Exception as exc:
            self._close_failed_submission(
                attached=attached,
                job_id=job_id,
                reservation_token=reservation_token,
                controller_name=controller_name,
                error=str(exc),
            )
            if controller is not None and not attached:
                with suppress(Exception):
                    ray.kill(controller, no_restart=True)
            raise RuntimeError(f"Failed to submit capability job {job_id}: {exc}") from exc

        if (
            start_state is not None
            and _job_status_from_fields(_registry_status_from_raw(start_state["status"]), start_state["result_ref"])
            is JobStatus.FAILED
        ):
            raise RuntimeError(
                f"Failed to submit capability job {job_id}: {start_state['error'] or 'job failed during start'}"
            )

        record = self._record_after_submit_acceptance(
            new_registration,
            job_id=job_id,
            controller_name=controller_name,
            reservation_token=reservation_token,
            start_state=start_state,
        )
        return self._attach_job(record)

    @staticmethod
    def _registry_status_filter(
        status_filter: JobStatus | Sequence[JobStatus] | None,
    ) -> RegistryStatus | list[RegistryStatus] | None:
        """Convert public ``JobStatus`` filters to registry status values.

        The registry stores a few internal states that are more detailed than the
        public API, such as ``SUBMITTING`` and ``CANCELLING``. This maps public
        filters to the matching registry values before ``list_jobs()`` reads from
        the registry.
        """
        if status_filter is None:
            return None

        def values_for(status: JobStatus) -> list[RegistryStatus]:
            if not isinstance(status, JobStatus):
                raise TypeError("status_filter must be a JobStatus or a sequence of JobStatus values")
            if status is JobStatus.PENDING:
                return [RegistryStatus.SUBMITTING]
            if status is JobStatus.RUNNING:
                return [RegistryStatus.RUNNING, RegistryStatus.CANCELLING]
            return [RegistryStatus[status.name]]

        if isinstance(status_filter, JobStatus):
            values = values_for(status_filter)
        else:
            try:
                statuses = set(status_filter)
            except TypeError as exc:
                raise TypeError("status_filter must be a JobStatus or a sequence of JobStatus values") from exc
            if not all(isinstance(status, JobStatus) for status in statuses):
                raise TypeError("status_filter must be a JobStatus or a sequence of JobStatus values")
            values = []
            for status in statuses:
                values.extend(values_for(status))

        values = list(dict.fromkeys(values))
        return values[0] if len(values) == 1 else values

    def list_jobs(
        self,
        limit: int | None = DEFAULT_LIST_JOBS_LIMIT,
        status_filter: JobStatus | Sequence[JobStatus] | None = None,
        submitted_before_ts: float | None = None,
    ) -> list[RayJob]:
        """List jobs in this job backend's scope.

        The registry returns records, optionally limited or filtered by status and
        submission time. Each record is converted to a ``RayJob`` handle that can
        be polled, cancelled, or used to fetch the result.
        """
        raw_status_filter = self._registry_status_filter(status_filter)

        records = cast(
            list[JobRecord],
            _ray_get_with_backpressure(
                lambda: self._registry.list_jobs.remote(
                    self._idempotency_scope,
                    limit,
                    raw_status_filter,
                    submitted_before_ts,
                ),
                timeout_s=self._registry_update_timeout_s,
                action="listing jobs",
            ),
        )
        return [self._attach_job(record) for record in records]

    def get_job(self, job_id: str) -> RayJob:
        """Return a ``RayJob`` handle for an existing job ID in this scope.

        Raises ``KeyError`` when the registry has no record for the requested job.
        """
        record = cast(
            JobRecord | None,
            _ray_get_with_backpressure(
                lambda: self._registry.get_job.remote(self._idempotency_scope, job_id),
                timeout_s=self._registry_update_timeout_s,
                action=f"reading job {job_id!r}",
            ),
        )
        if record is None:
            raise KeyError(f"No job with ID {job_id!r}")
        return self._attach_job(record)

    def shutdown(self, wait: bool = True) -> None:
        """Wait for known jobs if requested, then shut down the local Ray runtime.

        Only jobs returned by this job backend instance are waited on. When ``wait``
        is ``False``, this returns immediately and does not call ``ray.shutdown()``.
        """
        if not wait:
            return

        for job in self._jobs.values():
            with suppress(Exception):
                job.wait(timeout=None)

        if ray.is_initialized():
            ray.shutdown()
