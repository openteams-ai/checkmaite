from __future__ import annotations

import copy
import enum
import heapq
import time
import uuid
from typing import Any, Literal, TypeAlias, TypedDict, cast

import ray
from ray.actor import ActorHandle

from checkmaite.jobs.protocol import CapabilityRunRef, CapabilityRunRefPayload


class RegistryStatus(str, enum.Enum):
    """Internal registry lifecycle states stored as plain string values."""

    SUBMITTING = "SUBMITTING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


REGISTRY_TERMINAL_STATUSES = {
    RegistryStatus.COMPLETED,
    RegistryStatus.FAILED,
    RegistryStatus.CANCELLED,
}

# COMPLETED keeps the dedupe key so repeated submissions return the canonical
# completed job. FAILED and CANCELLED release the dedupe key so users can retry
# the same logical work in the same scope.
REGISTRY_DEDUPE_RELEASE_STATUSES = {
    RegistryStatus.FAILED,
    RegistryStatus.CANCELLED,
}


class JobRecord(TypedDict):
    """Registry metadata record for one Ray-backed job.

    Fields
    ------
    scope
        Idempotency scope that owns the job.
    scoped_run_key
        Dedupe key for the logical capability run within ``scope``.
    job_id
        Unique registry job ID.
    status
        Internal lifecycle state stored by the registry.
    controller_actor_name
        Name of the detached controller actor for this job, if attached.
    controller_namespace
        Ray namespace where the controller actor lives.
    controller_token
        Token used by the controller to prove it owns the job.
    controller_created_at_ts
        Unix timestamp when the controller was attached.
    controller_heartbeat_at_ts
        Unix timestamp of the latest controller heartbeat.
    controller_lease_expires_at_ts
        Unix timestamp after which a live controller is considered stale.
    controller_retain_until_ts
        Unix timestamp until which a terminal controller may be kept alive.
    controller_cleaned_at_ts
        Unix timestamp when registry cleanup killed or forgot the controller.
    cancellation_requested_at_ts
        Unix timestamp when cancellation was requested, if any.
    result_ref
        Serialized ``CapabilityRunRef`` for a completed job.
    error
        Error message for a failed job.
    submitted_at_ts
        Unix timestamp when the job reservation was created.
    completed_at_ts
        Unix timestamp when the job reached a terminal state.
    reservation_token
        Submitter token for the initial ``SUBMITTING`` reservation.
    reservation_expires_at_ts
        Unix timestamp after which an unstarted reservation can be expired.
    """

    scope: str
    scoped_run_key: str
    job_id: str
    status: RegistryStatus
    controller_actor_name: str | None
    controller_namespace: str | None
    controller_token: str | None
    controller_created_at_ts: float | None
    controller_heartbeat_at_ts: float | None
    controller_lease_expires_at_ts: float | None
    controller_retain_until_ts: float | None
    controller_cleaned_at_ts: float | None
    cancellation_requested_at_ts: float | None
    result_ref: CapabilityRunRefPayload | None
    error: str | None
    submitted_at_ts: float
    completed_at_ts: float | None
    reservation_token: str | None
    reservation_expires_at_ts: float | None


class ExistingJobRegistrationRecord(JobRecord):
    """Registration payload when a matching job already exists."""

    decision: Literal["existing"]


class NewJobRegistrationRecord(JobRecord):
    """Registration payload when a new reservation was created."""

    decision: Literal["new"]


JobRegistrationRecord: TypeAlias = ExistingJobRegistrationRecord | NewJobRegistrationRecord


DEFAULT_REGISTRY_ACTOR_NAME = "checkmaite_job_registry"
DEFAULT_REGISTRY_NAMESPACE = "checkmaite_jobs"
DEFAULT_RESERVATION_TTL_S = 60.0
DEFAULT_CONTROLLER_HEARTBEAT_TTL_S = 120.0
DEFAULT_CONTROLLER_RETENTION_S = 3600.0
DEFAULT_MAX_RETAINED_TERMINAL_CONTROLLERS = 1000
DEFAULT_TERMINAL_JOB_RETENTION_S = 7 * 24 * 60 * 60.0
DEFAULT_MAX_RETAINED_TERMINAL_JOBS_PER_SCOPE = 10000
DEFAULT_LIST_JOBS_LIMIT = 100
DEFAULT_REGISTRY_SWEEP_INTERVAL_S = 30.0
DEFAULT_REGISTRY_SWEEP_BATCH_LIMIT = 100


def _coerce_registry_status(status: RegistryStatus | str) -> RegistryStatus:
    if isinstance(status, RegistryStatus):
        return status
    try:
        return RegistryStatus(status.upper())
    except ValueError as exc:
        raise ValueError(f"Invalid registry status {status!r}") from exc


class JobRegistry:
    """Registry state machine for job identity, dedupe, and small metadata.

    The registry is deliberately a directory, not the live task owner. It keeps
    two indexes:

    - ``(scope, scoped_run_key) -> job_id`` for duplicate-submit suppression.
    - ``(scope, job_id) -> record`` for reattach/list/status lookup.

    Live Ray tasks are owned by detached per-job controller actors. Registry
    records store the controller actor name plus small lifecycle metadata and a
    small terminal ``CapabilityRunRef`` payload. They must not store datasets,
    models, logs, full run objects, or unbounded history.
    """

    def __init__(
        self,
        reservation_ttl_s: float = DEFAULT_RESERVATION_TTL_S,
        controller_heartbeat_ttl_s: float = DEFAULT_CONTROLLER_HEARTBEAT_TTL_S,
        controller_retention_s: float = DEFAULT_CONTROLLER_RETENTION_S,
        max_retained_terminal_controllers: int | None = DEFAULT_MAX_RETAINED_TERMINAL_CONTROLLERS,
        terminal_job_retention_s: float | None = DEFAULT_TERMINAL_JOB_RETENTION_S,
        max_retained_terminal_jobs_per_scope: int | None = DEFAULT_MAX_RETAINED_TERMINAL_JOBS_PER_SCOPE,
    ) -> None:
        """Create an empty registry and configure cleanup timeouts.

        Parameters
        ----------
        reservation_ttl_s
            How long a new ``SUBMITTING`` reservation may sit without a
            controller starting it before the registry can mark it failed.
        controller_heartbeat_ttl_s
            How long a ``RUNNING`` job may go without a controller heartbeat
            before the controller is treated as stale.
        controller_retention_s
            How long to keep a terminal controller actor around for reattach or
            late terminal-state retries before cleanup may kill it.
        max_retained_terminal_controllers
            Maximum number of terminal controller actors to keep. ``None`` means
            no count-based limit.
        terminal_job_retention_s
            How long terminal job records stay in the registry. ``None`` means
            keep terminal records indefinitely.
        max_retained_terminal_jobs_per_scope
            Maximum number of terminal job records to keep per scope. ``None``
            means no count-based limit.
        """
        self._reservation_ttl_s = float(reservation_ttl_s)
        self._controller_heartbeat_ttl_s = float(controller_heartbeat_ttl_s)
        self._controller_retention_s = float(controller_retention_s)
        self._max_retained_terminal_controllers = max_retained_terminal_controllers
        self._terminal_job_retention_s = None if terminal_job_retention_s is None else float(terminal_job_retention_s)
        self._max_retained_terminal_jobs_per_scope = max_retained_terminal_jobs_per_scope

        self._dedupe_index: dict[tuple[str, str], str] = {}
        self._job_index: dict[tuple[str, str], JobRecord] = {}

    @staticmethod
    def _copy_record(record: JobRecord) -> JobRecord:
        return cast(JobRecord, copy.deepcopy(record))

    @staticmethod
    def _copy_existing_registration(record: JobRecord) -> ExistingJobRegistrationRecord:
        out = cast(ExistingJobRegistrationRecord, JobRegistry._copy_record(record))
        out["decision"] = "existing"
        return out

    @staticmethod
    def _copy_new_registration(record: JobRecord) -> NewJobRegistrationRecord:
        reservation_token = record.get("reservation_token")
        if reservation_token is None:
            raise RuntimeError("new registry record missing reservation_token")
        out = cast(NewJobRegistrationRecord, JobRegistry._copy_record(record))
        out["decision"] = "new"
        out["reservation_token"] = reservation_token
        return out

    @staticmethod
    def _record_key(record: JobRecord) -> tuple[str, str]:
        return str(record["scope"]), str(record["job_id"])

    @staticmethod
    def _submitted_record_key(record: JobRecord) -> tuple[float, str]:
        return float(record["submitted_at_ts"]), str(record["job_id"])

    @staticmethod
    def _completed_record_key(record: JobRecord) -> tuple[float, str]:
        return float(record.get("completed_at_ts") or 0.0), str(record["job_id"])

    @staticmethod
    def _controller_update_matches(
        record: JobRecord,
        controller_actor_name: str | None,
        controller_token: str | None,
    ) -> bool:
        """Return whether a controller update is allowed for this record.

        Before a controller is attached, the submitter may update the reserved
        record only by presenting the reservation token and no controller
        identity. After attachment, updates must come
        from the recorded controller actor and include the matching controller
        token. This prevents stale or duplicate controllers from overwriting the
        current job state.
        """
        expected_controller = record.get("controller_actor_name")
        expected_token = record.get("controller_token")
        if expected_controller is None and expected_token is None:
            reservation_token = record.get("reservation_token")
            return (
                reservation_token is not None
                and controller_actor_name is None
                and controller_token == reservation_token
            )
        return controller_actor_name == expected_controller and controller_token == expected_token

    @staticmethod
    def _normalize_terminal_update(
        status: RegistryStatus | str,
        error: str | None,
        result_ref: CapabilityRunRefPayload | None,
    ) -> tuple[RegistryStatus, str | None, CapabilityRunRefPayload | None]:
        """Validate and clean a requested terminal status update.

        Only terminal statuses are accepted. Completed jobs must include a valid
        ``CapabilityRunRef`` payload; otherwise the update is converted to
        ``FAILED`` so clients never see a completed job with a broken result.
        Failed or cancelled jobs do not keep a result payload.
        """
        status = _coerce_registry_status(status)
        if status not in REGISTRY_TERMINAL_STATUSES:
            raise ValueError(f"Invalid terminal status {status.value!r}")
        if status is not RegistryStatus.COMPLETED:
            return status, error, None
        if result_ref is None:
            return RegistryStatus.FAILED, error or "completed job missing result_ref", None
        try:
            CapabilityRunRef.model_validate(result_ref)
        except Exception as exc:  # noqa: BLE001
            return RegistryStatus.FAILED, f"completed job has invalid result_ref: {exc}", None
        return status, error, result_ref

    def _release_dedupe_for_record(self, record: JobRecord) -> None:
        """Remove this job from the dedupe index when retries should be allowed.

        Failed and cancelled jobs should not permanently claim their run key;
        users need to be able to submit the same logical work again. Completed
        jobs keep the dedupe entry so duplicate submissions return the canonical
        completed job instead of running the work again.
        """
        dedupe_key = (str(record["scope"]), str(record["scoped_run_key"]))
        if self._dedupe_index.get(dedupe_key) == record["job_id"]:
            del self._dedupe_index[dedupe_key]

    def _purge_record(self, record: JobRecord, now: float) -> bool:
        """Delete one terminal job record if its controller can be cleaned up.

        Registry cleanup uses this after a terminal record is old enough or over
        a retention limit. Active jobs are never purged. If the record still has
        a detached controller actor, the registry tries to kill or forget that
        actor first; if that cleanup cannot be confirmed, the record is kept so
        a later sweep can retry. Purging also releases any dedupe entry so the
        registry does not point at a deleted job.
        """
        if record["status"] not in REGISTRY_TERMINAL_STATUSES:
            return False
        if record.get("controller_actor_name") and not self._cleanup_controller_for_record(record, now):
            return False
        self._release_dedupe_for_record(record)
        self._job_index.pop((str(record["scope"]), str(record["job_id"])), None)
        return True

    def _new_record(self, scope: str, scoped_run_key: str) -> JobRecord:
        now = time.time()
        return {
            "scope": scope,
            "scoped_run_key": scoped_run_key,
            "job_id": uuid.uuid4().hex,
            "status": RegistryStatus.SUBMITTING,
            "controller_actor_name": None,
            "controller_namespace": None,
            "controller_token": None,
            "controller_created_at_ts": None,
            "controller_heartbeat_at_ts": None,
            "controller_lease_expires_at_ts": None,
            "controller_retain_until_ts": None,
            "controller_cleaned_at_ts": None,
            "cancellation_requested_at_ts": None,
            "result_ref": None,
            "error": None,
            "submitted_at_ts": now,
            "completed_at_ts": None,
            "reservation_token": uuid.uuid4().hex,
            "reservation_expires_at_ts": now + self._reservation_ttl_s,
        }

    def _cleanup_controller_for_record(self, record: JobRecord, now: float) -> bool:
        """Kill or forget the detached controller stored on a record.

        Terminal jobs may leave behind detached controller actors for reattach or
        retry purposes. Cleanup first looks up the actor. If Ray says the actor
        no longer exists, the registry clears the controller metadata. If the
        actor exists, the registry kills it and then clears the metadata. Unknown
        Ray errors are treated as temporary failures so a later sweep can retry.
        """
        actor_name = record.get("controller_actor_name")
        if not actor_name:
            return False

        def clear_metadata() -> None:
            record["controller_actor_name"] = None
            record["controller_namespace"] = None
            record["controller_cleaned_at_ts"] = now

        namespace = record.get("controller_namespace")
        try:
            actor = ray.get_actor(str(actor_name), namespace=str(namespace) if namespace else None)
        except ValueError:
            clear_metadata()
            return True
        except Exception:  # noqa: BLE001
            # Unknown Ray/GCS errors may be transient. Keep actor metadata so a
            # later cleanup pass can retry instead of leaking a detached actor.
            return False

        try:
            ray.kill(actor, no_restart=True)
        except Exception:  # noqa: BLE001
            return False

        clear_metadata()
        return True

    def _expire_if_needed(self, record: JobRecord, now: float | None = None) -> bool:
        """Fail a stale ``SUBMITTING`` reservation once its start window expires.

        A new submission first creates a reservation, then attaches a controller
        and starts the worker. If the submitter crashes or cannot finish that
        handoff before ``reservation_expires_at_ts``, the reservation should not
        keep blocking the same run key forever. Expiring it marks the job failed
        and releases dedupe so a later submit can retry.
        """
        if record["status"] != RegistryStatus.SUBMITTING:
            return False
        expires_at = record.get("reservation_expires_at_ts")
        if expires_at is None:
            return False
        now = time.time() if now is None else now
        if float(expires_at) > now:
            return False
        self._expire_submission(record, now)
        return True

    def _expire_submission(self, record: JobRecord, now: float) -> None:
        """Mark an unstarted submission reservation as failed and retryable.

        This is the state change used by reservation expiry. The job never made
        it to ``RUNNING``, so the registry clears the reservation token, records
        a failure, releases the dedupe key, and cleans up any controller that may
        have been attached during a partial submit.
        """
        record["status"] = RegistryStatus.FAILED
        record["error"] = "submission reservation expired"
        record["completed_at_ts"] = now
        record["reservation_token"] = None
        record["reservation_expires_at_ts"] = None
        self._release_dedupe_for_record(record)
        self._cleanup_controller_for_record(record, now)

    def _cancel_submission(self, record: JobRecord, now: float) -> None:
        """Mark a job as cancelled before or during controller-owned work.

        Cancellation is terminal and retryable, so the result payload is cleared
        and the dedupe key is released. If a controller is attached, its metadata
        is kept with a retention deadline so clients can still reattach briefly
        and the controller can finish publishing terminal state.
        """
        record["status"] = RegistryStatus.CANCELLED
        record["error"] = None
        record["completed_at_ts"] = now
        record["reservation_token"] = None
        record["reservation_expires_at_ts"] = None
        record["controller_lease_expires_at_ts"] = None
        record["cancellation_requested_at_ts"] = record.get("cancellation_requested_at_ts") or now
        record["result_ref"] = None
        if record.get("controller_actor_name") is not None and record.get("controller_retain_until_ts") is None:
            record["controller_retain_until_ts"] = now + self._controller_retention_s
        self._release_dedupe_for_record(record)

    def _fail_stale_running_record(self, record: JobRecord, now: float) -> bool:
        """Fail a running job whose controller lease has expired.

        Running jobs depend on their controller actor to watch the worker task
        and publish the final result. If the controller stops heartbeating past
        its lease deadline, the registry treats the job as failed, releases the
        dedupe key for retry, and keeps controller metadata only long enough for
        cleanup.
        """
        if record["status"] not in {RegistryStatus.RUNNING, RegistryStatus.CANCELLING}:
            return False
        lease_expires_at = record.get("controller_lease_expires_at_ts")
        if lease_expires_at is None or float(lease_expires_at) > now:
            return False

        record["status"] = RegistryStatus.FAILED
        record["error"] = "job controller heartbeat expired"
        record["completed_at_ts"] = now
        record["reservation_token"] = None
        record["reservation_expires_at_ts"] = None
        record["controller_lease_expires_at_ts"] = None
        record["result_ref"] = None
        self._release_dedupe_for_record(record)
        self._cleanup_controller_for_record(record, now)
        return True

    def register_or_get(self, scope: str, scoped_run_key: str) -> JobRegistrationRecord:
        """Atomically dedupe submission and create a reservation if needed."""
        dedupe_key = (scope, scoped_run_key)
        existing_job_id = self._dedupe_index.get(dedupe_key)

        if existing_job_id is not None:
            existing = self._job_index.get((scope, existing_job_id))
            if existing is not None:
                now = time.time()
                if not self._expire_if_needed(existing, now) and not self._fail_stale_running_record(existing, now):
                    if existing["status"] in REGISTRY_DEDUPE_RELEASE_STATUSES:
                        self._release_dedupe_for_record(existing)
                    else:
                        return self._copy_existing_registration(existing)
            else:
                del self._dedupe_index[dedupe_key]

        record = self._new_record(scope, scoped_run_key)
        self._dedupe_index[dedupe_key] = record["job_id"]
        self._job_index[(scope, record["job_id"])] = record

        return self._copy_new_registration(record)

    def attach_controller(
        self,
        scope: str,
        job_id: str,
        reservation_token: str,
        controller_actor_name: str,
        controller_namespace: str,
    ) -> bool:
        """Record which detached controller owns a reserved job.

        Submission happens in steps: first the registry reserves a job ID, then
        the job backend creates a detached per-job controller actor, then that actor
        starts the worker task. This method links the reservation to the
        controller before any worker work begins. That link lets later clients
        find the controller for polling or cancellation, and it gives the
        registry a controller name/token to reject stale or duplicate actors.

        The attach only succeeds while the reservation is still ``SUBMITTING``
        and the reservation token still matches. If the reservation expired or
        already reached a terminal state, ``False`` is returned so the caller can
        avoid starting work for a dead reservation.
        """
        record = self._job_index.get((scope, job_id))
        if record is None:
            raise KeyError(f"Unknown job_id {job_id!r} in scope {scope!r}")

        if self._expire_if_needed(record):
            return False
        if record["status"] in REGISTRY_TERMINAL_STATUSES:
            return False
        if record["status"] != RegistryStatus.SUBMITTING:
            raise ValueError(f"Cannot attach controller for job {job_id!r} while status={record['status']!r}")
        if record["reservation_token"] != reservation_token:
            raise ValueError(f"Invalid reservation token for job {job_id!r}")

        existing_controller = record.get("controller_actor_name")
        existing_namespace = record.get("controller_namespace")
        if existing_controller is not None and (
            existing_controller != controller_actor_name or existing_namespace != controller_namespace
        ):
            raise ValueError(f"Job {job_id!r} already has controller {existing_controller!r}")

        record["controller_actor_name"] = controller_actor_name
        record["controller_namespace"] = controller_namespace
        record["controller_token"] = reservation_token
        record["controller_created_at_ts"] = time.time()
        return True

    def _validate_mark_running_owner(
        self,
        record: JobRecord,
        job_id: str,
        reservation_token: str,
        controller_actor_name: str,
        controller_namespace: str,
    ) -> None:
        """Ensure the controller starting work is the one attached to the job."""
        if (
            record.get("controller_actor_name") != controller_actor_name
            or record.get("controller_namespace") != controller_namespace
        ):
            raise ValueError(f"Job {job_id!r} is owned by another controller")
        if record.get("controller_token") != reservation_token:
            raise ValueError(f"Invalid controller token for job {job_id!r}")

    def _mark_running_action(self, record: JobRecord, job_id: str, reservation_token: str) -> str:
        """Decide what should happen before changing a reservation to running.

        The result is a small action string for ``mark_running``: the job may
        already be running, may need to be cancelled because cancellation was
        requested before launch, or may be ready to start. Invalid states or
        tokens raise because they indicate a stale or incorrect caller.
        """
        if record["status"] == RegistryStatus.RUNNING:
            return "already_running"
        if record["status"] != RegistryStatus.SUBMITTING:
            raise ValueError(f"Cannot mark job {job_id!r} running while status={record['status']!r}")
        if record["reservation_token"] != reservation_token:
            raise ValueError(f"Invalid reservation token for job {job_id!r}")
        if record.get("cancellation_requested_at_ts") is not None:
            self._cancel_submission(record, time.time())
            return "cancelled"
        return "start"

    def _mark_record_running(self, record: JobRecord, controller_actor_name: str, controller_namespace: str) -> None:
        """Apply the registry field updates for a job that is now running."""
        now = time.time()
        record["status"] = RegistryStatus.RUNNING
        record["controller_actor_name"] = controller_actor_name
        record["controller_namespace"] = controller_namespace
        if record.get("controller_created_at_ts") is None:
            record["controller_created_at_ts"] = now
        record["controller_heartbeat_at_ts"] = now
        record["controller_lease_expires_at_ts"] = now + self._controller_heartbeat_ttl_s
        record["reservation_token"] = None
        record["reservation_expires_at_ts"] = None

    def mark_running(
        self,
        scope: str,
        job_id: str,
        reservation_token: str,
        controller_actor_name: str,
        controller_namespace: str,
    ) -> bool:
        """Move a reserved job to ``RUNNING`` before worker launch.

        The controller calls this immediately before starting the Ray worker
        task. This is the last registry gate that prevents work from starting
        for an expired, cancelled, terminal, or wrongly-owned reservation. A
        ``False`` return means the controller must not launch the worker.
        """
        record = self._job_index.get((scope, job_id))
        if record is None:
            raise KeyError(f"Unknown job_id {job_id!r} in scope {scope!r}")

        if self._expire_if_needed(record) or record["status"] in REGISTRY_TERMINAL_STATUSES:
            return False

        self._validate_mark_running_owner(
            record, job_id, reservation_token, controller_actor_name, controller_namespace
        )
        action = self._mark_running_action(record, job_id, reservation_token)
        if action == "already_running":
            return True
        if action == "cancelled":
            return False

        self._mark_record_running(record, controller_actor_name, controller_namespace)
        return True

    def heartbeat_controller(
        self,
        scope: str,
        job_id: str,
        controller_actor_name: str,
        controller_token: str,
        now_ts: float | None = None,
    ) -> bool:
        """Refresh the controller lease for a running or cancelling job.

        The controller calls this periodically while it owns the worker task. The
        registry uses the refreshed lease deadline to distinguish a live
        controller from one that crashed or stopped heartbeating. Heartbeats from
        the wrong controller, with the wrong token, or for non-live jobs are
        rejected with ``False``.
        """
        record = self._job_index.get((scope, job_id))
        if record is None or record["status"] in REGISTRY_TERMINAL_STATUSES:
            return False
        if record.get("controller_actor_name") != controller_actor_name:
            return False
        if record.get("controller_token") != controller_token:
            return False
        if record["status"] not in {RegistryStatus.RUNNING, RegistryStatus.CANCELLING}:
            return False

        now = float(time.time() if now_ts is None else now_ts)
        record["controller_heartbeat_at_ts"] = now
        record["controller_lease_expires_at_ts"] = now + self._controller_heartbeat_ttl_s
        return True

    def request_cancellation(
        self,
        scope: str,
        job_id: str,
        controller_actor_name: str | None = None,
        controller_token: str | None = None,
    ) -> JobRecord | None:
        """Record that a caller wants this job cancelled.

        For a still-reserved job, cancellation is immediate because no worker has
        started, but the caller must present the reservation token. For a running
        job, the registry moves to ``CANCELLING`` and the controller is
        responsible for cancelling the Ray task and later writing a terminal
        state. The controller identity is checked when one is already attached,
        so stale callers cannot cancel someone else's job.
        """
        record = self._job_index.get((scope, job_id))
        if record is None:
            return None
        if self._expire_if_needed(record):
            return self._copy_record(record)
        if record["status"] in REGISTRY_TERMINAL_STATUSES:
            return self._copy_record(record)

        now = time.time()
        if not self._controller_update_matches(record, controller_actor_name, controller_token):
            return self._copy_record(record)

        if record["status"] == RegistryStatus.SUBMITTING:
            self._cancel_submission(record, now)
            return self._copy_record(record)

        if record["status"] == RegistryStatus.RUNNING:
            record["status"] = RegistryStatus.CANCELLING
        record["cancellation_requested_at_ts"] = now
        return self._copy_record(record)

    def update_terminal(
        self,
        scope: str,
        job_id: str,
        status: RegistryStatus | str,
        error: str | None = None,
        result_ref: CapabilityRunRefPayload | None = None,
        controller_actor_name: str | None = None,
        controller_token: str | None = None,
    ) -> bool:
        """Write the final state for a job.

        Terminal updates are accepted only once and only from the current
        controller or valid reservation owner. Completed jobs must include a
        valid result reference; failed and cancelled jobs store only status and
        error metadata. Failed and cancelled jobs also release the dedupe key so
        the same logical run can be submitted again.
        """
        status, error, result_ref = self._normalize_terminal_update(status, error, result_ref)

        record = self._job_index.get((scope, job_id))
        if record is None:
            return False
        if record["status"] in REGISTRY_TERMINAL_STATUSES:
            return record["status"] == status

        if not self._controller_update_matches(record, controller_actor_name, controller_token):
            return False

        now = time.time()
        record["status"] = status
        record["error"] = error
        record["completed_at_ts"] = now
        record["reservation_token"] = None
        record["reservation_expires_at_ts"] = None
        record["controller_lease_expires_at_ts"] = None
        record["result_ref"] = None
        if status == RegistryStatus.COMPLETED and result_ref is not None:
            record["result_ref"] = cast(CapabilityRunRefPayload, copy.deepcopy(result_ref))
        if record.get("controller_actor_name") is not None and record.get("controller_retain_until_ts") is None:
            record["controller_retain_until_ts"] = now + self._controller_retention_s

        if status in REGISTRY_DEDUPE_RELEASE_STATUSES:
            self._release_dedupe_for_record(record)
        return True

    def fail_submission(self, scope: str, job_id: str, reservation_token: str, error: str) -> None:
        """Fail a reservation when submission cannot finish.

        The job backend calls this when it reserved a job but failed before the
        controller successfully started work, for example because resource
        validation, actor creation, serialization, or attach/start failed. The
        reservation token must match so another submitter cannot close the wrong
        job.
        """
        record = self._job_index.get((scope, job_id))
        if record is None or record["status"] in REGISTRY_TERMINAL_STATUSES:
            return
        if record["status"] != RegistryStatus.SUBMITTING:
            return
        if record["reservation_token"] != reservation_token:
            return
        self.update_terminal(
            scope,
            job_id,
            RegistryStatus.FAILED,
            error=error,
            controller_actor_name=record.get("controller_actor_name"),
            controller_token=record.get("controller_token") or reservation_token,
        )

    def mark_controller_unavailable(
        self,
        scope: str,
        job_id: str,
        controller_actor_name: str | None,
        controller_token: str | None,
        error: str = "job controller unavailable",
    ) -> JobRecord | None:
        """Fail a non-terminal job when its controller cannot be reached.

        The caller must prove ownership with the current controller name and
        token so a stale observation about an old controller cannot fail a job
        now owned by a different one.
        """
        record = self._job_index.get((scope, job_id))
        if record is None:
            return None
        if record["status"] in REGISTRY_TERMINAL_STATUSES:
            return self._copy_record(record)
        if record.get("controller_actor_name") != controller_actor_name:
            return self._copy_record(record)
        expected_token = record.get("controller_token")
        if expected_token is None or expected_token != controller_token:
            return self._copy_record(record)
        self.update_terminal(
            scope,
            job_id,
            RegistryStatus.FAILED,
            error=error,
            controller_actor_name=controller_actor_name,
            controller_token=controller_token,
        )
        return self._copy_record(record)

    def get_job(self, scope: str, job_id: str) -> JobRecord | None:
        """Return one job record after applying lazy timeout checks.

        Reads also perform cheap maintenance: expired ``SUBMITTING`` reservations
        are failed, and running jobs with expired controller leases are failed.
        A deep copy is returned so callers cannot mutate registry state.
        """
        record = self._job_index.get((scope, job_id))
        if record is None:
            return None
        now = time.time()
        self._expire_if_needed(record, now)
        self._fail_stale_running_record(record, now)
        return self._copy_record(record)

    def list_jobs(
        self,
        scope: str,
        limit: int | None = DEFAULT_LIST_JOBS_LIMIT,
        status_filter: RegistryStatus | str | list[RegistryStatus | str] | None = None,
        before_submitted_at_ts: float | None = None,
    ) -> list[JobRecord]:
        wanted_statuses: set[RegistryStatus] | None
        if status_filter is None:
            wanted_statuses = None
        elif isinstance(status_filter, (RegistryStatus, str)):
            wanted_statuses = {_coerce_registry_status(status_filter)}
        else:
            wanted_statuses = {_coerce_registry_status(status) for status in status_filter}

        now = time.time()
        candidates: list[JobRecord] = []
        for (row_scope, _job_id), record in self._job_index.items():
            if row_scope != scope:
                continue
            self._expire_if_needed(record, now)
            self._fail_stale_running_record(record, now)
            if wanted_statuses is not None and record["status"] not in wanted_statuses:
                continue
            if before_submitted_at_ts is not None and float(record["submitted_at_ts"]) >= before_submitted_at_ts:
                continue
            candidates.append(record)

        if limit is None:
            selected = sorted(candidates, key=self._submitted_record_key, reverse=True)
        else:
            selected = heapq.nlargest(max(0, int(limit)), candidates, key=self._submitted_record_key)
        return [self._copy_record(record) for record in selected]

    def sweep_expired_submissions(self, now_ts: float | None = None, limit: int | None = None) -> int:
        """Expire old ``SUBMITTING`` reservations in a bounded cleanup pass.

        This handles submitters that reserved a job ID but crashed or failed
        before starting a controller. Each expired reservation is marked failed
        and releases its dedupe key so the same run can be submitted again.
        ``limit`` caps the amount of cleanup done in one call.
        """
        now = float(time.time() if now_ts is None else now_ts)
        swept = 0
        for record in self._job_index.values():
            if limit is not None and swept >= limit:
                break
            if record["status"] != RegistryStatus.SUBMITTING:
                continue
            if not self._expire_if_needed(record, now):
                continue
            swept += 1
        return swept

    def sweep_stale_running_jobs(
        self,
        scope: str | None = None,
        job_id: str | None = None,
        now_ts: float | None = None,
        limit: int | None = None,
    ) -> int:
        """Fail live jobs whose controller lease has expired.

        This is a bounded cleanup pass for ``RUNNING`` and ``CANCELLING`` jobs.
        It can scan all scopes or a single scope/job. Jobs whose controllers have
        stopped heartbeating are marked failed and made retryable. ``limit`` caps
        the amount of cleanup done in one call.
        """
        now = float(time.time() if now_ts is None else now_ts)
        swept = 0
        for (record_scope, record_job_id), record in self._job_index.items():
            if limit is not None and swept >= limit:
                break
            if scope is not None and record_scope != scope:
                continue
            if job_id is not None and record_job_id != job_id:
                continue
            if self._fail_stale_running_record(record, now):
                swept += 1
        return swept

    def _retained_job_record_candidates(
        self,
        terminal_records: list[JobRecord],
        now: float,
        retention_s: float | None,
        max_per_scope: int | None,
    ) -> dict[tuple[str, str], JobRecord]:
        """Choose terminal job records that exceed retention limits.

        A record is a purge candidate if it is older than the time-based
        retention window or if its scope has more terminal records than the
        count-based limit. The result is keyed by record identity so a record
        selected by both rules is purged only once.
        """
        candidates: dict[tuple[str, str], JobRecord] = {}
        if retention_s is not None:
            cutoff = now - float(retention_s)
            for record in terminal_records:
                completed_at = record.get("completed_at_ts")
                if completed_at is not None and float(completed_at) <= cutoff:
                    candidates[self._record_key(record)] = record

        if max_per_scope is not None and max_per_scope >= 0:
            by_scope: dict[str, list[JobRecord]] = {}
            for record in terminal_records:
                by_scope.setdefault(str(record["scope"]), []).append(record)
            for rows in by_scope.values():
                newest_first = sorted(rows, key=self._completed_record_key, reverse=True)
                for record in newest_first[int(max_per_scope) :]:
                    candidates[self._record_key(record)] = record
        return candidates

    def sweep_retained_job_records(
        self,
        scope: str | None = None,
        now_ts: float | None = None,
        terminal_job_retention_s: float | None = None,
        max_retained_terminal_jobs_per_scope: int | None = None,
        limit: int | None = None,
    ) -> int:
        """Purge retained terminal job records to bound registry memory use.

        Completed, failed, and cancelled records are kept for a while so clients
        can list jobs and reattach to recent results. This sweep removes terminal
        records that exceed the time or count retention policy. Active jobs are
        never removed, and records with controllers are purged only after their
        controller actor is cleaned up. ``limit`` caps one sweep call.
        """
        now = float(time.time() if now_ts is None else now_ts)
        retention_s = self._terminal_job_retention_s if terminal_job_retention_s is None else terminal_job_retention_s
        max_per_scope = (
            self._max_retained_terminal_jobs_per_scope
            if max_retained_terminal_jobs_per_scope is None
            else max_retained_terminal_jobs_per_scope
        )

        terminal_records = [
            record
            for (record_scope, _job_id), record in self._job_index.items()
            if record["status"] in REGISTRY_TERMINAL_STATUSES and (scope is None or record_scope == scope)
        ]

        candidates = self._retained_job_record_candidates(terminal_records, now, retention_s, max_per_scope)
        ordered = sorted(candidates.values(), key=self._completed_record_key)
        if limit is not None:
            ordered = ordered[:limit]

        swept = 0
        for record in ordered:
            if self._purge_record(record, now):
                swept += 1
        return swept

    def sweep_terminal_controllers(self, now_ts: float | None = None, limit: int | None = None) -> int:
        """Clean up detached controller actors for terminal jobs.

        Controllers may be kept briefly after a job finishes so clients can
        reattach and so a controller can retry writing terminal state. This sweep
        kills or forgets terminal controllers that are past their retain-until
        time or over the count-based retention limit. The job record itself is
        kept; only the controller metadata is cleared. ``limit`` caps one sweep.
        """
        now = float(time.time() if now_ts is None else now_ts)
        terminal_records = [
            record
            for record in self._job_index.values()
            if record["status"] in REGISTRY_TERMINAL_STATUSES and record.get("controller_actor_name")
        ]

        candidates: dict[tuple[str, str], JobRecord] = {}
        for record in terminal_records:
            retain_until = record.get("controller_retain_until_ts")
            if retain_until is not None and float(retain_until) <= now:
                candidates[(str(record["scope"]), str(record["job_id"]))] = record

        max_retained = self._max_retained_terminal_controllers
        if max_retained is not None and max_retained >= 0:
            newest_first = sorted(
                terminal_records,
                key=lambda record: (float(record.get("completed_at_ts") or 0.0), str(record["job_id"])),
                reverse=True,
            )
            for record in newest_first[max_retained:]:
                candidates[(str(record["scope"]), str(record["job_id"]))] = record

        ordered = sorted(
            candidates.values(),
            key=lambda record: (float(record.get("completed_at_ts") or 0.0), str(record["job_id"])),
        )
        if limit is not None:
            ordered = ordered[:limit]

        swept = 0
        for record in ordered:
            if self._cleanup_controller_for_record(record, now):
                swept += 1
        return swept


JobRegistryActor = ray.remote(max_restarts=0)(JobRegistry)


def get_or_create_registry_actor(
    *,
    name: str,
    namespace: str,
    reservation_ttl_s: float,
    registry_num_cpus: float | None = None,
    registry_memory: float | None = None,
    registry_resources: dict[str, float] | None = None,
    registry_max_pending_calls: int | None = None,
    controller_heartbeat_ttl_s: float = DEFAULT_CONTROLLER_HEARTBEAT_TTL_S,
    controller_retention_s: float = DEFAULT_CONTROLLER_RETENTION_S,
    max_retained_terminal_controllers: int | None = DEFAULT_MAX_RETAINED_TERMINAL_CONTROLLERS,
    terminal_job_retention_s: float | None = DEFAULT_TERMINAL_JOB_RETENTION_S,
    max_retained_terminal_jobs_per_scope: int | None = DEFAULT_MAX_RETAINED_TERMINAL_JOBS_PER_SCOPE,
) -> ActorHandle:
    """Get existing detached registry actor or create it if absent."""
    try:
        return ray.get_actor(name, namespace=namespace)
    except ValueError:
        pass

    options: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
    }
    if registry_num_cpus is not None:
        options["num_cpus"] = float(registry_num_cpus)
    if registry_memory is not None:
        options["memory"] = float(registry_memory)
    if registry_resources is not None:
        options["resources"] = registry_resources
    if registry_max_pending_calls is not None:
        options["max_pending_calls"] = int(registry_max_pending_calls)

    try:
        return cast(
            ActorHandle,
            JobRegistryActor.options(**options).remote(
                reservation_ttl_s=float(reservation_ttl_s),
                controller_heartbeat_ttl_s=float(controller_heartbeat_ttl_s),
                controller_retention_s=float(controller_retention_s),
                max_retained_terminal_controllers=max_retained_terminal_controllers,
                terminal_job_retention_s=terminal_job_retention_s,
                max_retained_terminal_jobs_per_scope=max_retained_terminal_jobs_per_scope,
            ),
        )
    except ValueError:
        return ray.get_actor(name, namespace=namespace)
