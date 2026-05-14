from __future__ import annotations

import enum
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypedDict, TypeVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from checkmaite.core.capability_core import Capability as _Capability

    CapabilityType: TypeAlias = _Capability[Any, Any, Any, Any, Any]
else:
    CapabilityType: TypeAlias = Any


class JobStatus(enum.Enum):
    """Lifecycle status for submitted jobs.

    Lifecycle model
    ---------------
    Typical progression is::

        submit_capability() -> PENDING -> RUNNING -> COMPLETED

    with failure/cancellation branches from non-terminal states.

    Transition notes
    ----------------
    - ``PENDING`` usually transitions to ``RUNNING`` or ``CANCELLED``.
      Some backends may also report ``FAILED`` from ``PENDING`` when
      queue/wait timeout policies are enforced.
    - ``RUNNING`` transitions to ``COMPLETED``, ``FAILED``, or ``CANCELLED``.
      In distributed schedulers, ``RUNNING`` may transiently return to
      ``PENDING`` if the work is rescheduled after worker loss.
    - ``FAILED`` can represent user-code exceptions, repeated worker crashes
      (for example OOM/segfault) exhausting retries, client-side timeout
      policies, or scheduler-side timeout policies.

    Terminal states are immutable.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether this status is terminal (and therefore immutable)."""
        return self in (self.COMPLETED, self.FAILED, self.CANCELLED)


class CapabilityRunRefPayload(TypedDict):
    """Plain serialized form of ``CapabilityRunRef`` used at storage boundaries.

    ``CapabilityRunRef`` is the validated public API object returned to users.
    Backends store and pass this payload form in registries, actor messages, and
    future durable metadata stores so job records stay JSON-compatible plain
    data instead of long-lived Pydantic/Python objects. Convert models to this
    shape with ``CapabilityRunRef.model_dump(mode="json")`` and validate it back
    with ``CapabilityRunRef.model_validate(...)`` before returning it to users.
    """

    run_uid: str
    capability_id: str
    store_uri: str
    outputs_uri: str | None
    summary: dict[str, Any]


class CapabilityRunRef(BaseModel):
    """Lightweight reference returned by successfully completed jobs.

    Semantics
    ---------
    run_uid
        Deterministic identity for the run.
    store_uri
        Stable URI for a concrete analytics-store payload object containing
        data for this run.
    outputs_uri
        URI for full serialized outputs/artifacts when available. May be
        ``None`` when cache/artifact materialization is not enabled.
    summary
        Small human-readable reports used by the checkmaite package
        (for example markdown report snippets).
    """

    run_uid: str
    capability_id: str
    store_uri: str
    outputs_uri: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class JobError(Exception):
    """Base error for job-handles."""

    def __init__(self, job_id: str, message: str) -> None:
        self.job_id = job_id
        super().__init__(f"Job {job_id}: {message}")


class JobFailedError(JobError):
    """Raised when a job completes with failure."""

    def __init__(self, job_id: str, error: BaseException | str) -> None:
        super().__init__(job_id, f"failed: {error}")


class JobCancelledError(JobError):
    """Raised when a job is cancelled."""

    def __init__(self, job_id: str) -> None:
        super().__init__(job_id, "cancelled")


class JobTimeoutError(JobError, TimeoutError):
    """Raised when waiting for a job exceeds timeout."""

    def __init__(self, job_id: str, timeout: float) -> None:
        super().__init__(job_id, f"timed out after {timeout:.3f}s")


class RunArtifactNotAvailableError(JobError):
    """Raised when requested run artifacts are unavailable."""


T = TypeVar("T", covariant=True)


class Job(Protocol, Generic[T]):
    """Backend-agnostic handle for an asynchronous submission.

    A ``Job`` represents one submitted unit of work and exposes a minimal,
    notebook-friendly lifecycle API shared across backends.

    Status observations follow :class:`JobStatus` semantics. Terminal states
    (``COMPLETED``, ``FAILED``, ``CANCELLED``) are immutable.

    ``T`` is the payload type returned by :meth:`result`. For capability
    submissions this is ``Job[CapabilityRunRef]``, i.e. a small reference-first
    payload (not the full run artifact object).
    """

    @property
    def job_id(self) -> str:
        """Unique identifier for this submission in the active backend."""
        ...

    @property
    def status(self) -> JobStatus:
        """Latest observed lifecycle status for this job.

        Returns a snapshot and may change between calls until terminal.
        """
        ...

    @property
    def created_at(self) -> datetime:
        """Timestamp (UTC) when the job handle was created/submitted."""
        ...

    def result(self, timeout: float | None = None) -> T:
        """Wait for terminal success and return the result payload.

        For capability submission (``Job[CapabilityRunRef]``), this means:
        - wait for terminal success,
        - return :class:`CapabilityRunRef`,
        - re-raise remote exceptions as :class:`JobFailedError` on failure.

        Parameters
        ----------
        timeout
            Maximum number of seconds to wait. ``None`` waits indefinitely.

        Returns
        -------
        T
            The backend result payload for this job when status reaches
            ``COMPLETED``.

        Raises
        ------
        JobTimeoutError
            If ``timeout`` elapses before completion (including while
            ``PENDING``).
        JobCancelledError
            If the job reaches ``CANCELLED``.
        JobFailedError
            If the job reaches ``FAILED`` (for example user exception,
            repeated worker crashes, or timeout policy failure).
        """
        ...

    def wait(self, timeout: float | None = None) -> JobStatus:
        """Block until terminal or timeout and return the observed status.

        Unlike :meth:`result`, this method does not return a payload and does
        not raise on failure/cancellation by default.

        Notes
        -----
        If ``timeout`` expires, implementations typically return the latest
        non-terminal status (commonly ``PENDING`` or ``RUNNING``). Backends
        with explicit timeout policies may instead surface ``FAILED``.
        """
        ...

    def exception(self) -> BaseException | None:
        """Return the captured failure exception when available.

        Returns ``None`` for non-failed jobs or when no exception has been
        resolved yet by the backend wrapper.
        """
        ...

    def cancel(self) -> bool:
        """Request cancellation of this job.

        Returns
        -------
        bool
            ``True`` if a cancellation request was issued, ``False`` if the
            job is already in a terminal state.
        """
        ...


class Backend(Protocol):
    """Backend protocol for capability job submission.

    Implementations are responsible for mapping backend-native execution state
    to the shared :class:`JobStatus` lifecycle and exception contracts.

    For capability submission, backends should follow reference-first results
    and return :class:`CapabilityRunRef` via :meth:`Job.result`.
    """

    def submit_capability(self, capability: CapabilityType, **kwargs: Any) -> Job[CapabilityRunRef]:
        """Submit a capability and return ``Job[CapabilityRunRef]``.

        Parameters
        ----------
        capability
            A checkmaite capability instance (same conceptual input as
            ``Capability.run(...)``).
        **kwargs
            Submission/run arguments (models, datasets, metrics, config,
            cache flags, backend-specific options).

        Returns
        -------
        Job[CapabilityRunRef]
            Asynchronous handle whose :meth:`Job.result` follows reference-first
            semantics.
        """
        ...

    def list_jobs(
        self,
        limit: int | None = None,
        status_filter: JobStatus | Sequence[JobStatus] | None = None,
        submitted_before_ts: float | None = None,
    ) -> Sequence[Job[CapabilityRunRef]]:
        """Return recent capability-submission jobs tracked by this backend.

        Parameters
        ----------
        limit
            Maximum number of jobs to return, ordered newest first. ``None`` returns all tracked jobs.
        status_filter
            Optional status or sequence of statuses to include.
        submitted_before_ts
            Optional Unix timestamp cursor. When provided, only jobs submitted before this timestamp are returned.
        """
        ...

    def get_job(self, job_id: str) -> Job[CapabilityRunRef]:
        """Return a tracked capability-submission job by ``job_id``.

        Raises
        ------
        KeyError
            If the backend does not track ``job_id``.
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Shut down backend resources/tracking state.

        Parameters
        ----------
        wait
            When ``True``, wait for tracked jobs to reach terminal states before
            completing shutdown. When ``False``, return immediately without
            waiting.
        """
        ...
