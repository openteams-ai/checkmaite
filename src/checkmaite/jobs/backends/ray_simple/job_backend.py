from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

import ray
from ray.exceptions import GetTimeoutError, TaskCancelledError

from checkmaite.core.analytics_store import AnalyticsStore
from checkmaite.core.capability_core import CapabilityRunBase
from checkmaite.jobs._store import AnalyticsStoreConfig, build_analytics_store, write_run_and_get_store_uri
from checkmaite.jobs._submission import prepare_job_submission_run_kwargs
from checkmaite.jobs.protocol import (
    CapabilityRunRef,
    CapabilityType,
    Job,
    JobCancelledError,
    JobFailedError,
    JobStatus,
    JobTimeoutError,
)

logger = logging.getLogger(__name__)


def _collect_md_report(run: CapabilityRunBase[Any, Any], threshold: float) -> str | list[Any]:
    try:
        return run.collect_md_report(threshold=threshold)
    except NotImplementedError:
        return []


def _get_worker_store(store_config: AnalyticsStoreConfig | dict[str, Any]) -> AnalyticsStore:
    return build_analytics_store(store_config)


def _write_run_and_collect_store_metadata(
    store: AnalyticsStore,
    run: CapabilityRunBase[Any, Any],
) -> str:
    return write_run_and_get_store_uri(store, run)


def _execute_capability_ref(capability: CapabilityType, run_kwargs: dict[str, Any]) -> CapabilityRunRef:
    """Execute one capability submission inside a Ray worker process.

    This function is serialized by the client and submitted via
    ``ray.remote(...)(_execute_capability_ref).remote(...)``. Ray then executes
    it on a worker, where it runs the capability, writes analytics-store data,
    and returns a lightweight :class:`CapabilityRunRef` to the client.
    """
    # TODO: Future work should support a remote/shared cache backend
    # (for example object storage) that workers can read from. At that point,
    # worker execution can safely opt into cache usage.
    run_kwargs = prepare_job_submission_run_kwargs(run_kwargs)

    report_threshold = float(run_kwargs.pop("report_threshold", 0.5))
    raw_store_config = run_kwargs.pop("_analytics_store")

    run = capability.run(**run_kwargs)

    store = _get_worker_store(raw_store_config)
    store_uri = _write_run_and_collect_store_metadata(
        store,
        run,
    )

    summary: dict[str, Any] = {
        "md_report": _collect_md_report(run, threshold=report_threshold),
    }

    return CapabilityRunRef(
        run_uid=run.run_uid,
        capability_id=run.capability_id,
        store_uri=store_uri,
        outputs_uri=None,
        summary=summary,
    )


class RaySimpleJob(Job[CapabilityRunRef]):
    """Thin checkmaite wrapper over Ray ObjectRef lifecycle primitives."""

    def __init__(self, job_id: str, created_at: datetime, obj_ref: ray.ObjectRef[CapabilityRunRef]) -> None:
        self._job_id = job_id
        self._created_at = created_at
        self._obj_ref = obj_ref

        self._terminal_status: JobStatus | None = None
        self._resolved_exception: BaseException | None = None
        self._has_been_polled = False

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def status(self) -> JobStatus:
        if self._terminal_status is not None:
            return self._terminal_status

        ready, _ = ray.wait([self._obj_ref], timeout=0)

        if not ready:
            if not self._has_been_polled:
                self._has_been_polled = True
                return JobStatus.PENDING
            return JobStatus.RUNNING

        try:
            ray.get(self._obj_ref, timeout=0)
            self._terminal_status = JobStatus.COMPLETED
        except GetTimeoutError:
            # Rare race: object reported ready but fetch timed out.
            if not self._has_been_polled:
                self._has_been_polled = True
                return JobStatus.PENDING
            return JobStatus.RUNNING
        except TaskCancelledError:
            self._terminal_status = JobStatus.CANCELLED
        except Exception as exc:  # noqa: BLE001
            self._resolved_exception = exc
            self._terminal_status = JobStatus.FAILED

        return self._terminal_status

    def result(self, timeout: float | None = None) -> CapabilityRunRef:
        if self._terminal_status is JobStatus.CANCELLED:
            raise JobCancelledError(self._job_id)

        try:
            ref = ray.get(self._obj_ref, timeout=timeout)
            self._terminal_status = JobStatus.COMPLETED
            return ref if isinstance(ref, CapabilityRunRef) else CapabilityRunRef.model_validate(ref)
        except GetTimeoutError:
            raise JobTimeoutError(self._job_id, timeout or 0.0) from None
        except TaskCancelledError:
            self._terminal_status = JobStatus.CANCELLED
            raise JobCancelledError(self._job_id) from None
        except Exception as exc:  # noqa: BLE001
            self._resolved_exception = exc
            self._terminal_status = JobStatus.FAILED
            raise JobFailedError(self._job_id, exc) from None

    def wait(self, timeout: float | None = None) -> JobStatus:
        deadline = None if timeout is None else time.time() + timeout

        while True:
            st = self.status
            if st.is_terminal:
                return st

            if deadline is not None and time.time() >= deadline:
                return st

            time.sleep(0.2)

    def cancel(self) -> bool:
        if self.status.is_terminal:
            return False

        ray.cancel(self._obj_ref)
        # Ray cancellation is best-effort at the scheduler/worker boundary.
        # Once a cancel request is accepted, this handle is treated as cancelled.
        self._terminal_status = JobStatus.CANCELLED
        return True

    def exception(self) -> BaseException | None:
        if self.status is JobStatus.FAILED:
            return self._resolved_exception
        return None


class RaySimpleJobBackend:
    """Ray Core backend for asynchronous capability execution."""

    def __init__(
        self,
        analytics_store: AnalyticsStoreConfig | dict[str, Any],
        address: str | None = None,
        runtime_env: dict[str, Any] | None = None,
        max_retries: int = 0,
        force_reinit: bool = False,
    ) -> None:
        if force_reinit and ray.is_initialized():
            ray.shutdown()

        if not ray.is_initialized():
            ray.init(address=address, runtime_env=runtime_env)
        elif address is not None or runtime_env is not None:
            logger.warning(
                "Ray runtime already initialized; new address/runtime_env were ignored. "
                "Pass force_reinit=True to reconnect (may interrupt in-flight jobs)."
            )

        self._analytics_store = AnalyticsStoreConfig.model_validate(analytics_store)
        self._max_retries = max_retries
        self._jobs: dict[str, RaySimpleJob] = {}

    def _resolve_resources(self, capability: CapabilityType, run_kwargs: dict[str, Any]) -> dict[str, float | int]:
        resources = run_kwargs.get("resources") or {}
        config = run_kwargs.get("config")

        num_gpus = resources.get("num_gpus")
        num_cpus = resources.get("num_cpus")

        if num_gpus is None and config is not None:
            num_gpus = getattr(config, "num_gpus", None)
        if num_cpus is None and config is not None:
            num_cpus = getattr(config, "num_cpus", None)

        if num_gpus is None:
            num_gpus = getattr(capability, "default_num_gpus", 0)
        if num_cpus is None:
            num_cpus = getattr(capability, "default_num_cpus", 1)

        return {"num_gpus": float(num_gpus), "num_cpus": int(num_cpus)}

    def submit_capability(self, capability: CapabilityType, **kwargs: Any) -> RaySimpleJob:
        run_kwargs = prepare_job_submission_run_kwargs(kwargs)
        job_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc)

        resources = self._resolve_resources(capability, run_kwargs)
        run_kwargs.pop("resources", None)

        remote_fn = ray.remote(
            num_gpus=resources["num_gpus"],
            num_cpus=resources["num_cpus"],
            max_retries=self._max_retries,
        )(_execute_capability_ref)

        run_kwargs["_analytics_store"] = self._analytics_store.model_dump(mode="python")

        try:
            obj_ref = remote_fn.remote(capability, run_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to submit capability job {job_id}: {exc}") from exc

        job = RaySimpleJob(job_id=job_id, created_at=created_at, obj_ref=obj_ref)

        self._jobs[job_id] = job
        return job

    @staticmethod
    def _status_filter_values(status_filter: JobStatus | Sequence[JobStatus]) -> set[JobStatus]:
        """Normalize a non-None status filter into a set of JobStatus values."""
        if isinstance(status_filter, JobStatus):
            return {status_filter}

        try:
            statuses = set(status_filter)
        except TypeError as exc:
            raise TypeError("status_filter must be a JobStatus or a sequence of JobStatus values") from exc

        if not all(isinstance(status, JobStatus) for status in statuses):
            raise TypeError("status_filter must be a JobStatus or a sequence of JobStatus values")
        return statuses

    def list_jobs(
        self,
        limit: int | None = None,
        status_filter: JobStatus | Sequence[JobStatus] | None = None,
        submitted_before_ts: float | None = None,
    ) -> list[RaySimpleJob]:
        jobs = list(self._jobs.values())
        if status_filter is not None:
            statuses = self._status_filter_values(status_filter)
            jobs = [job for job in jobs if job.status in statuses]
        if submitted_before_ts is not None:
            jobs = [job for job in jobs if job.created_at.timestamp() < submitted_before_ts]
        jobs.sort(key=lambda job: (job.created_at, job.job_id), reverse=True)
        return jobs if limit is None else jobs[: max(0, int(limit))]

    def get_job(self, job_id: str) -> RaySimpleJob:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError(f"No job with ID {job_id!r}") from None

    def shutdown(self, wait: bool = True) -> None:
        if not wait:
            return

        for job in self._jobs.values():
            with suppress(Exception):
                job.wait(timeout=None)

        if ray.is_initialized():
            ray.shutdown()
