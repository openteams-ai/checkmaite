from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import ray

from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend
from checkmaite.jobs import (
    CapabilityRunRef,
    JobCancelledError,
    JobFailedError,
    JobStatus,
    JobTimeoutError,
    configure_backend,
    get_job,
    list_jobs,
    shutdown_backend,
    submit_capability,
)
from checkmaite.jobs.backends.ray_simple import RaySimpleBackend, RaySimpleJob
from tests.test_jobs.fakes import AppendMarkerCapability, TinyCapability, TinyConfig


@pytest.fixture(name="ray_simple_runtime")
def _ray_simple_runtime():
    """Function-scoped local Ray runtime for ray-simple integration tests."""
    shutdown_backend(wait=False)
    ray.shutdown()
    ray.init(address="local")

    try:
        yield
    finally:
        shutdown_backend(wait=False)
        ray.shutdown()


@pytest.fixture
def local_ray_simple(ray_simple_runtime, tmp_path: Path):
    store_path = tmp_path / "analytics-store"

    shutdown_backend(wait=False)
    configure_backend(
        "ray-simple",
        analytics_store={"backend": "parquet", "uri": str(store_path)},
    )

    try:
        yield store_path
    finally:
        shutdown_backend(wait=False)


def _store(path: Path) -> AnalyticsStore:
    return AnalyticsStore(ParquetBackend(str(path)))


@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_ray_simple_backend_smoke_contract_exercises_default_backend_coverage(tmp_path: Path) -> None:
    store_path = tmp_path / "analytics-store"
    backend = None

    shutdown_backend(wait=False)

    try:
        backend = RaySimpleBackend(
            address=None,
            analytics_store={"backend": "parquet", "uri": str(store_path)},
        )
        capability = TinyCapability()

        completed = backend.submit_capability(capability, config=TinyConfig(text="simple-smoke"), use_cache=False)
        assert backend.get_job(completed.job_id).job_id == completed.job_id
        assert any(job.job_id == completed.job_id for job in backend.list_jobs())

        ref = completed.result(timeout=30)
        assert ref.summary["md_report"] == "simple-smoke:0.5"
        assert completed.wait(timeout=1) is JobStatus.COMPLETED
        assert completed.status is JobStatus.COMPLETED
        assert completed.exception() is None

        failed = backend.submit_capability(capability, config=TinyConfig(fail=True), use_cache=False)
        with pytest.raises(JobFailedError, match="tiny capability failure"):
            failed.result(timeout=30)
        assert failed.status is JobStatus.FAILED
        assert failed.exception() is not None

        cancellable = backend.submit_capability(
            capability, config=TinyConfig(text="cancel", sleep_s=1.0), use_cache=False
        )
        assert cancellable.cancel() is True
        assert cancellable.wait(timeout=10) is JobStatus.CANCELLED
        with pytest.raises(JobCancelledError):
            cancellable.result(timeout=1)

        assert [job.job_id for job in backend.list_jobs(limit=1)]
        assert any(job.job_id == completed.job_id for job in backend.list_jobs(status_filter=JobStatus.COMPLETED))
        assert any(job.job_id == failed.job_id for job in backend.list_jobs(status_filter=JobStatus.FAILED))

        result = _store(store_path).query_sql("SELECT payload FROM tiny_jobs")
        assert "simple-smoke" in set(result["payload"].to_list())
    finally:
        if backend is not None:
            backend.shutdown(wait=False)


def _listed_job(job_id: str, created_at: datetime, status: JobStatus) -> RaySimpleJob:
    job = object.__new__(RaySimpleJob)
    job._job_id = job_id
    job._created_at = created_at
    job._terminal_status = status
    job._resolved_exception = None
    job._has_been_polled = True
    job._obj_ref = None
    return job


@pytest.mark.parametrize(
    "status_filter",
    ["completed", ["completed"], [JobStatus.COMPLETED, "failed"]],
)
def test_status_filter_values_rejects_non_job_status(status_filter) -> None:
    with pytest.raises(TypeError, match="status_filter must be a JobStatus"):
        RaySimpleBackend._status_filter_values(status_filter)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("status_filter", "expected"),
    [
        (JobStatus.COMPLETED, {JobStatus.COMPLETED}),
        ([JobStatus.COMPLETED, JobStatus.FAILED], {JobStatus.COMPLETED, JobStatus.FAILED}),
    ],
)
def test_status_filter_values_accepts_job_status_values(status_filter, expected) -> None:
    assert RaySimpleBackend._status_filter_values(status_filter) == expected


def test_get_job_rejects_unknown_job_id() -> None:
    backend = object.__new__(RaySimpleBackend)
    backend._jobs = {}

    with pytest.raises(KeyError, match="No job with ID"):
        backend.get_job("missing")


def test_list_jobs_limit_status_and_submitted_before_filters() -> None:
    backend = object.__new__(RaySimpleBackend)
    now = datetime.now(timezone.utc)
    old = _listed_job("old", now - timedelta(seconds=20), JobStatus.COMPLETED)
    middle = _listed_job("middle", now - timedelta(seconds=10), JobStatus.FAILED)
    new = _listed_job("new", now - timedelta(seconds=1), JobStatus.COMPLETED)
    backend._jobs = {job.job_id: job for job in (old, middle, new)}

    assert [job.job_id for job in backend.list_jobs()] == ["new", "middle", "old"]
    assert [job.job_id for job in backend.list_jobs(limit=2)] == ["new", "middle"]
    assert [job.job_id for job in backend.list_jobs(status_filter=JobStatus.COMPLETED)] == ["new", "old"]
    assert [
        job.job_id
        for job in backend.list_jobs(
            status_filter=[JobStatus.COMPLETED, JobStatus.FAILED],
            submitted_before_ts=(now - timedelta(seconds=5)).timestamp(),
        )
    ] == ["middle", "old"]


def test_resource_resolution_priority() -> None:
    backend = object.__new__(RaySimpleBackend)

    class DefaultHintCapability(TinyCapability):
        default_num_cpus = 8
        default_num_gpus = 0.75

    capability = DefaultHintCapability()

    class ConfigHints:
        num_cpus = 4
        num_gpus = 0.5

    explicit = backend._resolve_resources(
        capability,
        {
            "resources": {
                "num_cpus": 2,
                "num_gpus": 1.5,
            },
            "config": ConfigHints(),
        },
    )
    assert explicit == {"num_cpus": 2, "num_gpus": 1.5}

    from_config = backend._resolve_resources(capability, {"config": ConfigHints()})
    assert from_config == {"num_cpus": 4, "num_gpus": 0.5}

    from_capability_defaults = backend._resolve_resources(capability, {})
    assert from_capability_defaults == {"num_cpus": 8, "num_gpus": 0.75}

    from_fallback = backend._resolve_resources(object(), {})
    assert from_fallback == {"num_cpus": 1, "num_gpus": 0.0}


@pytest.mark.ray
def test_submit_capability_returns_ref_and_writes_store(local_ray_simple: Path) -> None:
    capability = TinyCapability()

    job = submit_capability(capability, config=TinyConfig(text="hello-simple"), use_cache=False)
    assert isinstance(job, RaySimpleJob)
    assert get_job(job.job_id).job_id == job.job_id
    assert any(j.job_id == job.job_id for j in list_jobs())

    ref = job.result(timeout=30)

    assert isinstance(ref, CapabilityRunRef)
    assert ref.capability_id == capability.id
    assert ref.outputs_uri is None
    assert ref.store_uri.endswith(".parquet")
    assert "#" not in ref.store_uri
    assert ref.summary["md_report"] == "hello-simple:0.5"

    assert job.wait(timeout=1) is JobStatus.COMPLETED
    assert job.status is JobStatus.COMPLETED
    assert job.exception() is None

    result = _store(local_ray_simple).query_sql("SELECT run_uid, payload FROM tiny_jobs")
    assert result.shape[0] == 1
    assert result["run_uid"][0] == ref.run_uid
    assert result["payload"][0] == "hello-simple"


@pytest.mark.ray
def test_result_timeout_and_cancel(local_ray_simple: Path) -> None:
    capability = TinyCapability()

    job = submit_capability(capability, config=TinyConfig(text="slow", sleep_s=5.0), use_cache=False)

    with pytest.raises(JobTimeoutError):
        job.result(timeout=0.01)

    assert job.cancel() is True
    assert job.wait(timeout=10) is JobStatus.CANCELLED

    with pytest.raises(JobCancelledError):
        job.result(timeout=10)


@pytest.mark.ray
def test_result_timeout_does_not_cancel_task(local_ray_simple: Path) -> None:
    finish_marker = local_ray_simple.parent / "timeout-finished.txt"
    job = submit_capability(
        TinyCapability(),
        config=TinyConfig(text="timeout-continues", sleep_s=0.5, finish_marker_path=str(finish_marker)),
        use_cache=False,
    )

    with pytest.raises(JobTimeoutError):
        job.result(timeout=0.01)

    ref = job.result(timeout=30)
    assert ref.summary["md_report"] == "timeout-continues:0.5"
    assert finish_marker.read_text() == "finished"
    assert job.status is JobStatus.COMPLETED


@pytest.mark.ray
def test_wait_timeout_does_not_cancel_task(local_ray_simple: Path) -> None:
    job = submit_capability(TinyCapability(), config=TinyConfig(text="wait-continues", sleep_s=0.5), use_cache=False)

    status = job.wait(timeout=0.01)
    assert status is JobStatus.RUNNING

    ref = job.result(timeout=30)
    assert ref.summary["md_report"] == "wait-continues:0.5"
    assert job.status is JobStatus.COMPLETED


@pytest.mark.ray
def test_worker_execution_ignores_local_cache(local_ray_simple: Path) -> None:
    capability = AppendMarkerCapability()
    marker = local_ray_simple.parent / "cache-marker.txt"
    config = TinyConfig(text="cache-ignored", start_marker_path=str(marker))

    cached_run = capability.run(config=config, use_cache=True)
    assert marker.read_text().splitlines() == ["run"]

    job = submit_capability(capability, config=config, use_cache=True)
    ref = job.result(timeout=30)

    assert ref.run_uid == cached_run.run_uid
    assert ref.summary["md_report"] == "cache-ignored:0.5"
    assert marker.read_text().splitlines() == ["run", "run"]


@pytest.mark.ray
def test_failure_is_mapped_to_job_failed_error(local_ray_simple: Path) -> None:
    capability = TinyCapability()

    job = submit_capability(capability, config=TinyConfig(fail=True), use_cache=False)

    with pytest.raises(JobFailedError, match="tiny capability failure"):
        job.result(timeout=30)

    assert job.status is JobStatus.FAILED
    assert job.exception() is not None


@pytest.mark.ray
def test_reconfigure_wait_false_does_not_interrupt_inflight_job(local_ray_simple: Path) -> None:
    capability = TinyCapability()

    job = submit_capability(capability, config=TinyConfig(text="inflight", sleep_s=1.5), use_cache=False)

    # Default reconfigure path should be non-blocking and not tear down runtime.
    configure_backend(
        "ray-simple",
        analytics_store={"backend": "parquet", "uri": str(local_ray_simple)},
    )

    ref = job.result(timeout=30)
    assert ref.summary["md_report"] == "inflight:0.5"


@pytest.mark.ray
def test_shutdown_wait_false_leaves_ray_initialized(ray_simple_runtime, tmp_path: Path) -> None:
    backend = RaySimpleBackend(analytics_store={"backend": "parquet", "uri": str(tmp_path / "analytics-store")})

    backend.shutdown(wait=False)

    assert ray.is_initialized()


@pytest.mark.ray
def test_shutdown_wait_true_waits_for_jobs_and_shuts_down_ray(ray_simple_runtime, tmp_path: Path) -> None:
    store_path = tmp_path / "analytics-store"
    finish_marker = tmp_path / "shutdown-finished.txt"
    backend = RaySimpleBackend(
        analytics_store={"backend": "parquet", "uri": str(store_path)},
    )

    backend.submit_capability(
        TinyCapability(),
        config=TinyConfig(text="shutdown", sleep_s=0.2, finish_marker_path=str(finish_marker)),
        use_cache=False,
    )
    backend.shutdown(wait=True)

    assert not ray.is_initialized()
    assert finish_marker.read_text() == "finished"
    result = _store(store_path).query_sql("SELECT payload FROM tiny_jobs")
    assert result.shape[0] == 1
    assert result["payload"][0] == "shutdown"


@pytest.mark.ray
def test_store_write_failure_raises_by_default(local_ray_simple: Path) -> None:
    capability = TinyCapability()

    # Point analytics store to a regular file so table directory creation fails.
    bad_store_root = local_ray_simple.parent / "not-a-directory"
    bad_store_root.write_text("this is a file")

    configure_backend(
        "ray-simple",
        analytics_store={"backend": "parquet", "uri": str(bad_store_root)},
    )

    job = submit_capability(capability, config=TinyConfig(text="no-store"), use_cache=False)
    with pytest.raises(JobFailedError):
        job.result(timeout=30)


@pytest.mark.ray
def test_repeated_submissions_are_independent_jobs(local_ray_simple: Path) -> None:
    capability = TinyCapability()
    config = TinyConfig(text="repeat-simple")

    job1 = submit_capability(capability, config=config, use_cache=True)
    job2 = submit_capability(capability, config=config, use_cache=True)

    assert isinstance(job1, RaySimpleJob)
    assert isinstance(job2, RaySimpleJob)
    assert job1.job_id != job2.job_id

    ref1 = job1.result(timeout=30)
    ref2 = job2.result(timeout=30)
    assert ref1.summary["md_report"] == "repeat-simple:0.5"
    assert ref2.summary["md_report"] == "repeat-simple:0.5"
