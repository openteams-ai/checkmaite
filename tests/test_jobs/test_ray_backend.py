from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest
import ray

from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend
from checkmaite.jobs import (
    CapabilityRunRef,
    JobCancelledError,
    JobFailedError,
    JobStatus,
    JobTimeoutError,
    RayBackend,
    RayJob,
    configure_backend,
    get_job,
    list_jobs,
    shutdown_backend,
    submit_capability,
)
from tests.test_jobs.fakes import TinyCapability, TinyConfig, TinyDatasetCapability


@pytest.fixture
def local_ray(tmp_path: Path):
    store_path = tmp_path / "analytics-store"

    shutdown_backend(wait=False)
    ray.shutdown()

    configure_backend(
        "ray",
        address="local",
        analytics_store={"backend": "parquet", "uri": str(store_path)},
    )

    try:
        yield store_path
    finally:
        shutdown_backend(wait=False)
        ray.shutdown()


def _store(path: Path) -> AnalyticsStore:
    return AnalyticsStore(ParquetBackend(str(path)))


@dataclass
class _ListedJob:
    job_id: str
    created_at: datetime
    status: JobStatus


@pytest.mark.ray
def test_submit_capability_returns_ref_and_writes_store(local_ray: Path) -> None:
    capability = TinyCapability()

    job = submit_capability(capability, config=TinyConfig(text="hello"), use_cache=False)
    assert get_job(job.job_id).job_id == job.job_id
    assert any(j.job_id == job.job_id for j in list_jobs())

    ref = job.result(timeout=30)

    assert isinstance(ref, CapabilityRunRef)
    assert ref.capability_id == capability.id
    assert ref.outputs_uri is None
    assert ref.store_uri.endswith(".parquet")
    assert "#" not in ref.store_uri
    assert ref.summary["md_report"] == "hello:0.5"

    assert job.wait(timeout=1) is JobStatus.COMPLETED
    assert job.status is JobStatus.COMPLETED
    assert job.exception() is None

    result = _store(local_ray).query_sql("SELECT run_uid, payload FROM tiny_jobs")
    assert result.shape[0] == 1
    assert result["run_uid"][0] == ref.run_uid
    assert result["payload"][0] == "hello"


@pytest.mark.ray
def test_result_timeout_and_cancel(local_ray: Path) -> None:
    capability = TinyCapability()

    job = submit_capability(capability, config=TinyConfig(text="slow", sleep_s=5.0), use_cache=False)

    with pytest.raises(JobTimeoutError):
        job.result(timeout=0.01)

    assert job.cancel() is True
    assert job.wait(timeout=10) is JobStatus.CANCELLED

    with pytest.raises(JobCancelledError):
        job.result(timeout=10)


@pytest.mark.ray
def test_failure_is_mapped_to_job_failed_error(local_ray: Path) -> None:
    capability = TinyCapability()

    job = submit_capability(capability, config=TinyConfig(fail=True), use_cache=False)

    with pytest.raises(JobFailedError, match="tiny capability failure"):
        job.result(timeout=30)

    assert job.status is JobStatus.FAILED
    assert job.exception() is not None


@pytest.mark.ray
def test_submit_capability_does_not_check_local_cache_before_submission(local_ray: Path) -> None:
    capability = TinyCapability()
    config = TinyConfig(text="cached")

    cached_run = capability.run(config=config, use_cache=True)

    job = submit_capability(capability, config=config, use_cache=True)

    assert isinstance(job, RayJob)

    ref = job.result(timeout=30)
    assert ref.run_uid == cached_run.run_uid
    assert ref.summary["md_report"] == "cached:0.5"

    result = _store(local_ray).query_sql("SELECT run_uid, payload FROM tiny_jobs")
    assert result.shape[0] == 1
    assert result["run_uid"][0] == cached_run.run_uid
    assert result["payload"][0] == "cached"


def test_list_jobs_status_filter_rejects_non_job_status() -> None:
    with pytest.raises(TypeError, match="status_filter must be a JobStatus"):
        RayBackend._status_filter_values("completed")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "status_filter",
    [JobStatus.COMPLETED, [JobStatus.COMPLETED, JobStatus.FAILED]],
)
def test_list_jobs_status_filter_accepts_job_status_values(
    status_filter: JobStatus | list[JobStatus],
) -> None:
    assert RayBackend._status_filter_values(status_filter) == set(
        status_filter if isinstance(status_filter, list) else [status_filter]
    )


def test_list_jobs_limit_status_and_submitted_before_filters() -> None:
    backend = object.__new__(RayBackend)
    now = datetime.now(timezone.utc)
    old = _ListedJob("old", now - timedelta(seconds=20), JobStatus.COMPLETED)
    middle = _ListedJob("middle", now - timedelta(seconds=10), JobStatus.FAILED)
    new = _ListedJob("new", now - timedelta(seconds=1), JobStatus.COMPLETED)
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
    class DefaultHintCapability(TinyCapability):
        default_num_cpus = 8
        default_num_gpus = 0.75

    capability = DefaultHintCapability()

    class ConfigHints:
        num_cpus = 4
        num_gpus = 0.5

    explicit = RayBackend._resolve_resources(
        capability,
        {
            "resources": {"num_cpus": 2, "num_gpus": 1.5},
            "config": ConfigHints(),
        },
    )
    assert explicit == {"num_cpus": 2, "num_gpus": 1.5}

    from_config = RayBackend._resolve_resources(capability, {"config": ConfigHints()})
    assert from_config == {"num_cpus": 4, "num_gpus": 0.5}

    from_capability_defaults = RayBackend._resolve_resources(capability, {})
    assert from_capability_defaults == {"num_cpus": 8, "num_gpus": 0.75}

    from_fallback = RayBackend._resolve_resources(object(), {})
    assert from_fallback == {"num_cpus": 1, "num_gpus": 0.0}


@pytest.mark.ray
def test_reconfigure_wait_false_does_not_interrupt_inflight_job(local_ray: Path) -> None:
    capability = TinyCapability()

    job = submit_capability(capability, config=TinyConfig(text="inflight", sleep_s=1.5), use_cache=False)

    # Default reconfigure path should be non-blocking and not tear down runtime.
    configure_backend("ray", analytics_store={"backend": "parquet", "uri": str(local_ray)})

    ref = job.result(timeout=30)
    assert ref.summary["md_report"] == "inflight:0.5"


@pytest.mark.ray
def test_backend_reconfigure_applies_new_runtime_env_with_force_reinit(local_ray: Path) -> None:
    capability = TinyCapability()
    env_key = "CHECKMAITE_TEST_RECONFIG_ENV"

    configure_backend(
        "ray",
        address="local",
        force_reinit=True,
        analytics_store={"backend": "parquet", "uri": str(local_ray)},
        runtime_env={
            "env_vars": {
                env_key: "one",
            }
        },
    )
    ref_one = submit_capability(
        capability,
        config=TinyConfig(env_key=env_key),
        use_cache=False,
    ).result(timeout=30)
    assert ref_one.summary["md_report"] == "one:0.5"

    configure_backend(
        "ray",
        address="local",
        force_reinit=True,
        analytics_store={"backend": "parquet", "uri": str(local_ray)},
        runtime_env={
            "env_vars": {
                env_key: "two",
            }
        },
    )
    ref_two = submit_capability(
        capability,
        config=TinyConfig(env_key=env_key),
        use_cache=False,
    ).result(timeout=30)
    assert ref_two.summary["md_report"] == "two:0.5"


@pytest.mark.ray
def test_store_write_failure_raises_by_default(local_ray: Path) -> None:
    capability = TinyCapability()

    # Point analytics store to a regular file so table directory creation fails.
    bad_store_root = local_ray.parent / "not-a-directory"
    bad_store_root.write_text("this is a file")

    configure_backend(
        "ray",
        address="local",
        force_reinit=True,
        analytics_store={"backend": "parquet", "uri": str(bad_store_root)},
    )

    job = submit_capability(capability, config=TinyConfig(text="no-store"), use_cache=False)
    with pytest.raises(JobFailedError):
        job.result(timeout=30)


def test_submit_requires_explicit_backend_configuration() -> None:
    shutdown_backend(wait=False)
    ray.shutdown()

    with pytest.raises(RuntimeError, match="configure_backend"):
        submit_capability(TinyCapability(), config=TinyConfig(text="no-backend"), use_cache=False)


@pytest.mark.ray
def test_repeated_submissions_with_matching_local_cache_do_not_duplicate_runs_mapping_rows(
    local_ray: Path, fake_ic_dataset_default
) -> None:
    capability = TinyDatasetCapability()
    config = TinyConfig(text="cached-repeat")

    # Seed capability cache. Job submission should still submit remote work.
    cached_run = capability.run(datasets=[fake_ic_dataset_default], config=config, use_cache=True)

    job1 = submit_capability(
        capability,
        datasets=[fake_ic_dataset_default],
        config=config,
        use_cache=True,
    )
    assert isinstance(job1, RayJob)
    ref1 = job1.result(timeout=30)

    job2 = submit_capability(
        capability,
        datasets=[fake_ic_dataset_default],
        config=config,
        use_cache=True,
    )
    assert isinstance(job2, RayJob)
    ref2 = job2.result(timeout=30)
    assert ref1.run_uid == cached_run.run_uid
    assert ref2.run_uid == cached_run.run_uid

    payload_rows = _store(local_ray).query_sql("SELECT run_uid FROM tiny_jobs")
    payload_rows = payload_rows.filter(pl.col("run_uid") == cached_run.run_uid)
    assert payload_rows.shape[0] == 1

    run_rows = _store(local_ray).query_sql("SELECT run_uid, entity_type, entity_id FROM runs")
    run_rows = run_rows.filter(pl.col("run_uid") == cached_run.run_uid)
    assert run_rows.shape[0] == 1
