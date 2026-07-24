from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import pytest
import ray

from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend, configure_provenance, reset_provenance
from checkmaite.jobs import (
    CapabilityRunRef,
    JobCancelledError,
    JobFailedError,
    JobStatus,
    JobTimeoutError,
    RayJobBackend,
    configure_job_backend,
    get_job,
    list_jobs,
    shutdown_job_backend,
    submit_capability,
)
from checkmaite.jobs.backends.ray import RegistryStatus
from tests.test_jobs.fakes import TinyCapability, TinyConfig, TinyDatasetCapability
from tests.test_jobs.ray_test_utils import init_local_ray


@pytest.fixture(scope="module", name="ray_runtime")
def _ray_runtime():
    shutdown_job_backend(wait=False)
    ray.shutdown()
    init_local_ray()

    try:
        yield
    finally:
        shutdown_job_backend(wait=False)
        ray.shutdown()


@pytest.fixture
def local_ray(ray_runtime, tmp_path: Path):
    store_path = tmp_path / "analytics-store"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    configure_job_backend(
        "ray",
        analytics_store={"backend": "parquet", "uri": str(store_path)},
        idempotency_scope=f"scope-{uuid4().hex}",
        controller_num_cpus=0.0,
    )

    try:
        yield store_path
    finally:
        shutdown_job_backend(wait=False)


@pytest.fixture
def isolated_local_ray(tmp_path: Path):
    store_path = tmp_path / "analytics-store"

    shutdown_job_backend(wait=False)
    ray.shutdown()
    init_local_ray()

    configure_job_backend(
        "ray",
        analytics_store={"backend": "parquet", "uri": str(store_path)},
        idempotency_scope=f"scope-{uuid4().hex}",
        controller_num_cpus=0.0,
    )

    try:
        yield store_path
    finally:
        shutdown_job_backend(wait=False)
        ray.shutdown()


def _store(path: Path) -> AnalyticsStore:
    return AnalyticsStore(ParquetBackend(str(path)))


def _wait_for_path(path: Path, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out after {timeout_s:.1f}s waiting for {path}")


@pytest.fixture
def ray_job_backend_smoke(_jobs_smoke_ray_runtime, tmp_path: Path) -> tuple[RayJobBackend, Path]:
    """Direct Ray job backend for the unmarked smoke test.

    ``shutdown_job_backend(wait=False)`` clears any global jobs API backend left by another test without waiting
    for stale jobs or shutting down the Ray runtime owned by ``_jobs_smoke_ray_runtime``.
    """
    store_path = tmp_path / "analytics-store"
    backend = None

    shutdown_job_backend(wait=False)

    try:
        backend = RayJobBackend(
            address=None,
            analytics_store={"backend": "parquet", "uri": str(store_path)},
            idempotency_scope=f"smoke-{uuid4().hex}",
            registry_actor_name=f"registry-{uuid4().hex}",
            registry_namespace=f"checkmaite-smoke-{uuid4().hex}",
            controller_num_cpus=0.0,
            registry_sweep_interval_s=0.0,
        )
        yield backend, store_path
    finally:
        if backend is not None:
            backend.shutdown(wait=False)

        # The smoke test uses the direct RayJobBackend above, not the global jobs API backend.
        # This is defensive cleanup for any global backend another test may have left behind.
        shutdown_job_backend(wait=False)


def test_ray_job_backend_smoke_contract_exercises_default_backend_coverage(
    ray_job_backend_smoke: tuple[RayJobBackend, Path],
) -> None:
    """Exercise the default Ray job backend path with a real local Ray runtime.

    Most Ray tests are marked ``ray`` and excluded from the default test run. This
    single smoke contract intentionally stays unmarked so default coverage includes
    the live backend path without maintaining a fake Ray implementation.
    """
    backend, store_path = ray_job_backend_smoke
    capability = TinyCapability()

    completed = backend.submit_capability(capability, config=TinyConfig(text="smoke"), use_cache=False)
    assert backend.get_job(completed.job_id).job_id == completed.job_id
    assert any(job.job_id == completed.job_id for job in backend.list_jobs())

    ref = completed.result(timeout=30)
    assert ref.report.content == "smoke:0.5"
    assert completed.wait(timeout=1) is JobStatus.COMPLETED
    assert completed.status is JobStatus.COMPLETED
    assert completed.exception() is None

    failed = backend.submit_capability(capability, config=TinyConfig(fail=True), use_cache=False)
    with pytest.raises(JobFailedError, match="tiny capability failure"):
        failed.result(timeout=30)
    assert failed.wait(timeout=1) is JobStatus.FAILED
    assert failed.status is JobStatus.FAILED
    assert failed.exception() is not None

    start_marker = store_path.parent / "cancel-started.txt"
    cancellable = backend.submit_capability(
        capability,
        config=TinyConfig(text="cancel", sleep_s=1.0, start_marker_path=str(start_marker)),
        use_cache=False,
    )
    _wait_for_path(start_marker)
    assert cancellable.cancel() is True
    assert cancellable.wait(timeout=15) is JobStatus.CANCELLED
    with pytest.raises(JobCancelledError):
        cancellable.result(timeout=1)

    completed_jobs = backend.list_jobs(status_filter=JobStatus.COMPLETED)
    assert any(job.job_id == completed.job_id for job in completed_jobs)
    assert all(job.status is JobStatus.COMPLETED for job in completed_jobs)
    assert any(job.job_id == failed.job_id for job in backend.list_jobs(status_filter=JobStatus.FAILED))

    sweep_counts = backend.sweep_registry(limit=10)
    assert set(sweep_counts) == {
        "expired_submissions",
        "stale_running_jobs",
        "retained_job_records",
        "terminal_controllers",
    }
    assert all(isinstance(count, int) for count in sweep_counts.values())

    result = _store(store_path).query_sql("SELECT payload FROM tiny_jobs")
    assert "smoke" in set(result["payload"].to_list())


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
    assert ref.report.content == "hello:0.5"

    assert job.wait(timeout=1) is JobStatus.COMPLETED
    assert job.status is JobStatus.COMPLETED
    assert job.exception() is None

    result = _store(local_ray).query_sql("SELECT run_uid, payload FROM tiny_jobs")
    assert result.shape[0] == 1
    assert result["run_uid"][0] == ref.run_uid
    assert result["payload"][0] == "hello"


@pytest.mark.ray
def test_submitted_ray_job_writes_provenance(local_ray: Path, fake_ic_dataset_default) -> None:
    reset_provenance()
    configure_provenance(user_id="alice", workspace_id="workspace-a", environment="databricks")
    try:
        capability = TinyDatasetCapability()

        job = submit_capability(
            capability,
            datasets=[fake_ic_dataset_default],
            config=TinyConfig(text="provenance"),
            use_cache=False,
        )
        ref = job.result(timeout=30)

        result = _store(local_ray).query_sql(
            """
            SELECT user_id, workspace_id, environment, job_id, backend,
                   submitted_at, completed_at, run_event_id
            FROM runs
            """
        )

        assert result.shape[0] == 1
        assert result["user_id"].to_list() == ["alice"]
        assert result["workspace_id"].to_list() == ["workspace-a"]
        assert result["environment"].to_list() == ["databricks"]
        assert result["job_id"].to_list() == [job.job_id]
        assert result["backend"].to_list() == ["ray"]
        assert result["run_event_id"].to_list() == [job.job_id]
        assert result["submitted_at"].drop_nulls().len() == 1
        assert result["completed_at"].drop_nulls().len() == 1
        assert ref.run_uid
    finally:
        reset_provenance()


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


@pytest.mark.parametrize(
    "status_filter",
    ["completed", ["completed"], [JobStatus.COMPLETED, "failed"]],
)
def test_registry_status_filter_rejects_non_job_status(status_filter) -> None:
    with pytest.raises(TypeError, match="status_filter must be a JobStatus"):
        RayJobBackend._registry_status_filter(status_filter)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("status_filter", "expected"),
    [
        (None, None),
        (JobStatus.PENDING, RegistryStatus.SUBMITTING),
        (JobStatus.RUNNING, [RegistryStatus.RUNNING, RegistryStatus.CANCELLING]),
        (JobStatus.COMPLETED, RegistryStatus.COMPLETED),
        (JobStatus.FAILED, RegistryStatus.FAILED),
        (JobStatus.CANCELLED, RegistryStatus.CANCELLED),
        (
            [JobStatus.COMPLETED, JobStatus.FAILED],
            {RegistryStatus.COMPLETED, RegistryStatus.FAILED},
        ),
    ],
)
def test_registry_status_filter_accepts_job_status_values(status_filter, expected) -> None:
    actual = RayJobBackend._registry_status_filter(status_filter)
    if isinstance(expected, set):
        assert set(actual) == expected
    else:
        assert actual == expected


def test_resource_resolution_priority() -> None:
    class DefaultHintCapability(TinyCapability):
        default_num_cpus = 8
        default_num_gpus = 0.75

    capability = DefaultHintCapability()

    class ConfigHints:
        num_cpus = 4
        num_gpus = 0.5

    explicit = RayJobBackend._resolve_resources(
        capability,
        {
            "resources": {
                "num_cpus": 0.5,
                "num_gpus": 1.5,
                "custom_accelerator": 2,
                "resources": {"node_affinity": 0.25},
            },
            "config": ConfigHints(),
        },
    )
    assert explicit.as_dict() == {
        "num_cpus": 0.5,
        "num_gpus": 1.5,
        "resources": {"custom_accelerator": 2.0, "node_affinity": 0.25},
    }

    from_config = RayJobBackend._resolve_resources(capability, {"config": ConfigHints()})
    assert from_config.as_dict() == {"num_cpus": 4, "num_gpus": 0.5}

    from_capability_defaults = RayJobBackend._resolve_resources(capability, {})
    assert from_capability_defaults.as_dict() == {"num_cpus": 8, "num_gpus": 0.75}

    from_fallback = RayJobBackend._resolve_resources(object(), {})
    assert from_fallback.as_dict() == {"num_cpus": 1, "num_gpus": 0.0}


@pytest.mark.ray
def test_reconfigure_wait_false_does_not_interrupt_inflight_job(local_ray: Path) -> None:
    capability = TinyCapability()

    job = submit_capability(capability, config=TinyConfig(text="inflight", sleep_s=1.5), use_cache=False)

    # Default reconfigure path should be non-blocking and not tear down runtime.
    configure_job_backend(
        "ray",
        analytics_store={"backend": "parquet", "uri": str(local_ray)},
        idempotency_scope=f"scope-{uuid4().hex}",
        controller_num_cpus=0.0,
    )

    ref = job.result(timeout=30)
    assert ref.report.content == "inflight:0.5"


@pytest.mark.ray
def test_backend_reconfigure_applies_new_runtime_env_with_force_reinit(tmp_path: Path) -> None:
    capability = TinyCapability()
    env_key = "CHECKMAITE_TEST_RECONFIG_ENV"
    store_path = tmp_path / "analytics-store"

    try:
        configure_job_backend(
            "ray",
            address="local",
            force_reinit=True,
            analytics_store={"backend": "parquet", "uri": str(store_path)},
            idempotency_scope=f"scope-{uuid4().hex}",
            controller_num_cpus=0.0,
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
        assert ref_one.report.content == "one:0.5"

        configure_job_backend(
            "ray",
            address="local",
            force_reinit=True,
            analytics_store={"backend": "parquet", "uri": str(store_path)},
            idempotency_scope=f"scope-{uuid4().hex}",
            controller_num_cpus=0.0,
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
        assert ref_two.report.content == "two:0.5"
    finally:
        shutdown_job_backend(wait=False)
        ray.shutdown()


@pytest.mark.ray
def test_store_write_failure_raises_by_default(isolated_local_ray: Path) -> None:
    reset_provenance()
    configure_provenance(user_id="alice")
    capability = TinyCapability()

    # Point analytics store to a regular file so table directory creation fails.
    bad_store_root = isolated_local_ray.parent / "not-a-directory"
    bad_store_root.write_text("this is a file")

    configure_job_backend(
        "ray",
        analytics_store={"backend": "parquet", "uri": str(bad_store_root)},
        idempotency_scope=f"scope-{uuid4().hex}",
        controller_num_cpus=0.0,
    )

    try:
        job = submit_capability(capability, config=TinyConfig(text="no-store"), use_cache=False)
        with pytest.raises(JobFailedError):
            job.result(timeout=30)
    finally:
        reset_provenance()


@pytest.mark.ray
def test_submit_requires_explicit_backend_configuration() -> None:
    shutdown_job_backend(wait=False)
    ray.shutdown()

    with pytest.raises(RuntimeError, match="configure_job_backend"):
        submit_capability(TinyCapability(), config=TinyConfig(text="no-backend"), use_cache=False)


@pytest.mark.ray
def test_submit_rejects_use_cache_true_even_with_matching_local_cache(local_ray: Path, fake_ic_dataset_default) -> None:
    capability = TinyDatasetCapability()
    config = TinyConfig(text="cached-repeat")

    cached_run = capability.run(datasets=[fake_ic_dataset_default], config=config, use_cache=True)

    with pytest.raises(ValueError, match="use_cache=True is not supported"):
        submit_capability(
            capability,
            datasets=[fake_ic_dataset_default],
            config=config,
            use_cache=True,
        )

    assert cached_run.run_uid
    assert list_jobs() == []
