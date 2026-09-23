from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import pytest
import ray
from ray.exceptions import RayActorError, RayTaskError

from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend
from checkmaite.core.report import InlineTextReport
from checkmaite.jobs import (
    BackpressureError,
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
from checkmaite.jobs.backends.ray import (
    JobControllerActor,
    RayJob,
    RegistryCompatibilityError,
    RegistryStartupError,
    RegistryStatus,
    get_or_create_controller_actor,
    get_or_create_registry_actor,
)
from checkmaite.jobs.backends.ray import controller as controller_module
from checkmaite.jobs.backends.ray.controller import _update_registry_terminal_best_effort
from checkmaite.jobs.backends.ray.registry import (
    DEFAULT_REGISTRY_ACTOR_NAME,
    JobRegistry,
    _registry_configuration,
    _registry_descriptor,
)
from tests.test_jobs.fakes import TinyCapability, TinyConfig
from tests.test_jobs.ray_test_utils import init_local_ray

TEST_ARTIFACT_STORE_URI = str((Path.cwd() / ".checkmaite-test-artifacts").resolve())


@pytest.fixture(scope="module", name="ray_registry_runtime")
def _ray_registry_runtime():
    shutdown_job_backend(wait=False)
    ray.shutdown()
    init_local_ray()

    try:
        yield
    finally:
        shutdown_job_backend(wait=False)
        ray.shutdown()


@pytest.fixture
def local_ray_registry(ray_registry_runtime, tmp_path: Path):
    store_path = tmp_path / "analytics-store"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    scope = f"scope-{uuid4().hex}"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    configure_job_backend(
        "ray",
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(store_path)},
        idempotency_scope=scope,
        controller_num_cpus=0.0,
        registry_namespace=namespace,
    )

    try:
        yield {
            "store_path": store_path,
            "namespace": namespace,
            "scope": scope,
        }
    finally:
        shutdown_job_backend(wait=False)


def _store(path: Path) -> AnalyticsStore:
    return AnalyticsStore(ParquetBackend(str(path)))


def _configure_registry_backend(
    *,
    store_path: Path,
    scope: str,
    namespace: str,
    force_reinit: bool = False,
    address: str | None = None,
) -> None:
    configure_job_backend(
        "ray",
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        address=address,
        force_reinit=force_reinit,
        analytics_store={"backend": "parquet", "uri": str(store_path)},
        idempotency_scope=scope,
        controller_num_cpus=0.0,
        registry_namespace=namespace,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _subprocess_env() -> dict[str, str]:
    repo_root = _repo_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        str(path)
        for path in [
            repo_root / "src",
            repo_root,
            env.get("PYTHONPATH", ""),
        ]
        if str(path)
    )
    return env


def _in_process_registry(**kwargs):
    return JobRegistry(**kwargs)


def _registry_record(scope: str, job_id: str, submitted_at_ts: float, status: RegistryStatus):
    return {
        "scope": scope,
        "scoped_run_key": f"key-{job_id}",
        "job_id": job_id,
        "status": status,
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
        "submitted_at_ts": submitted_at_ts,
        "completed_at_ts": submitted_at_ts,
        "reservation_token": None,
        "reservation_expires_at_ts": None,
        "job_name": "unknown",
    }


def _completed_registry_record(scope: str, job_id: str, submitted_at_ts: float):
    record = _registry_record(scope, job_id, submitted_at_ts, RegistryStatus.COMPLETED)
    record["result_ref"] = CapabilityRunRef(
        run_uid=f"run-{job_id}",
        capability_id="tiny",
        store_uri=f"memory://{job_id}.parquet",
        report=InlineTextReport(media_type="text/plain", content=job_id, filename="report.txt"),
    ).model_dump(mode="json")
    return record


def test_registry_list_jobs_limit_status_and_submitted_before_filters() -> None:
    registry = _in_process_registry(
        terminal_job_retention_s=None,
        max_retained_terminal_jobs_per_scope=None,
    )
    scope = "scope"
    other_scope = "other-scope"
    now = time.time()
    old = _registry_record(scope, "old", now - 20, RegistryStatus.COMPLETED)
    middle = _registry_record(scope, "middle", now - 10, RegistryStatus.FAILED)
    new = _registry_record(scope, "new", now - 1, RegistryStatus.COMPLETED)
    other = _registry_record(other_scope, "other", now, RegistryStatus.COMPLETED)
    registry._job_index = {(record["scope"], record["job_id"]): record for record in (old, middle, new, other)}

    assert [record["job_id"] for record in registry.list_jobs(scope)] == ["new", "middle", "old"]
    assert [record["job_id"] for record in registry.list_jobs(scope, limit=2)] == ["new", "middle"]
    assert [record["job_id"] for record in registry.list_jobs(scope, status_filter=RegistryStatus.COMPLETED)] == [
        "new",
        "old",
    ]
    assert [
        record["job_id"]
        for record in registry.list_jobs(
            scope,
            status_filter=[RegistryStatus.COMPLETED, RegistryStatus.FAILED],
            before_submitted_at_ts=now - 5,
        )
    ] == ["middle", "old"]


def _wait_for_path(path: Path, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {path}")


@pytest.mark.ray
def test_registry_duplicate_submission_same_scope_dedupes(local_ray_registry) -> None:
    capability = TinyCapability()

    job1 = submit_capability(capability, config=TinyConfig(text="dedupe"), use_cache=False)
    job2 = submit_capability(capability, config=TinyConfig(text="dedupe"), use_cache=False)

    assert job1.job_id == job2.job_id
    assert job1.job_name == capability.id
    assert job2.job_name == capability.id

    ref1 = job1.result(timeout=30)
    ref2 = job2.result(timeout=30)
    assert ref1.run_uid == ref2.run_uid

    jobs = list_jobs()
    assert {job.job_id for job in jobs} == {job1.job_id}


@pytest.mark.ray
def test_completed_job_remains_canonical_for_duplicate_submit(local_ray_registry) -> None:
    config = TinyConfig(text="completed-canonical")

    first = submit_capability(TinyCapability(), config=config, use_cache=False)
    first_ref = first.result(timeout=30)

    second = submit_capability(TinyCapability(), config=config, use_cache=False)
    second_ref = second.result(timeout=30)

    assert second.job_id == first.job_id
    assert second_ref.run_uid == first_ref.run_uid

    rows = _store(local_ray_registry["store_path"]).query_sql(
        "SELECT run_uid, payload FROM tiny_jobs WHERE payload = 'completed-canonical'"
    )
    assert rows.shape[0] == 1


def test_registry_different_scopes_do_not_collide() -> None:
    registry = _in_process_registry()
    scoped_run_key = "same-logical-run"

    row_a = registry.register_or_get("scope-a", scoped_run_key)
    row_b = registry.register_or_get("scope-b", scoped_run_key)
    row_a_again = registry.register_or_get("scope-a", scoped_run_key)

    assert row_a["decision"] == "new"
    assert row_b["decision"] == "new"
    assert row_a["job_id"] != row_b["job_id"]
    assert row_a_again["decision"] == "existing"
    assert row_a_again["job_id"] == row_a["job_id"]


@pytest.mark.ray
def test_registry_get_job_can_reattach_after_restart(local_ray_registry) -> None:
    capability = TinyCapability()

    job = submit_capability(capability, config=TinyConfig(text="restart", sleep_s=1.0), use_cache=False)
    job_id = job.job_id

    # Simulate client restart: drop active job backend handle and recreate it.
    shutdown_job_backend(wait=False)

    _configure_registry_backend(
        store_path=local_ray_registry["store_path"],
        scope=local_ray_registry["scope"],
        namespace=local_ray_registry["namespace"],
    )

    reattached = get_job(job_id)
    ref = reattached.result(timeout=30)

    assert isinstance(ref, CapabilityRunRef)
    assert ref.report.content == "restart:0.5"


@pytest.mark.ray
def test_registry_job_survives_submitter_process_exit(local_ray_registry) -> None:
    """Executable acceptance test for true kernel/client restart reattach semantics."""
    repo_root = _repo_root()
    env = _subprocess_env()

    start_marker = local_ray_registry["store_path"].parent / "driver-exit-started.txt"
    submitter = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            """
import json
import sys
import time
from pathlib import Path

from checkmaite.jobs import configure_job_backend, submit_capability
from tests.test_jobs.fakes import TinyCapability, TinyConfig

store_path, namespace, scope, start_marker_path = sys.argv[1:5]
start_marker = Path(start_marker_path)
configure_job_backend(
    "ray",
    address="auto",
    analytics_store={"backend": "parquet", "uri": store_path},
    artifact_store={"uri": str(Path(store_path).parent / "artifacts")},
    idempotency_scope=scope,
    controller_num_cpus=0.0,
    registry_namespace=namespace,
)
job = submit_capability(
    TinyCapability(),
    config=TinyConfig(text="driver-exit", sleep_s=5.0, start_marker_path=start_marker_path),
    use_cache=False,
)
deadline = time.monotonic() + 30.0
while not start_marker.exists() and time.monotonic() < deadline:
    time.sleep(0.05)
if not start_marker.exists():
    raise RuntimeError(f"Timed out waiting for {start_marker}")
print(json.dumps({"job_id": job.job_id}), flush=True)
""",
            str(local_ray_registry["store_path"]),
            local_ray_registry["namespace"],
            local_ray_registry["scope"],
            str(start_marker),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert submitter.returncode == 0, submitter.stderr

    job_line = next(line for line in reversed(submitter.stdout.splitlines()) if line.strip().startswith("{"))
    job_id = json.loads(job_line)["job_id"]
    reattached = get_job(job_id)
    ref = reattached.result(timeout=60)

    assert ref.report.content == "driver-exit:0.5"


@pytest.mark.ray
def test_running_job_reattaches_after_backend_restart(local_ray_registry) -> None:
    start_marker = local_ray_registry["store_path"].parent / "running-reattach-started.txt"
    job = submit_capability(
        TinyCapability(),
        config=TinyConfig(text="running-reattach", sleep_s=1.0, start_marker_path=str(start_marker)),
        use_cache=False,
    )
    job_id = job.job_id
    _wait_for_path(start_marker)

    shutdown_job_backend(wait=False)
    _configure_registry_backend(
        store_path=local_ray_registry["store_path"],
        scope=local_ray_registry["scope"],
        namespace=local_ray_registry["namespace"],
    )

    reattached = get_job(job_id)
    assert reattached.result(timeout=30).report.content == "running-reattach:0.5"


@pytest.mark.ray
def test_default_registry_is_shared_across_scopes_but_jobs_remain_partitioned(tmp_path: Path) -> None:
    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    namespace = f"checkmaite-shared-default-{uuid4().hex}"
    backend_a = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(tmp_path / "a")},
        idempotency_scope="workspace-a",
        registry_namespace=namespace,
        controller_num_cpus=0.0,
    )
    backend_b = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(tmp_path / "b")},
        idempotency_scope="workspace-b",
        registry_namespace=namespace,
        controller_num_cpus=0.0,
    )

    assert backend_a._registry._actor_id == backend_b._registry._actor_id
    assert ray.get_actor(DEFAULT_REGISTRY_ACTOR_NAME, namespace=namespace)._actor_id == backend_a._registry._actor_id

    job_a = backend_a.submit_capability(TinyCapability(), config=TinyConfig(text="partitioned"), use_cache=False)
    job_b = backend_b.submit_capability(TinyCapability(), config=TinyConfig(text="partitioned"), use_cache=False)

    assert job_a.job_id != job_b.job_id
    assert {job.job_id for job in backend_a.list_jobs()} == {job_a.job_id}
    assert {job.job_id for job in backend_b.list_jobs()} == {job_b.job_id}
    assert job_a.result(timeout=30).report.content == "partitioned:0.5"
    assert job_b.result(timeout=30).report.content == "partitioned:0.5"


@pytest.mark.ray
def test_different_ray_namespaces_select_independent_fixed_registries(tmp_path: Path) -> None:
    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    backend_a = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(tmp_path / "a")},
        idempotency_scope="same-scope",
        registry_namespace=f"namespace-a-{uuid4().hex}",
        controller_num_cpus=0.0,
    )
    backend_b = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(tmp_path / "b")},
        idempotency_scope="same-scope",
        registry_namespace=f"namespace-b-{uuid4().hex}",
        controller_num_cpus=0.0,
    )

    assert backend_a._registry._actor_id != backend_b._registry._actor_id


@pytest.mark.ray
def test_job_discovery_metadata_and_worker_scheduling_info(local_ray_registry) -> None:
    backend = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(local_ray_registry["store_path"])},
        idempotency_scope=local_ray_registry["scope"],
        controller_num_cpus=0.0,
        registry_namespace=local_ray_registry["namespace"],
    )

    capability = TinyCapability()
    job = backend.submit_capability(
        capability,
        config=TinyConfig(text="metadata"),
        resources={"num_cpus": 1, "num_gpus": 0},
        job_name="September evaluation",
        use_cache=False,
    )
    assert job.result(timeout=30).capability_id == capability.id

    recovered = backend.list_jobs()[0]
    assert recovered.job_name == "September evaluation"
    scheduling = recovered.scheduling_info
    assert scheduling["requested_resources"]["num_cpus"] == 1.0
    assert scheduling["scheduling_started_at"] is not None
    assert scheduling["worker_started_at"] is not None
    assert scheduling["worker_node_id"]


@pytest.mark.ray
def test_unscheduled_worker_request_fails_after_scheduling_timeout(tmp_path: Path) -> None:
    namespace = f"checkmaite-timeout-{uuid4().hex}"
    backend = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(tmp_path / "store")},
        idempotency_scope=f"scope-{uuid4().hex}",
        registry_namespace=namespace,
        controller_num_cpus=0.0,
        scheduling_timeout_s=2.0,
    )

    job = backend.submit_capability(
        TinyCapability(),
        config=TinyConfig(text="never-scheduled"),
        resources={"num_cpus": 10000},
        use_cache=False,
    )
    assert job.status is JobStatus.SCHEDULING
    assert job.wait(timeout=10) is JobStatus.FAILED
    with pytest.raises(JobFailedError, match="not scheduled within"):
        job.result(timeout=1)


@pytest.mark.ray
def test_cancel_resource_blocked_scheduling_job_raises_cancelled(tmp_path: Path) -> None:
    namespace = f"checkmaite-cancel-scheduling-{uuid4().hex}"
    backend = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(tmp_path / "store")},
        idempotency_scope=f"scope-{uuid4().hex}",
        registry_namespace=namespace,
        controller_num_cpus=0.0,
        scheduling_timeout_s=30.0,
    )
    job = backend.submit_capability(
        TinyCapability(),
        config=TinyConfig(text="cancel-before-placement"),
        resources={"num_cpus": 10000},
        use_cache=False,
    )

    assert job.status is JobStatus.SCHEDULING
    assert job.cancel() is True
    assert job.wait(timeout=10) is JobStatus.CANCELLED
    with pytest.raises(JobCancelledError):
        job.result(timeout=1)


@pytest.mark.ray
def test_scope_admission_limit_rejects_an_unbounded_backlog(tmp_path: Path) -> None:
    namespace = f"checkmaite-admission-{uuid4().hex}"
    backend = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(tmp_path / "store")},
        idempotency_scope=f"scope-{uuid4().hex}",
        registry_namespace=namespace,
        controller_num_cpus=0.0,
        max_active_jobs_per_scope=1,
    )
    first = backend.submit_capability(
        TinyCapability(),
        config=TinyConfig(text="first", sleep_s=2.0),
        use_cache=False,
    )

    with pytest.raises(BackpressureError, match="active limit"):
        backend.submit_capability(TinyCapability(), config=TinyConfig(text="second"), use_cache=False)

    assert first.result(timeout=30).capability_id == TinyCapability().id


@pytest.mark.ray
def test_scheduling_admission_limit_rejects_before_controller_creation(tmp_path: Path) -> None:
    namespace = f"checkmaite-scheduling-admission-{uuid4().hex}"
    scope = f"scope-{uuid4().hex}"
    backend = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(tmp_path / "store")},
        idempotency_scope=scope,
        registry_namespace=namespace,
        controller_num_cpus=0.0,
        max_scheduling_jobs_per_scope=1,
    )
    first = backend.submit_capability(
        TinyCapability(),
        config=TinyConfig(text="blocked"),
        resources={"num_cpus": 10000},
        use_cache=False,
    )
    assert first.status is JobStatus.SCHEDULING

    with pytest.raises(BackpressureError, match="scheduling queue limit"):
        backend.submit_capability(TinyCapability(), config=TinyConfig(text="rejected"), use_cache=False)

    records = ray.get(backend._registry.list_jobs.remote(scope))
    assert [record["job_id"] for record in records] == [first.job_id]
    assert first.cancel() is True
    assert first.wait(timeout=10) is JobStatus.CANCELLED


@pytest.mark.ray
def test_terminal_controller_cleanup_is_autonomous_when_retention_is_zero(tmp_path: Path) -> None:
    namespace = f"checkmaite-retirement-{uuid4().hex}"
    scope = f"scope-{uuid4().hex}"
    backend = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(tmp_path / "store")},
        idempotency_scope=scope,
        registry_namespace=namespace,
        controller_num_cpus=0.0,
        controller_retention_s=0.0,
    )
    capability = TinyCapability()
    job = backend.submit_capability(capability, config=TinyConfig(text="retire"), use_cache=False)
    record = ray.get(backend._registry.get_job.remote(scope, job.job_id))
    controller_name = record["controller_actor_name"]
    assert controller_name is not None

    assert job.result(timeout=30).capability_id == capability.id
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            ray.get_actor(controller_name, namespace=namespace)
        except ValueError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("terminal controller was not cleaned autonomously")

    stored = ray.get(backend._registry.get_job.remote(scope, job.job_id))
    assert stored["status"] == RegistryStatus.COMPLETED
    assert stored["controller_actor_name"] is None


@pytest.mark.ray
def test_cross_client_duplicate_submit_dedupes_to_one_running_job(local_ray_registry) -> None:
    backend_a = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(local_ray_registry["store_path"])},
        idempotency_scope=local_ray_registry["scope"],
        controller_num_cpus=0.0,
        registry_namespace=local_ray_registry["namespace"],
    )
    backend_b = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(local_ray_registry["store_path"])},
        idempotency_scope=local_ray_registry["scope"],
        controller_num_cpus=0.0,
        registry_namespace=local_ray_registry["namespace"],
    )

    start_marker = local_ray_registry["store_path"].parent / "cross-client-dedupe-started.txt"
    config = TinyConfig(text="cross-client-dedupe", sleep_s=2.0, start_marker_path=str(start_marker))
    job_1 = backend_a.submit_capability(TinyCapability(), config=config, use_cache=False)
    _wait_for_path(start_marker)
    job_2 = backend_b.submit_capability(TinyCapability(), config=config, use_cache=False)

    assert job_1.job_id == job_2.job_id
    assert job_2.result(timeout=30).report.content == "cross-client-dedupe:0.5"
    assert {job.job_id for job in backend_b.list_jobs()} == {job_1.job_id}

    rows = _store(local_ray_registry["store_path"]).query_sql("SELECT run_uid, payload FROM tiny_jobs")
    rows = rows.filter(rows["payload"] == "cross-client-dedupe")
    assert rows.shape[0] == 1


@pytest.mark.ray
def test_registry_cancel_after_backend_restart(local_ray_registry) -> None:
    start_marker = local_ray_registry["store_path"].parent / "cancel-reattach-started.txt"
    job = submit_capability(
        TinyCapability(),
        config=TinyConfig(text="cancel-reattach", sleep_s=5.0, start_marker_path=str(start_marker)),
        use_cache=False,
    )
    job_id = job.job_id
    _wait_for_path(start_marker)

    shutdown_job_backend(wait=False)
    _configure_registry_backend(
        store_path=local_ray_registry["store_path"],
        scope=local_ray_registry["scope"],
        namespace=local_ray_registry["namespace"],
    )
    reattached = get_job(job_id)

    assert reattached.cancel() is True
    assert get_job(job_id).wait(timeout=30) is JobStatus.CANCELLED

    with pytest.raises(JobCancelledError):
        get_job(job_id).result(timeout=1)


@pytest.mark.ray
def test_registry_shared_list_and_get_across_clients(local_ray_registry) -> None:
    capability = TinyCapability()

    backend_a = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(local_ray_registry["store_path"])},
        idempotency_scope=local_ray_registry["scope"],
        controller_num_cpus=0.0,
        registry_namespace=local_ray_registry["namespace"],
    )
    backend_b = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(local_ray_registry["store_path"])},
        idempotency_scope=local_ray_registry["scope"],
        controller_num_cpus=0.0,
        registry_namespace=local_ray_registry["namespace"],
    )

    job = backend_a.submit_capability(capability, config=TinyConfig(text="shared"), use_cache=False)

    observed = backend_b.get_job(job.job_id)
    assert observed.job_id == job.job_id
    assert any(listed.job_id == job.job_id for listed in backend_b.list_jobs())

    ref = observed.result(timeout=30)
    assert ref.report.content == "shared:0.5"


@pytest.mark.ray
def test_submit_returns_handle_when_final_registry_read_times_out(tmp_path: Path) -> None:
    actor_name = DEFAULT_REGISTRY_ACTOR_NAME
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    scope = f"scope-{uuid4().hex}"
    job_id = uuid4().hex

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    registry_descriptor = _registry_descriptor(_registry_configuration())

    @ray.remote
    class SlowFirstGetRegistry:
        def __init__(self, descriptor: dict[str, Any]) -> None:
            self.descriptor = descriptor
            now = time.time()
            self.get_count = 0
            self.record: dict[str, Any] = {
                "scope": scope,
                "scoped_run_key": "slow-final-read-key",
                "job_id": job_id,
                "status": RegistryStatus.SUBMITTING.value,
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
                "reservation_token": "token",
                "reservation_expires_at_ts": now + 30.0,
                "job_name": "unknown",
                "requested_resources": {},
                "scheduling_started_at_ts": None,
                "scheduling_deadline_at_ts": None,
                "worker_started_at_ts": None,
                "worker_node_id": None,
                "worker_pod_name": None,
                "worker_kubernetes_node_name": None,
            }

        def ping(self) -> bool:
            return True

        def describe(self) -> dict[str, Any]:
            return self.descriptor

        def register_or_get(
            self,
            _scope: str,
            _scoped_run_key: str,
            job_name: str,
            requested_resources: dict[str, Any],
        ) -> dict[str, Any]:
            self.record["job_name"] = job_name
            self.record["requested_resources"] = requested_resources
            out = dict(self.record)
            out["decision"] = "new"
            return out

        def attach_controller(
            self,
            _scope: str,
            _job_id: str,
            token: str,
            controller_actor_name: str,
            controller_namespace: str,
        ) -> bool:
            self.record["controller_actor_name"] = controller_actor_name
            self.record["controller_namespace"] = controller_namespace
            self.record["controller_token"] = token
            return True

        def mark_scheduling(
            self,
            _scope: str,
            _job_id: str,
            _token: str,
            controller_actor_name: str,
            controller_namespace: str,
            scheduling_timeout_s: float | None,
        ) -> bool:
            now = time.time()
            self.record["status"] = RegistryStatus.SCHEDULING.value
            self.record["controller_actor_name"] = controller_actor_name
            self.record["controller_namespace"] = controller_namespace
            self.record["controller_heartbeat_at_ts"] = now
            self.record["controller_lease_expires_at_ts"] = now + 30.0
            self.record["scheduling_started_at_ts"] = now
            self.record["scheduling_deadline_at_ts"] = (
                None if scheduling_timeout_s is None else now + scheduling_timeout_s
            )
            self.record["reservation_token"] = None
            self.record["reservation_expires_at_ts"] = None
            return True

        def mark_worker_running(
            self,
            _scope: str,
            _job_id: str,
            _controller_actor_name: str,
            _controller_token: str,
            worker_info: dict[str, Any],
        ) -> tuple[bool, RegistryStatus | None]:
            self.record["status"] = RegistryStatus.RUNNING.value
            self.record["worker_started_at_ts"] = time.time()
            self.record["worker_node_id"] = worker_info.get("node_id")
            return True, RegistryStatus.RUNNING

        def heartbeat_controller(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[bool, RegistryStatus]:
            return True, RegistryStatus(self.record["status"])

        def update_terminal(
            self,
            _scope: str,
            _job_id: str,
            status: str,
            error: str | None = None,
            result_ref: dict[str, Any] | None = None,
            *_args: object,
        ) -> bool:
            self.record["status"] = status
            self.record["error"] = error
            self.record["result_ref"] = result_ref
            self.record["completed_at_ts"] = time.time()
            return True

        def get_job(self, _scope: str, _job_id: str) -> dict[str, Any]:
            self.get_count += 1
            if self.get_count == 1:
                time.sleep(0.5)
            return dict(self.record)

    try:
        registry = SlowFirstGetRegistry.options(name=actor_name, namespace=namespace, lifetime="detached").remote(
            registry_descriptor
        )
        assert ray.get(registry.ping.remote()) is True
        backend = RayJobBackend(
            artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
            address="local",
            analytics_store={"backend": "parquet", "uri": str(tmp_path / "analytics-store")},
            idempotency_scope=scope,
            controller_num_cpus=0.0,
            registry_namespace=namespace,
            registry_sweep_on_submit=False,
            registry_update_timeout_s=0.2,
        )

        job = backend.submit_capability(TinyCapability(), config=TinyConfig(text="slow-final-read"), use_cache=False)

        assert job.job_id == job_id
        assert job.result(timeout=30).report.content == "slow-final-read:0.5"
    finally:
        shutdown_job_backend(wait=False)


@pytest.mark.ray
def test_registry_submit_failure_closes_reservation(local_ray_registry) -> None:
    class NonSerializableCapability(TinyCapability):
        def __init__(self) -> None:
            self._not_picklable = Lock()

    with pytest.raises(RuntimeError):
        submit_capability(NonSerializableCapability(), config=TinyConfig(text="boom"), use_cache=False)

    jobs = list_jobs()
    assert jobs
    assert jobs[0].status is JobStatus.FAILED


@pytest.mark.ray
def test_registry_submit_failure_releases_dedupe_for_retry(local_ray_registry) -> None:
    class SameIdentityCapability(TinyCapability):
        @property
        def id(self) -> str:
            return "tests.same-identity-capability"

    class NonSerializableSameIdentityCapability(SameIdentityCapability):
        def __init__(self) -> None:
            self._not_picklable = Lock()

    config = TinyConfig(text="retry-after-submit-failure")

    with pytest.raises(RuntimeError):
        submit_capability(NonSerializableSameIdentityCapability(), config=config, use_cache=False)

    failed_job_id = list_jobs()[0].job_id
    retry = submit_capability(SameIdentityCapability(), config=config, use_cache=False)

    assert retry.job_id != failed_job_id
    assert retry.result(timeout=30).report.content == "retry-after-submit-failure:0.5"


@pytest.mark.ray
def test_registry_resource_validation_failure_has_no_ray_side_effect_and_allows_retry(local_ray_registry) -> None:
    config = TinyConfig(text="retry-after-resource-failure")

    before_ids = {job.job_id for job in list_jobs()}
    with pytest.raises(TypeError, match="num_cpus"):
        submit_capability(
            TinyCapability(),
            config=config,
            resources={"num_cpus": "not-an-int"},
            use_cache=False,
        )

    assert {job.job_id for job in list_jobs()} == before_ids
    retry = submit_capability(TinyCapability(), config=config, resources={"num_cpus": 1}, use_cache=False)

    assert retry.job_id not in before_ids
    assert retry.result(timeout=30).report.content == "retry-after-resource-failure:0.5"


@pytest.mark.ray
def test_controller_launch_failure_keeps_terminal_controller_for_retry(local_ray_registry) -> None:
    class FailingControllerStartJobBackend(RayJobBackend):
        def _start_controller(self, **_kwargs: Any) -> Any:
            raise RuntimeError("controller launch failed")

    prefix = f"controller-launch-failure-{uuid4().hex}"
    backend = FailingControllerStartJobBackend(
        analytics_store={"backend": "parquet", "uri": str(local_ray_registry["store_path"])},
        artifact_store={"uri": str(local_ray_registry["store_path"].parent / "artifacts")},
        idempotency_scope=local_ray_registry["scope"],
        controller_num_cpus=0.0,
        registry_namespace=local_ray_registry["namespace"],
        controller_actor_prefix=prefix,
    )

    with pytest.raises(RuntimeError):
        backend.submit_capability(TinyCapability(), config=TinyConfig(text="bad-launch-resource"), use_cache=False)

    failed = backend.list_jobs(status_filter=JobStatus.FAILED)[0]
    assert failed.status is JobStatus.FAILED
    # The failed terminal controller is intentionally retained instead of being
    # killed by the submitter; if its first terminal write timed out, its retry
    # loop can still commit FAILED to the shared registry.
    ray.get_actor(f"{prefix}_{failed.job_id}", namespace=local_ray_registry["namespace"])


@pytest.mark.ray
def test_expired_submitting_reservation_cannot_start_controller_work(tmp_path: Path) -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    scope = f"scope-{uuid4().hex}"
    store_path = tmp_path / "analytics-store"
    start_marker = tmp_path / "expired-reservation-started.txt"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    try:
        registry = get_or_create_registry_actor(
            name=actor_name,
            namespace=namespace,
            reservation_ttl_s=0.001,
        )
        row = ray.get(registry.register_or_get.remote(scope, "expired-key"))
        swept = ray.get(
            registry.sweep_expired_submissions.remote(
                now_ts=float(row["reservation_expires_at_ts"]) + 1.0,
            )
        )
        assert swept == 1

        controller = get_or_create_controller_actor(
            name=f"controller-{row['job_id']}",
            namespace=namespace,
            registry_name=actor_name,
            registry_namespace=namespace,
            scope=scope,
            job_id=row["job_id"],
            registry_update_timeout_s=5.0,
            controller_heartbeat_interval_s=10.0,
            controller_terminal_retry_interval_s=1.0,
            controller_num_cpus=0.01,
            controller_memory=None,
            controller_resources=None,
        )
        state = ray.get(
            controller.start.remote(
                TinyCapability(),
                {
                    "config": TinyConfig(text="should-not-start", start_marker_path=str(start_marker)),
                    "use_cache": False,
                    "_analytics_store": {"backend": "parquet", "uri": str(store_path)},
                },
                {"num_cpus": 1, "num_gpus": 0.0},
                0,
                row["reservation_token"],
            )
        )

        assert state["status"] == RegistryStatus.SUBMITTING.value
        assert not start_marker.exists()
        stored = ray.get(registry.get_job.remote(scope, row["job_id"]))
        assert stored["status"] == RegistryStatus.FAILED.value
    finally:
        shutdown_job_backend(wait=False)


@pytest.mark.ray
def test_cancel_after_controller_attach_before_start_commits_cancelled(tmp_path: Path) -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    scope = f"scope-{uuid4().hex}"
    store_path = tmp_path / "analytics-store"
    start_marker = tmp_path / "cancel-before-start-started.txt"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    try:
        registry = get_or_create_registry_actor(
            name=actor_name,
            namespace=namespace,
            reservation_ttl_s=30.0,
        )
        row = ray.get(registry.register_or_get.remote(scope, "cancel-before-start-key"))
        job_id = row["job_id"]
        token = row["reservation_token"]
        controller_name = f"controller-{job_id}"
        controller = get_or_create_controller_actor(
            name=controller_name,
            namespace=namespace,
            registry_name=actor_name,
            registry_namespace=namespace,
            scope=scope,
            job_id=job_id,
            registry_update_timeout_s=5.0,
            controller_heartbeat_interval_s=10.0,
            controller_terminal_retry_interval_s=1.0,
            controller_num_cpus=0.01,
            controller_memory=None,
            controller_resources=None,
            controller_token=token,
        )
        assert ray.get(registry.attach_controller.remote(scope, job_id, token, controller_name, namespace)) is True

        job = RayJob(
            registry=registry,
            scope=scope,
            job_id=job_id,
            created_at=datetime.now(timezone.utc),
            control_plane_timeout_s=5.0,
            initial_status=JobStatus.PENDING,
        )
        assert job.cancel() is True
        stored = ray.get(registry.get_job.remote(scope, job_id))
        assert stored["status"] == RegistryStatus.CANCELLED.value

        state = ray.get(
            controller.start.remote(
                TinyCapability(),
                {
                    "config": TinyConfig(text="should-not-start", start_marker_path=str(start_marker)),
                    "use_cache": False,
                    "_analytics_store": {"backend": "parquet", "uri": str(store_path)},
                },
                {"num_cpus": 1, "num_gpus": 0.0},
                0,
                token,
            )
        )

        assert state["status"] == RegistryStatus.CANCELLED.value
        assert not start_marker.exists()
    finally:
        shutdown_job_backend(wait=False)


def test_running_job_is_not_swept_by_submission_ttl() -> None:
    registry = _in_process_registry(reservation_ttl_s=30.0)
    scope = "scope-running-not-submitting"
    controller_name = "controller-running-not-submitting"
    namespace = "namespace"

    row = registry.register_or_get(scope, "running-key")
    job_id = row["job_id"]
    token = row["reservation_token"]
    assert registry.attach_controller(scope, job_id, token, controller_name, namespace) is True
    assert registry.mark_running(scope, job_id, token, controller_name, namespace) is True

    swept = registry.sweep_expired_submissions(now_ts=float(row["reservation_expires_at_ts"]) + 100.0)
    assert swept == 0
    stored = registry.get_job(scope, job_id)
    assert stored["status"] is RegistryStatus.RUNNING


@pytest.mark.ray
def test_registry_cancel_updates_shared_state(local_ray_registry) -> None:
    backend_a = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(local_ray_registry["store_path"])},
        idempotency_scope=local_ray_registry["scope"],
        controller_num_cpus=0.0,
        registry_namespace=local_ray_registry["namespace"],
    )
    backend_b = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        analytics_store={"backend": "parquet", "uri": str(local_ray_registry["store_path"])},
        idempotency_scope=local_ray_registry["scope"],
        controller_num_cpus=0.0,
        registry_namespace=local_ray_registry["namespace"],
    )

    start_marker = local_ray_registry["store_path"].parent / "cancel-shared-started.txt"
    job = backend_a.submit_capability(
        TinyCapability(),
        config=TinyConfig(text="cancel-shared", sleep_s=3.0, start_marker_path=str(start_marker)),
        use_cache=False,
    )
    _wait_for_path(start_marker)
    observed = backend_b.get_job(job.job_id)

    assert observed.cancel() is True
    assert backend_a.get_job(job.job_id).wait(timeout=15) is JobStatus.CANCELLED
    assert backend_b.get_job(job.job_id).status is JobStatus.CANCELLED

    with pytest.raises(JobCancelledError):
        backend_b.get_job(job.job_id).result(timeout=1)


def test_registry_requires_explicit_scope(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="idempotency_scope is required"):
        RayJobBackend(
            artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
            analytics_store={"backend": "parquet", "uri": str(tmp_path / "analytics-store")},
            registry_namespace=f"checkmaite-test-ns-{uuid4().hex}",
        )


def test_registry_rejects_unsafe_heartbeat_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="registry_controller_heartbeat_ttl_s"):
        RayJobBackend(
            artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
            analytics_store={"backend": "parquet", "uri": str(tmp_path / "analytics-store")},
            idempotency_scope=f"scope-{uuid4().hex}",
            controller_num_cpus=0.0,
            registry_namespace=f"checkmaite-test-ns-{uuid4().hex}",
            registry_controller_heartbeat_ttl_s=1.0,
            controller_heartbeat_interval_s=1.0,
            registry_update_timeout_s=1.0,
        )


@pytest.mark.ray
def test_registry_terminal_update_is_bounded_best_effort(local_ray_registry) -> None:
    @ray.remote
    class SlowTerminalRegistry:
        def commit_controller_terminal(self, *args, **kwargs) -> None:
            time.sleep(5.0)

    registry = SlowTerminalRegistry.remote()

    started = time.monotonic()
    updated = _update_registry_terminal_best_effort(
        registry,
        scope=local_ray_registry["scope"],
        job_id="job-timeout",
        status="COMPLETED",
        timeout_s=0.05,
    )
    elapsed = time.monotonic() - started

    assert updated.outcome is controller_module._RegistryCallOutcome.UNAVAILABLE
    assert elapsed < 1.0


@pytest.mark.ray
def test_ray_job_commits_controller_terminal_state_before_returning_result(tmp_path: Path) -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    scope = "scope-commit-terminal"
    controller_name = "controller-commit-terminal"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    @ray.remote
    class CompletedController:
        def __init__(self, job_id: str, result_ref: dict[str, object]) -> None:
            self._job_id = job_id
            self._result_ref = result_ref

        def describe(self, _controller_token: str | None = None) -> dict[str, object]:
            return {"compatibility_version": 1, "scope": scope, "job_id": self._job_id}

        def get_state(self, _controller_token: str | None = None) -> dict[str, object]:
            return {
                "job_id": self._job_id,
                "status": RegistryStatus.COMPLETED.value,
                "result_ref": self._result_ref,
                "error": None,
                "terminal_at_ts": time.time(),
            }

    try:
        registry = get_or_create_registry_actor(
            name=actor_name,
            namespace=namespace,
            reservation_ttl_s=30.0,
        )
        row = ray.get(registry.register_or_get.remote(scope, "commit-terminal-key"))
        job_id = row["job_id"]
        token = row["reservation_token"]
        ref = CapabilityRunRef(
            run_uid="run-commit-terminal",
            capability_id="tiny",
            store_uri=str(tmp_path / "payload.parquet"),
            report=InlineTextReport(media_type="text/markdown", content="committed", filename="report.md"),
        )

        controller = CompletedController.options(name=controller_name, namespace=namespace, lifetime="detached").remote(
            job_id,
            ref.model_dump(mode="json"),
        )
        assert ray.get(registry.attach_controller.remote(scope, job_id, token, controller_name, namespace)) is True
        assert ray.get(registry.mark_running.remote(scope, job_id, token, controller_name, namespace)) is True

        job = RayJob(
            registry=registry,
            scope=scope,
            job_id=job_id,
            created_at=datetime.now(timezone.utc),
            control_plane_timeout_s=5.0,
            initial_status=JobStatus.RUNNING,
        )
        assert job.result(timeout=10).run_uid == ref.run_uid

        stored = ray.get(registry.get_job.remote(scope, job_id))
        assert stored["status"] == RegistryStatus.COMPLETED.value
        assert CapabilityRunRef.model_validate(stored["result_ref"]).run_uid == ref.run_uid

        ray.kill(controller, no_restart=True)
        reattached = RayJob(
            registry=registry,
            scope=scope,
            job_id=job_id,
            created_at=job.created_at,
            control_plane_timeout_s=5.0,
            initial_status=JobStatus.RUNNING,
        )
        assert reattached.result(timeout=5).run_uid == ref.run_uid
    finally:
        shutdown_job_backend(wait=False)


@pytest.mark.ray
def test_ray_job_status_and_result_use_bounded_controller_reads() -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    scope = "scope-slow-controller"
    controller_name = "controller-slow"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    @ray.remote
    class SlowController:
        def describe(self, _controller_token: str | None = None) -> dict[str, object]:
            return {"compatibility_version": 1, "scope": scope, "job_id": job_id}

        def get_state(self, _controller_token: str | None = None) -> dict[str, str]:
            time.sleep(5.0)
            return {"status": RegistryStatus.RUNNING.value}

    try:
        registry = get_or_create_registry_actor(
            name=actor_name,
            namespace=namespace,
            reservation_ttl_s=30.0,
        )
        row = ray.get(registry.register_or_get.remote(scope, "slow-controller-key"))
        job_id = row["job_id"]
        token = row["reservation_token"]
        _controller = SlowController.options(name=controller_name, namespace=namespace, lifetime="detached").remote()
        assert ray.get(registry.attach_controller.remote(scope, job_id, token, controller_name, namespace)) is True
        assert ray.get(registry.mark_running.remote(scope, job_id, token, controller_name, namespace)) is True

        job = RayJob(
            registry=registry,
            scope=scope,
            job_id=job_id,
            created_at=datetime.now(timezone.utc),
            control_plane_timeout_s=0.05,
            initial_status=JobStatus.RUNNING,
        )

        started = time.monotonic()
        assert job.status is JobStatus.RUNNING
        assert time.monotonic() - started < 1.0

        started = time.monotonic()
        with pytest.raises(JobTimeoutError):
            job.result(timeout=0.1)
        assert time.monotonic() - started < 1.0
    finally:
        shutdown_job_backend(wait=False)


@pytest.mark.ray
def test_ray_job_records_cancellation_when_controller_is_missing() -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    scope = "scope-missing-controller"
    controller_name = "missing-controller"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    try:
        registry = get_or_create_registry_actor(
            name=actor_name,
            namespace=namespace,
            reservation_ttl_s=30.0,
        )
        row = ray.get(registry.register_or_get.remote(scope, "missing-controller-key"))
        job_id = row["job_id"]
        token = row["reservation_token"]
        assert ray.get(registry.attach_controller.remote(scope, job_id, token, controller_name, namespace)) is True
        assert ray.get(registry.mark_running.remote(scope, job_id, token, controller_name, namespace)) is True

        job = RayJob(
            registry=registry,
            scope=scope,
            job_id=job_id,
            created_at=datetime.now(timezone.utc),
            control_plane_timeout_s=0.05,
            initial_status=JobStatus.RUNNING,
        )

        assert job.status is JobStatus.RUNNING
        assert job.wait(timeout=0.1) is JobStatus.RUNNING
        with pytest.raises(JobTimeoutError):
            job.result(timeout=0.1)
        assert job.cancel() is True

        stored = ray.get(registry.get_job.remote(scope, job_id))
        assert stored["status"] == RegistryStatus.CANCELLING.value

        swept = ray.get(
            registry.sweep_stale_running_jobs.remote(
                scope,
                job_id,
                float(stored["controller_lease_expires_at_ts"]) + 1.0,
            )
        )
        assert swept == 1
        stored = ray.get(registry.get_job.remote(scope, job_id))
        assert stored["status"] == RegistryStatus.CANCELLED.value
    finally:
        shutdown_job_backend(wait=False)


@pytest.mark.ray
def test_registry_get_and_list_sweep_stale_running_jobs() -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    scope = "scope-stale-read-sweep"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    try:
        registry = get_or_create_registry_actor(
            name=actor_name,
            namespace=namespace,
            reservation_ttl_s=30.0,
            controller_heartbeat_ttl_s=0.001,
        )
        rows = [ray.get(registry.register_or_get.remote(scope, f"stale-key-{idx}")) for idx in range(2)]
        for idx, row in enumerate(rows):
            controller_name = f"missing-stale-controller-{idx}"
            assert (
                ray.get(
                    registry.attach_controller.remote(
                        scope, row["job_id"], row["reservation_token"], controller_name, namespace
                    )
                )
                is True
            )
            assert (
                ray.get(
                    registry.mark_running.remote(
                        scope, row["job_id"], row["reservation_token"], controller_name, namespace
                    )
                )
                is True
            )

        time.sleep(0.01)

        listed = ray.get(registry.list_jobs.remote(scope))
        assert {record["job_id"]: record["status"] for record in listed}[
            rows[0]["job_id"]
        ] == RegistryStatus.FAILED.value

        stored = ray.get(registry.get_job.remote(scope, rows[1]["job_id"]))
        assert stored["status"] == RegistryStatus.FAILED.value
    finally:
        shutdown_job_backend(wait=False)


@pytest.mark.ray
def test_controller_heartbeat_keeps_running_job_from_being_swept(tmp_path: Path) -> None:
    store_path = tmp_path / "analytics-store"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    scope = f"scope-{uuid4().hex}"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    backend = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        address="local",
        analytics_store={"backend": "parquet", "uri": str(store_path)},
        idempotency_scope=scope,
        controller_num_cpus=0.0,
        registry_namespace=namespace,
        registry_controller_heartbeat_ttl_s=6.0,
        controller_heartbeat_interval_s=0.05,
        registry_update_timeout_s=5.0,
    )
    try:
        start_marker = tmp_path / "heartbeat-started.txt"
        job = backend.submit_capability(
            TinyCapability(),
            config=TinyConfig(text="heartbeat", sleep_s=2.0, start_marker_path=str(start_marker)),
            use_cache=False,
        )
        _wait_for_path(start_marker)
        time.sleep(0.2)

        swept = ray.get(backend._registry.sweep_stale_running_jobs.remote(scope, job.job_id))
        assert swept == 0
        stored = ray.get(backend._registry.get_job.remote(scope, job.job_id))
        assert stored["status"] in {RegistryStatus.RUNNING.value, RegistryStatus.COMPLETED.value}
        assert job.result(timeout=30).report.content == "heartbeat:0.5"
    finally:
        backend.shutdown(wait=False)
        shutdown_job_backend(wait=False)


@pytest.mark.ray
def test_cancel_records_cancelling_when_ack_times_out() -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    scope = "scope-slow-cancel"
    controller_name = "controller-slow-cancel"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    @ray.remote
    class SlowCancelController:
        def describe(self, _controller_token: str | None = None) -> dict[str, object]:
            return {"compatibility_version": 1, "scope": scope, "job_id": job_id}

        def get_state(self, _controller_token: str | None = None) -> dict[str, str]:
            return {"status": RegistryStatus.RUNNING.value}

        def cancel(self, _controller_token: str | None = None) -> bool:
            time.sleep(5.0)
            return True

    try:
        registry = get_or_create_registry_actor(
            name=actor_name,
            namespace=namespace,
            reservation_ttl_s=30.0,
        )
        row = ray.get(registry.register_or_get.remote(scope, "slow-cancel-key"))
        job_id = row["job_id"]
        token = row["reservation_token"]
        _controller = SlowCancelController.options(
            name=controller_name,
            namespace=namespace,
            lifetime="detached",
        ).remote()
        assert ray.get(registry.attach_controller.remote(scope, job_id, token, controller_name, namespace)) is True
        assert ray.get(registry.mark_running.remote(scope, job_id, token, controller_name, namespace)) is True

        job = RayJob(
            registry=registry,
            scope=scope,
            job_id=job_id,
            created_at=datetime.now(timezone.utc),
            control_plane_timeout_s=0.05,
            initial_status=JobStatus.RUNNING,
        )
        assert job.cancel() is True

        stored = ray.get(registry.get_job.remote(scope, job_id))
        assert stored["status"] == RegistryStatus.CANCELLING.value
    finally:
        shutdown_job_backend(wait=False)


def test_list_jobs_is_limited_and_caches_listed_handles(monkeypatch) -> None:
    class RemoteListJobs:
        def __init__(self, registry) -> None:
            self._registry = registry

        def remote(self, *args):
            return self._registry._list_jobs(*args)

    class FakeRegistry:
        def __init__(self, records) -> None:
            self.records = records
            self.calls = []
            self.list_jobs = RemoteListJobs(self)

        def _list_jobs(self, scope, limit, status_filter, submitted_before_ts):
            self.calls.append((scope, limit, status_filter, submitted_before_ts))
            return self.records

    scope = "scope-list-handles"
    now = time.time()
    records = [
        _completed_registry_record(scope, "listed-2", now - 1),
        _completed_registry_record(scope, "listed-1", now - 2),
    ]
    registry = FakeRegistry(records)
    monkeypatch.setattr(ray, "get", lambda value, timeout=None: value)

    backend = object.__new__(RayJobBackend)
    backend._registry = registry
    backend._idempotency_scope = scope
    backend._registry_update_timeout_s = 5.0
    backend._jobs = {}

    listed = backend.list_jobs(limit=2, status_filter=JobStatus.COMPLETED)

    assert [job.job_id for job in listed] == ["listed-2", "listed-1"]
    assert set(backend._jobs) == {"listed-2", "listed-1"}
    assert listed == [backend._jobs["listed-2"], backend._jobs["listed-1"]]
    assert registry.calls == [(scope, 2, RegistryStatus.COMPLETED, None)]


@pytest.mark.ray
def test_registry_sweeps_retained_terminal_job_records(tmp_path: Path) -> None:
    store_path = tmp_path / "analytics-store"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    scope = f"scope-{uuid4().hex}"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    backend = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        address="local",
        analytics_store={"backend": "parquet", "uri": str(store_path)},
        idempotency_scope=scope,
        controller_num_cpus=0.0,
        registry_namespace=namespace,
        terminal_job_retention_s=0.0,
    )
    try:
        job = backend.submit_capability(TinyCapability(), config=TinyConfig(text="purge-record"), use_cache=False)
        job.result(timeout=30)

        assert ray.get(backend._registry.get_job.remote(scope, job.job_id)) is not None
        swept = ray.get(backend._registry.sweep_retained_job_records.remote(scope, time.time() + 1.0))
        assert swept == 1
        assert ray.get(backend._registry.get_job.remote(scope, job.job_id)) is None
    finally:
        backend.shutdown(wait=False)
        shutdown_job_backend(wait=False)


@pytest.mark.ray
def test_controller_retries_terminal_registry_commit(tmp_path: Path) -> None:
    actor_name = f"flaky-terminal-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    controller_name = f"controller-terminal-retry-{uuid4().hex}"
    scope = "scope-terminal-retry"
    job_id = uuid4().hex
    store_path = tmp_path / "analytics-store"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    @ray.remote
    class FlakyTerminalRegistry:
        def __init__(self) -> None:
            self.update_count = 0
            self.status = None

        def mark_scheduling(self, *args) -> bool:
            return True

        def mark_worker_running(self, *args) -> tuple[bool, RegistryStatus | None]:
            return True, RegistryStatus.RUNNING

        def heartbeat_controller(self, *args, **kwargs) -> tuple[bool, RegistryStatus]:
            return True, RegistryStatus.RUNNING

        def commit_controller_terminal(self, _scope, _job_id, status, *_args) -> RegistryStatus:
            self.update_count += 1
            if self.update_count == 1:
                raise RuntimeError("first terminal update fails")
            self.status = status
            return RegistryStatus(status)

        def snapshot(self) -> dict[str, object]:
            return {"update_count": self.update_count, "status": self.status}

    try:
        registry = FlakyTerminalRegistry.options(name=actor_name, namespace=namespace, lifetime="detached").remote()
        controller = JobControllerActor.options(name=controller_name, namespace=namespace, lifetime="detached").remote(
            actor_name=controller_name,
            registry_name=actor_name,
            registry_namespace=namespace,
            scope=scope,
            job_id=job_id,
            registry_update_timeout_s=1.0,
            controller_heartbeat_interval_s=0.05,
            controller_terminal_retry_interval_s=0.05,
        )

        ray.get(
            controller.start.remote(
                TinyCapability(),
                {
                    "config": TinyConfig(text="terminal-retry"),
                    "use_cache": False,
                    "_analytics_store": {"backend": "parquet", "uri": str(store_path)},
                    "_artifact_store": {"uri": TEST_ARTIFACT_STORE_URI},
                },
                {"num_cpus": 1, "num_gpus": 0.0},
                0,
                "token",
            ),
            timeout=10,
        )

        deadline = time.monotonic() + 10.0
        observed = {"update_count": 0, "status": None}
        while time.monotonic() < deadline:
            observed = ray.get(registry.snapshot.remote())
            if observed["update_count"] >= 2 and observed["status"] == RegistryStatus.COMPLETED.value:
                break
            time.sleep(0.05)

        assert observed["update_count"] >= 2
        assert observed["status"] == RegistryStatus.COMPLETED.value
    finally:
        shutdown_job_backend(wait=False)


@pytest.mark.ray
def test_registry_report_threshold_is_not_part_of_submission_identity(local_ray_registry) -> None:
    config = TinyConfig(text="threshold-identity")

    first = submit_capability(TinyCapability(), config=config, report_threshold=0.5, use_cache=False)
    first_ref = first.result(timeout=30)

    second = submit_capability(TinyCapability(), config=config, report_threshold=0.9, use_cache=False)
    second_ref = second.result(timeout=30)

    # Dedupe key is run_uid-only, so both submits resolve to the same job.
    assert second.job_id == first.job_id
    assert second_ref.run_uid == first_ref.run_uid
    # The first submission's report threshold determines the stored summary.
    assert first_ref.report.content == "threshold-identity:0.5"
    assert second_ref.report.content == "threshold-identity:0.5"


@pytest.mark.ray
def test_registry_sweeps_retained_terminal_controllers(tmp_path: Path) -> None:
    store_path = tmp_path / "analytics-store"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    scope = f"scope-{uuid4().hex}"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    backend = RayJobBackend(
        artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
        address="local",
        analytics_store={"backend": "parquet", "uri": str(store_path)},
        idempotency_scope=scope,
        controller_num_cpus=0.0,
        registry_namespace=namespace,
        controller_retention_s=3600.0,
        max_retained_terminal_controllers=0,
    )
    try:
        job = backend.submit_capability(TinyCapability(), config=TinyConfig(text="cleanup"), use_cache=False)
        ref = job.result(timeout=30)

        before = ray.get(backend._registry.get_job.remote(scope, job.job_id))
        assert before["status"] == "COMPLETED"
        assert before["controller_actor_name"] is not None

        swept = ray.get(backend._registry.sweep_terminal_controllers.remote())
        assert swept == 1

        after = ray.get(backend._registry.get_job.remote(scope, job.job_id))
        assert after["status"] == "COMPLETED"
        assert after["controller_actor_name"] is None
        assert CapabilityRunRef.model_validate(after["result_ref"]).run_uid == ref.run_uid
    finally:
        backend.shutdown(wait=False)
        shutdown_job_backend(wait=False)


@pytest.mark.ray
def test_registry_failed_terminal_state_propagates_after_restart(local_ray_registry) -> None:
    failed = submit_capability(TinyCapability(), config=TinyConfig(fail=True), use_cache=False)
    with pytest.raises(JobFailedError):
        failed.result(timeout=30)

    # Reattach after local restart and verify shared failed terminal state.
    shutdown_job_backend(wait=False)
    _configure_registry_backend(
        store_path=local_ray_registry["store_path"],
        scope=local_ray_registry["scope"],
        namespace=local_ray_registry["namespace"],
    )

    failed_again = get_job(failed.job_id)
    assert failed_again.status is JobStatus.FAILED
    with pytest.raises(JobFailedError):
        failed_again.result(timeout=30)


@pytest.mark.ray
def test_registry_result_contract_and_store_semantics_preserved(local_ray_registry) -> None:
    capability = TinyCapability()

    job = submit_capability(capability, config=TinyConfig(text="store"), use_cache=False)
    ref = job.result(timeout=30)

    assert isinstance(ref, CapabilityRunRef)
    assert ref.outputs_uri is None
    assert ref.store_uri.endswith(".parquet")

    result = _store(local_ray_registry["store_path"]).query_sql("SELECT run_uid, payload FROM tiny_jobs")
    assert result.shape[0] == 1
    assert result["run_uid"][0] == ref.run_uid


@pytest.mark.ray
def test_registry_store_write_failure_still_fails_job(local_ray_registry) -> None:
    capability = TinyCapability()

    bad_store_root = local_ray_registry["store_path"].parent / "not-a-directory"
    bad_store_root.write_text("this is a file")

    _configure_registry_backend(
        store_path=bad_store_root,
        scope=local_ray_registry["scope"],
        namespace=local_ray_registry["namespace"],
        force_reinit=True,
    )

    job = submit_capability(capability, config=TinyConfig(text="no-store"), use_cache=False)
    with pytest.raises(JobFailedError):
        job.result(timeout=30)


def test_registry_terminal_update_requires_attached_controller_owner() -> None:
    registry = _in_process_registry(reservation_ttl_s=30.0)
    namespace = "namespace"
    scope = "scope-owner"
    controller_name = "controller-owner"

    row = registry.register_or_get(scope, "owner-key")
    job_id = row["job_id"]
    token = row["reservation_token"]

    assert registry.attach_controller(scope, job_id, token, controller_name, namespace) is True
    assert registry.mark_running(scope, job_id, token, controller_name, namespace) is True

    assert registry.update_terminal(scope, job_id, RegistryStatus.FAILED) is False
    stored = registry.get_job(scope, job_id)
    assert stored["status"] is RegistryStatus.RUNNING

    assert (
        registry.update_terminal(
            scope,
            job_id,
            RegistryStatus.FAILED,
            None,
            None,
            "wrong-controller",
            token,
        )
        is False
    )
    stored = registry.get_job(scope, job_id)
    assert stored["status"] is RegistryStatus.RUNNING

    assert (
        registry.update_terminal(
            scope,
            job_id,
            RegistryStatus.FAILED,
            "owned failure",
            None,
            controller_name,
            token,
        )
        is True
    )
    stored = registry.get_job(scope, job_id)
    assert stored["status"] is RegistryStatus.FAILED
    assert stored["error"] == "owned failure"


def test_registry_invalid_completed_result_ref_fails_without_storing_payload() -> None:
    registry = _in_process_registry(reservation_ttl_s=30.0)
    namespace = "namespace"
    scope = "scope-invalid-result"
    controller_name = "controller-invalid-result"

    row = registry.register_or_get(scope, "invalid-result-key")
    job_id = row["job_id"]
    token = row["reservation_token"]

    assert registry.attach_controller(scope, job_id, token, controller_name, namespace) is True
    assert registry.mark_running(scope, job_id, token, controller_name, namespace) is True

    assert (
        registry.update_terminal(
            scope,
            job_id,
            RegistryStatus.COMPLETED,
            None,
            {"run_uid": "missing-required-fields"},
            controller_name,
            token,
        )
        is True
    )

    stored = registry.get_job(scope, job_id)
    assert stored["status"] is RegistryStatus.FAILED
    assert stored["result_ref"] is None

    retry = registry.register_or_get(scope, "invalid-result-key")
    assert retry["decision"] == "new"
    assert retry["job_id"] != job_id


def test_registry_sweep_expired_submissions_honors_limit() -> None:
    registry = _in_process_registry(reservation_ttl_s=30.0)
    scope = "scope-sweep-limit"

    rows = [registry.register_or_get(scope, f"key-{idx}") for idx in range(3)]
    now_ts = max(float(row["reservation_expires_at_ts"]) for row in rows) + 1.0

    assert registry.sweep_expired_submissions(now_ts, 2) == 2
    failed = [registry.get_job(scope, row["job_id"])["status"] is RegistryStatus.FAILED for row in rows]
    assert sum(failed) == 2


@pytest.mark.ray
@pytest.mark.usefixtures("ray_registry_runtime")
def test_registry_actor_reuse_rejects_legacy_actor_without_handshake() -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"

    @ray.remote
    class LegacyRegistry:
        def ping(self) -> bool:
            return True

    registry = LegacyRegistry.options(name=actor_name, namespace=namespace, lifetime="detached").remote()
    try:
        assert ray.get(registry.ping.remote()) is True
        with pytest.raises(RegistryCompatibilityError, match="does not support.*registry handshake"):
            get_or_create_registry_actor(name=actor_name, namespace=namespace, reservation_ttl_s=10.0)
    finally:
        ray.kill(registry, no_restart=True)


@pytest.mark.ray
@pytest.mark.usefixtures("ray_registry_runtime")
def test_registry_actor_reuse_rejects_invalid_handshake_payload() -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"

    @ray.remote
    class InvalidRegistry:
        def describe(self) -> list[object]:
            return []

    registry = InvalidRegistry.options(name=actor_name, namespace=namespace, lifetime="detached").remote()
    try:
        with pytest.raises(RegistryCompatibilityError, match="invalid compatibility descriptor"):
            get_or_create_registry_actor(name=actor_name, namespace=namespace, reservation_ttl_s=10.0)
    finally:
        ray.kill(registry, no_restart=True)


@pytest.mark.ray
@pytest.mark.usefixtures("ray_registry_runtime")
def test_registry_actor_death_during_handshake_is_startup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    get_or_create_registry_actor(name=actor_name, namespace=namespace, reservation_ttl_s=10.0)
    real_get_actor = ray.get_actor

    def get_then_kill(name: str, *, namespace: str | None = None):
        registry = real_get_actor(name, namespace=namespace)
        ray.kill(registry, no_restart=True)
        return registry

    monkeypatch.setattr(ray, "get_actor", get_then_kill)

    with pytest.raises(RegistryStartupError, match="failed during its startup handshake") as exc_info:
        get_or_create_registry_actor(name=actor_name, namespace=namespace, reservation_ttl_s=10.0)

    assert isinstance(exc_info.value.__cause__, RayActorError)


@pytest.mark.ray
@pytest.mark.usefixtures("ray_registry_runtime")
def test_registry_describe_task_failure_is_startup_failure() -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"

    @ray.remote
    class FailingRegistry:
        def describe(self) -> dict[str, object]:
            raise RuntimeError("temporary describe failure")

    registry = FailingRegistry.options(name=actor_name, namespace=namespace, lifetime="detached").remote()
    try:
        with pytest.raises(RegistryStartupError, match="failed during its startup handshake") as exc_info:
            get_or_create_registry_actor(name=actor_name, namespace=namespace, reservation_ttl_s=10.0)
        assert isinstance(exc_info.value.__cause__, RayTaskError)
    finally:
        ray.kill(registry, no_restart=True)


@pytest.mark.ray
@pytest.mark.usefixtures("ray_registry_runtime")
def test_registry_describe_timeout_is_startup_failure() -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"

    @ray.remote
    class SlowRegistry:
        def describe(self) -> dict[str, object]:
            time.sleep(1.0)
            return {}

    registry = SlowRegistry.options(name=actor_name, namespace=namespace, lifetime="detached").remote()
    try:
        with pytest.raises(RegistryStartupError, match="did not become ready"):
            get_or_create_registry_actor(
                name=actor_name,
                namespace=namespace,
                reservation_ttl_s=10.0,
                startup_timeout_s=0.01,
            )
    finally:
        ray.kill(registry, no_restart=True)


@pytest.mark.ray
@pytest.mark.usefixtures("ray_registry_runtime")
def test_registry_reuse_does_not_compare_handle_pending_call_limits(tmp_path: Path) -> None:
    namespace = f"checkmaite-test-ns-{uuid4().hex}"
    registry = get_or_create_registry_actor(
        name=DEFAULT_REGISTRY_ACTOR_NAME,
        namespace=namespace,
        reservation_ttl_s=60.0,
        registry_max_pending_calls=None,
    )
    backend: RayJobBackend | None = None
    try:
        reattached = get_or_create_registry_actor(
            name=DEFAULT_REGISTRY_ACTOR_NAME,
            namespace=namespace,
            reservation_ttl_s=60.0,
            registry_max_pending_calls=2048,
        )
        assert ray.get(reattached.describe.remote())["configuration"] == _registry_configuration()

        backend = RayJobBackend(
            artifact_store={"uri": TEST_ARTIFACT_STORE_URI},
            analytics_store={"backend": "parquet", "uri": str(tmp_path / "analytics-store")},
            idempotency_scope=f"scope-{uuid4().hex}",
            registry_namespace=namespace,
            controller_num_cpus=0.0,
        )
        assert backend.list_jobs() == []
    finally:
        if backend is not None:
            backend.shutdown(wait=False)
        ray.kill(registry, no_restart=True)


@pytest.mark.ray
def test_registry_actor_reuse_rejects_incompatible_configuration() -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    get_or_create_registry_actor(
        name=actor_name,
        namespace=namespace,
        reservation_ttl_s=10.0,
    )

    with pytest.raises(RegistryCompatibilityError, match="reservation_ttl_s"):
        get_or_create_registry_actor(
            name=actor_name,
            namespace=namespace,
            reservation_ttl_s=20.0,
        )


@pytest.mark.ray
def test_registry_actor_sweep_expired_submissions_marks_failed(tmp_path: Path) -> None:
    actor_name = f"checkmaite-test-registry-{uuid4().hex}"
    namespace = f"checkmaite-test-ns-{uuid4().hex}"

    shutdown_job_backend(wait=False)
    if not ray.is_initialized():
        init_local_ray()

    registry = get_or_create_registry_actor(
        name=actor_name,
        namespace=namespace,
        reservation_ttl_s=0.001,
    )

    scope = "scope-sweep"
    row = ray.get(registry.register_or_get.remote(scope, "key-1"))
    assert row["status"] == RegistryStatus.SUBMITTING.value

    swept = ray.get(
        registry.sweep_expired_submissions.remote(
            now_ts=float(row["reservation_expires_at_ts"]) + 1.0,
        )
    )
    assert swept == 1

    stored = ray.get(registry.get_job.remote(scope, row["job_id"]))
    assert stored is not None
    assert stored["status"] == RegistryStatus.FAILED.value

    retry_row = ray.get(registry.register_or_get.remote(scope, "key-1"))
    assert retry_row["decision"] == "new"
    assert retry_row["job_id"] != row["job_id"]

    shutdown_job_backend(wait=False)
