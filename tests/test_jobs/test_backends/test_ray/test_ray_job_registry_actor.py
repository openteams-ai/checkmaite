from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
import ray

from checkmaite.core.report import InlineTextReport
from checkmaite.jobs import CapabilityRunRef, JobCancelledError, JobFailedError, JobStatus, shutdown_job_backend
from checkmaite.jobs.backends.ray.job_backend import RayJob
from checkmaite.jobs.backends.ray.registry import RegistryStatus, get_or_create_registry_actor

pytestmark = pytest.mark.ray


def _ref_payload(text: str = "ok") -> dict[str, object]:
    return CapabilityRunRef(
        run_uid=f"run-{text}",
        capability_id="tiny",
        store_uri=f"memory://{text}",
        outputs_uri=None,
        report=InlineTextReport(media_type="text/plain", content=text, filename="report.txt"),
    ).model_dump(mode="json")


def _registry(namespace: str):
    return get_or_create_registry_actor(
        name=f"registry-{uuid4().hex}",
        namespace=namespace,
        reservation_ttl_s=30.0,
        registry_num_cpus=0.0,
    )


def _terminal_job(registry, scope: str, key: str, status: RegistryStatus, *, error: str | None = None):
    registration = ray.get(registry.register_or_get.remote(scope, key), timeout=5)
    result_ref = _ref_payload(key) if status is RegistryStatus.COMPLETED else None
    assert ray.get(
        registry.update_terminal.remote(
            scope,
            registration["job_id"],
            status,
            error,
            result_ref,
            None,
            registration["reservation_token"],
        ),
        timeout=5,
    )
    return RayJob(
        registry=registry,
        scope=scope,
        job_id=registration["job_id"],
        created_at=datetime.now(timezone.utc),
        control_plane_timeout_s=5.0,
    )


@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_ray_job_reads_completed_terminal_record_from_real_registry() -> None:
    shutdown_job_backend(wait=False)
    namespace = f"ray-job-{uuid4().hex}"
    scope = f"scope-{uuid4().hex}"
    registry = _registry(namespace)
    job = _terminal_job(registry, scope, "completed", RegistryStatus.COMPLETED)

    assert job.status is JobStatus.COMPLETED
    assert job.wait(timeout=5) is JobStatus.COMPLETED
    assert job.result(timeout=5).run_uid == "run-completed"
    assert job.exception() is None
    assert job.cancel() is False


@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_ray_job_maps_failed_and_cancelled_terminal_records() -> None:
    shutdown_job_backend(wait=False)
    namespace = f"ray-job-{uuid4().hex}"
    scope = f"scope-{uuid4().hex}"
    registry = _registry(namespace)
    failed = _terminal_job(registry, scope, "failed", RegistryStatus.FAILED, error="failed remotely")
    cancelled = _terminal_job(registry, scope, "cancelled", RegistryStatus.CANCELLED)

    assert failed.status is JobStatus.FAILED
    assert failed.wait(timeout=5) is JobStatus.FAILED
    assert failed.exception() is not None
    with pytest.raises(JobFailedError, match="failed remotely"):
        failed.result(timeout=5)
    assert failed.cancel() is False

    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.wait(timeout=5) is JobStatus.CANCELLED
    assert cancelled.exception() is None
    with pytest.raises(JobCancelledError):
        cancelled.result(timeout=5)
    assert cancelled.cancel() is False


@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_ray_job_missing_registry_record_raises_key_error() -> None:
    shutdown_job_backend(wait=False)
    registry = _registry(f"ray-job-{uuid4().hex}")
    job = RayJob(
        registry=registry,
        scope="scope",
        job_id="missing",
        created_at=datetime.now(timezone.utc),
        control_plane_timeout_s=5.0,
    )

    with pytest.raises(KeyError, match="No job with ID"):
        job.result(timeout=5)
