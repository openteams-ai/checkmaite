from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest
import ray

from checkmaite.jobs import CapabilityRunRef, JobFailedError, JobStatus
from checkmaite.jobs.backends.ray_simple.job_backend import RaySimpleJob


def _ref(text: str = "ok") -> CapabilityRunRef:
    return CapabilityRunRef(
        run_uid=f"run-{text}",
        capability_id="tiny",
        store_uri=f"memory://{text}",
        outputs_uri=None,
        summary={"text": text},
    )


def _job(obj_ref) -> RaySimpleJob:
    return RaySimpleJob("job-1", datetime.now(timezone.utc), obj_ref)


def test_ray_simple_job_status_uses_local_polling_heuristic_for_non_ready_ref(monkeypatch) -> None:
    obj_ref = object()
    job = _job(obj_ref)

    def fake_wait(refs, timeout):
        assert refs == [obj_ref]
        assert timeout == 0
        return [], refs

    def fail_get(*args, **kwargs):
        raise AssertionError("ray.get should not be called for a non-ready ref")

    monkeypatch.setattr(ray, "wait", fake_wait)
    monkeypatch.setattr(ray, "get", fail_get)

    assert job.status is JobStatus.PENDING
    assert job.status is JobStatus.RUNNING


@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_ray_simple_job_result_validates_dict_payload_and_terminal_cancel_is_false() -> None:
    job = _job(ray.put(_ref("dict").model_dump(mode="json")))

    ref = job.result(timeout=30)

    assert ref.run_uid == "run-dict"
    assert job.status is JobStatus.COMPLETED
    assert job.cancel() is False
    assert job.exception() is None


@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_ray_simple_job_wait_timeout_returns_latest_non_terminal_status() -> None:
    obj_ref = ray.remote(lambda: time.sleep(0.5)).remote()
    job = _job(obj_ref)

    status = job.wait(timeout=0.01)

    assert status is JobStatus.RUNNING
    ray.get(obj_ref, timeout=30)


@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_ray_simple_job_invalid_ready_payload_maps_to_failed_result() -> None:
    job = _job(ray.put({"not": "a CapabilityRunRef"}))

    with pytest.raises(JobFailedError, match="validation"):
        job.result(timeout=30)

    assert job.status is JobStatus.FAILED
    assert job.exception() is not None
