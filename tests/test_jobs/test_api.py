from __future__ import annotations

import pytest

from checkmaite.jobs import JobStatus, _api
from tests.test_jobs.fakes import TinyCapability, TinyConfig


class RecordingJobBackend:
    def __init__(self) -> None:
        self.submissions = []
        self.list_calls = []
        self.get_calls = []
        self.shutdown_calls = []
        self.job = object()

    def submit_capability(self, capability, **kwargs):
        self.submissions.append((capability, kwargs))
        return self.job

    def list_jobs(self, **kwargs):
        self.list_calls.append(kwargs)
        return [self.job]

    def get_job(self, job_id: str):
        self.get_calls.append(job_id)
        return self.job

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_calls.append(wait)


class FakeRayJobBackend(RecordingJobBackend):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs


class FakeRaySimpleJobBackend(RecordingJobBackend):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def _reset_active_job_backend():
    previous = _api._active_job_backend
    _api._active_job_backend = None
    try:
        yield
    finally:
        if _api._active_job_backend is not None:
            _api._active_job_backend.shutdown(wait=False)
        _api._active_job_backend = previous


def test_submit_and_get_require_configured_backend() -> None:
    with pytest.raises(RuntimeError, match="No active job backend"):
        _api.submit_capability(TinyCapability(), config=TinyConfig())

    with pytest.raises(RuntimeError, match="No active job backend"):
        _api.get_job("missing")


def test_list_jobs_returns_empty_without_configured_backend() -> None:
    assert _api.list_jobs() == []


def test_api_helpers_forward_to_active_job_backend() -> None:
    backend = RecordingJobBackend()
    _api._active_job_backend = backend
    capability = TinyCapability()
    config = TinyConfig(text="forward")

    submitted = _api.submit_capability(
        capability,
        models=["model"],
        datasets=["dataset"],
        metrics=["metric"],
        config=config,
        use_cache=False,
        extra="value",
    )
    listed = _api.list_jobs(limit=5, status_filter=JobStatus.COMPLETED, submitted_before_ts=123.0)
    fetched = _api.get_job("job-1")

    assert submitted is backend.job
    assert listed == [backend.job]
    assert fetched is backend.job
    assert backend.submissions == [
        (
            capability,
            {
                "models": ["model"],
                "datasets": ["dataset"],
                "metrics": ["metric"],
                "config": config,
                "use_cache": False,
                "extra": "value",
            },
        )
    ]
    assert backend.list_calls == [
        {
            "limit": 5,
            "status_filter": JobStatus.COMPLETED,
            "submitted_before_ts": 123.0,
        }
    ]
    assert backend.get_calls == ["job-1"]


def test_submit_capability_defaults_to_no_cache_for_job_submission() -> None:
    backend = RecordingJobBackend()
    _api._active_job_backend = backend

    _api.submit_capability(TinyCapability(), config=TinyConfig())

    assert backend.submissions[0][1]["use_cache"] is False


def test_submit_capability_rejects_cache_for_job_submission() -> None:
    backend = RecordingJobBackend()
    _api._active_job_backend = backend

    with pytest.raises(ValueError, match="use_cache=True is not supported"):
        _api.submit_capability(TinyCapability(), config=TinyConfig(), use_cache=True)

    assert backend.submissions == []


def test_shutdown_job_backend_noops_when_unconfigured_and_clears_active_job_backend() -> None:
    _api.shutdown_job_backend(wait=True)

    backend = RecordingJobBackend()
    _api._active_job_backend = backend

    _api.shutdown_job_backend(wait=False)

    assert backend.shutdown_calls == [False]
    assert _api._active_job_backend is None


def test_configure_job_backend_rejects_unknown_backend_after_shutting_down_existing_backend(tmp_path) -> None:
    backend = RecordingJobBackend()
    _api._active_job_backend = backend

    with pytest.raises(ValueError, match="Unknown job backend"):
        _api.configure_job_backend("unknown", analytics_store={"backend": "parquet", "uri": str(tmp_path / "store")})

    assert backend.shutdown_calls == [False]


@pytest.mark.parametrize(
    ("kind", "expected_backend_type"),
    [
        ("ray", FakeRayJobBackend),
        ("ray-simple", FakeRaySimpleJobBackend),
    ],
)
def test_configure_job_backend_instantiates_requested_backend(
    monkeypatch,
    tmp_path,
    kind,
    expected_backend_type,
) -> None:
    monkeypatch.setattr(_api, "RayJobBackend", FakeRayJobBackend)
    monkeypatch.setattr(_api, "RaySimpleJobBackend", FakeRaySimpleJobBackend)

    store = {"backend": "parquet", "uri": str(tmp_path / "store")}

    _api.configure_job_backend(kind, analytics_store=store, option="value")

    backend = _api._active_job_backend
    assert isinstance(backend, expected_backend_type)
    assert backend.kwargs == {"analytics_store": store, "option": "value"}


def test_configure_job_backend_shuts_down_existing_job_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(_api, "RayJobBackend", FakeRayJobBackend)
    previous = RecordingJobBackend()
    _api._active_job_backend = previous

    _api.configure_job_backend(
        "ray",
        analytics_store={"backend": "parquet", "uri": str(tmp_path / "store")},
    )

    assert previous.shutdown_calls == [False]
    assert isinstance(_api._active_job_backend, FakeRayJobBackend)
