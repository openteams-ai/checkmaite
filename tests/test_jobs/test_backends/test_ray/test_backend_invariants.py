from __future__ import annotations

import inspect
from datetime import timezone

import pytest

from checkmaite.jobs import (
    BackpressureError,
    CapabilityRunRef,
    JobCancelledError,
    JobFailedError,
    JobStatus,
    RayJob,
    RayJobBackend,
    RaySimpleJobBackend,
)
from checkmaite.jobs.backends.ray import RegistryStatus
from checkmaite.jobs.backends.ray import job_backend as job_backend_module
from tests.test_jobs.fakes import TinyCapability, TinyConfig


def _ref_payload(text: str = "ok") -> dict[str, object]:
    return CapabilityRunRef(
        run_uid=f"run-{text}",
        capability_id="tiny",
        store_uri=f"memory://{text}",
        outputs_uri=None,
        summary={"text": text},
    ).model_dump(mode="json")


def _record(**overrides):
    record = {
        "scope": "scope",
        "scoped_run_key": "key",
        "job_id": "job-1",
        "status": RegistryStatus.SUBMITTING,
        "controller_actor_name": "controller-1",
        "controller_namespace": "namespace",
        "controller_token": "token",
        "controller_created_at_ts": None,
        "controller_heartbeat_at_ts": None,
        "controller_lease_expires_at_ts": None,
        "controller_retain_until_ts": None,
        "controller_cleaned_at_ts": None,
        "cancellation_requested_at_ts": None,
        "result_ref": None,
        "error": None,
        "submitted_at_ts": 123.0,
        "completed_at_ts": None,
        "reservation_token": None,
        "reservation_expires_at_ts": None,
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (RegistryStatus.RUNNING, RegistryStatus.RUNNING),
        ("running", RegistryStatus.RUNNING),
        ("RUNNING", RegistryStatus.RUNNING),
    ],
)
def test_registry_status_from_raw_normalizes_supported_values(raw, expected) -> None:
    assert job_backend_module._registry_status_from_raw(raw) is expected


def test_registry_status_from_raw_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unknown registry status"):
        job_backend_module._registry_status_from_raw("paused")


@pytest.mark.parametrize(
    ("registry_status", "result_ref", "expected"),
    [
        (RegistryStatus.SUBMITTING, None, JobStatus.PENDING),
        (RegistryStatus.RUNNING, None, JobStatus.RUNNING),
        (RegistryStatus.CANCELLING, None, JobStatus.RUNNING),
        (RegistryStatus.FAILED, None, JobStatus.FAILED),
        (RegistryStatus.CANCELLED, None, JobStatus.CANCELLED),
        (RegistryStatus.COMPLETED, _ref_payload(), JobStatus.COMPLETED),
        (RegistryStatus.COMPLETED, None, JobStatus.FAILED),
        (RegistryStatus.COMPLETED, {"not": "a CapabilityRunRef"}, JobStatus.FAILED),
    ],
)
def test_job_status_from_fields_maps_internal_registry_states_to_public_states(
    registry_status,
    result_ref,
    expected,
) -> None:
    assert job_backend_module._job_status_from_fields(registry_status, result_ref) is expected


def test_controller_state_from_raw_normalizes_actor_payload() -> None:
    state = job_backend_module._controller_state_from_raw(
        {
            "job_id": 123,
            "status": "completed",
            "result_ref": _ref_payload("controller"),
            "error": 42,
            "terminal_at_ts": "456.5",
        }
    )

    assert state.as_dict() == {
        "job_id": "123",
        "status": RegistryStatus.COMPLETED,
        "result_ref": _ref_payload("controller"),
        "error": "42",
        "terminal_at_ts": 456.5,
    }


def test_controller_state_from_raw_rejects_non_mapping_result_ref() -> None:
    with pytest.raises(TypeError, match="controller result_ref"):
        job_backend_module._controller_state_from_raw(
            {
                "job_id": "job-1",
                "status": RegistryStatus.COMPLETED,
                "result_ref": "not-a-mapping",
                "error": None,
                "terminal_at_ts": None,
            }
        )


def test_snapshot_from_controller_state_preserves_registry_metadata_and_overlays_live_fields() -> None:
    record = _record(
        job_id="registry-job",
        scoped_run_key="original-key",
        status=RegistryStatus.RUNNING,
        result_ref=None,
        error=None,
    )
    state = job_backend_module.ControllerState(
        job_id="registry-job",
        status=RegistryStatus.COMPLETED,
        result_ref=_ref_payload("done"),
        error="ignored-success-message",
        terminal_at_ts=789.0,
    )

    snapshot = job_backend_module._snapshot_from_controller_state(record, state)

    assert snapshot["scoped_run_key"] == "original-key"
    assert snapshot["job_id"] == "registry-job"
    assert snapshot["status"] is RegistryStatus.COMPLETED
    assert snapshot["result_ref"] == _ref_payload("done")
    assert snapshot["error"] == "ignored-success-message"
    assert snapshot["terminal_at_ts"] == 789.0


def test_snapshot_from_controller_state_rejects_mismatched_job_id() -> None:
    record = _record(job_id="registry-job", status=RegistryStatus.RUNNING)
    state = job_backend_module.ControllerState(
        job_id="controller-job",
        status=RegistryStatus.RUNNING,
        result_ref=None,
        error=None,
        terminal_at_ts=None,
    )

    with pytest.raises(ValueError, match="does not match"):
        job_backend_module._snapshot_from_controller_state(record, state)


def test_snapshot_from_controller_state_does_not_clear_existing_result_or_error_with_none() -> None:
    record = _record(result_ref=_ref_payload("old"), error="old-error")
    state = job_backend_module.ControllerState(
        job_id="job-1",
        status=RegistryStatus.RUNNING,
        result_ref=None,
        error=None,
        terminal_at_ts=None,
    )

    snapshot = job_backend_module._snapshot_from_controller_state(record, state)

    assert snapshot["result_ref"] == _ref_payload("old")
    assert snapshot["error"] == "old-error"


def test_created_at_from_submitted_ts_is_utc_datetime() -> None:
    created = job_backend_module._created_at_from_submitted_ts(123.5)

    assert created.tzinfo is timezone.utc
    assert created.timestamp() == 123.5


def test_poll_delay_backoff_steps_and_caps() -> None:
    assert RayJob._next_poll_delay_s(0.05) == 0.25
    assert RayJob._next_poll_delay_s(0.25) == 1.0
    assert RayJob._next_poll_delay_s(1.0) == 5.0
    assert RayJob._next_poll_delay_s(5.0) == 5.0


@pytest.mark.parametrize("backend_cls", [RayJobBackend, RaySimpleJobBackend])
def test_job_backends_reject_use_cache_true_before_ray_side_effects(backend_cls) -> None:
    backend = object.__new__(backend_cls)

    with pytest.raises(ValueError, match="use_cache=True is not supported"):
        backend.submit_capability(TinyCapability(), config=TinyConfig(), use_cache=True)


def test_list_jobs_tracks_returned_handles_for_shutdown(monkeypatch) -> None:
    record = _record(status=RegistryStatus.RUNNING, submitted_at_ts=123.0)

    class _RemoteListJobs:
        def remote(self, *_args):
            return "list-ref"

    class _Registry:
        list_jobs = _RemoteListJobs()

    backend = object.__new__(RayJobBackend)
    backend._idempotency_scope = "scope"
    backend._registry = _Registry()
    backend._registry_update_timeout_s = 1.0
    backend._jobs = {}

    monkeypatch.setattr(job_backend_module.ray, "get", lambda _ref, timeout=None: [record])

    jobs = backend.list_jobs()

    assert len(jobs) == 1
    assert jobs[0] is backend._jobs["job-1"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"registry_update_timeout_s": 0},
        {"registry_controller_heartbeat_ttl_s": 0},
        {"controller_heartbeat_interval_s": 0},
        {
            "registry_controller_heartbeat_ttl_s": 2,
            "controller_heartbeat_interval_s": 1,
            "registry_update_timeout_s": 1,
        },
        {"controller_terminal_retry_interval_s": 0},
        {"registry_sweep_interval_s": -1},
        {"registry_sweep_batch_limit": -1},
        {"registry_sweep_batch_limit": True},
        {"registry_sweep_batch_limit": 1.5},
    ],
)
def test_validate_job_tracking_settings_rejects_unsafe_values(kwargs) -> None:
    params = {
        "registry_update_timeout_s": 1,
        "registry_controller_heartbeat_ttl_s": 5,
        "controller_heartbeat_interval_s": 1,
        "controller_terminal_retry_interval_s": 1,
        "registry_sweep_interval_s": 0,
        "registry_sweep_batch_limit": 1,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        RayJobBackend._validate_job_tracking_settings(**params)


def test_validate_job_tracking_settings_returns_normalized_values() -> None:
    assert RayJobBackend._validate_job_tracking_settings(1, 5, 1, 2, 3, 7) == (1.0, 5.0, 1.0, 2.0, 3.0, 7)
    assert RayJobBackend._validate_job_tracking_settings(1, 5, 1, 2, 3, None) == (
        1.0,
        5.0,
        1.0,
        2.0,
        3.0,
        None,
    )


@pytest.mark.parametrize("value", [None, 1, 3])
def test_validate_max_pending_calls_accepts_none_or_positive_int(value) -> None:
    assert RayJobBackend._validate_max_pending_calls("setting", value) == value


def test_ray_job_backend_defaults_bound_actor_pending_calls() -> None:
    signature = inspect.signature(RayJobBackend)

    assert signature.parameters["registry_actor_name"].default is None
    assert signature.parameters["registry_max_pending_calls"].default == 1024
    assert signature.parameters["controller_max_pending_calls"].default == 64


def test_default_registry_actor_name_is_scope_specific_and_stable() -> None:
    scope_a = RayJobBackend._default_registry_actor_name("scope-a")
    same_scope_a = RayJobBackend._default_registry_actor_name("scope-a")
    scope_b = RayJobBackend._default_registry_actor_name("scope-b")

    assert scope_a == same_scope_a
    assert scope_a != scope_b
    assert scope_a.startswith(f"{job_backend_module.DEFAULT_REGISTRY_ACTOR_NAME}_")


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_validate_max_pending_calls_rejects_non_int_values(value) -> None:
    with pytest.raises(TypeError, match="setting must be a positive integer or None"):
        RayJobBackend._validate_max_pending_calls("setting", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1])
def test_validate_max_pending_calls_rejects_non_positive_ints(value) -> None:
    with pytest.raises(ValueError, match="setting must be a positive integer or None"):
        RayJobBackend._validate_max_pending_calls("setting", value)


def test_resolve_resources_rejects_non_mapping_resources() -> None:
    with pytest.raises(TypeError, match="resources must be a mapping"):
        RayJobBackend._resolve_resources(TinyCapability(), {"resources": "not-a-mapping"})


def test_compute_scoped_run_key_uses_default_config_and_metadata() -> None:
    capability = TinyCapability()

    explicit = RayJobBackend._compute_scoped_run_key(capability, {"config": TinyConfig(text="same")})
    defaulted = RayJobBackend._compute_scoped_run_key(capability, {})

    assert explicit != defaulted
    assert defaulted == RayJobBackend._compute_scoped_run_key(capability, {"config": TinyConfig()})


def test_compute_scoped_run_key_requires_config_factory_and_run_type() -> None:
    class NoConfigFactory:
        id = "no-config"

    with pytest.raises(ValueError, match="provide config"):
        RayJobBackend._compute_scoped_run_key(NoConfigFactory(), {})

    class NoRunType:
        id = "no-run-type"

        @staticmethod
        def _create_config():
            return TinyConfig()

    with pytest.raises(ValueError, match="define _RUN_TYPE"):
        RayJobBackend._compute_scoped_run_key(NoRunType(), {})


def test_format_controller_actor_name_uses_prefix() -> None:
    assert RayJobBackend._format_controller_actor_name("prefix", "job-1") == "prefix_job-1"


def test_ray_job_get_controller_actor_returns_none_for_missing_handles(monkeypatch) -> None:
    assert RayJob._get_controller_actor(None, "namespace") is None

    def raise_lookup_error(*_args, **_kwargs):
        raise ValueError("missing")

    monkeypatch.setattr(job_backend_module.ray, "get_actor", raise_lookup_error)

    assert RayJob._get_controller_actor("controller", "namespace") is None


def test_ray_job_fetch_record_maps_pending_call_limit_to_backpressure(monkeypatch) -> None:
    class PendingCallsLimitExceededError(Exception):
        pass

    class _RemoteGetJob:
        def remote(self, *_args):
            raise PendingCallsLimitExceededError("queue full")

    class _Registry:
        get_job = _RemoteGetJob()

    monkeypatch.setattr(job_backend_module, "PendingCallsLimitExceeded", PendingCallsLimitExceededError)

    job = RayJob(
        registry=_Registry(),
        scope="scope",
        job_id="job-1",
        created_at=job_backend_module._created_at_from_submitted_ts(0),
    )

    with pytest.raises(BackpressureError, match="Retry with exponential backoff"):
        job._fetch_record(timeout_s=1.0)


def test_ray_job_timeout_helpers_bound_control_plane_call_duration(monkeypatch) -> None:
    job = RayJob(
        registry=object(),
        scope="scope",
        job_id="job-1",
        created_at=job_backend_module._created_at_from_submitted_ts(0),
        control_plane_timeout_s=5,
    )

    assert job._remaining_s(None) is None
    assert job._control_call_timeout_s(None) == 5

    monkeypatch.setattr(job_backend_module.time, "time", lambda: 10.0)
    assert job._remaining_s(12.5) == 2.5
    assert job._remaining_s(9.0) == 0.0
    assert job._control_call_timeout_s(20.0) == 5
    assert job._control_call_timeout_s(12.0) == 2.0


def test_record_after_submit_acceptance_builds_fallback_on_registry_timeout(monkeypatch) -> None:
    def raise_timeout(*_args, **_kwargs):
        raise job_backend_module.GetTimeoutError("timeout")

    monkeypatch.setattr(job_backend_module, "_ray_get_with_backpressure", raise_timeout)

    backend = object.__new__(RayJobBackend)
    backend._idempotency_scope = "scope"
    backend._registry_update_timeout_s = 1.0
    backend._registry_namespace = "namespace"
    backend._registry = object()

    reservation = "reservation"
    registration = _record(job_id="job-1", reservation_token=reservation)
    registration["decision"] = "new"
    result_ref = _ref_payload("done")

    fallback = backend._record_after_submit_acceptance(
        registration,
        job_id="job-1",
        controller_name="controller-1",
        reservation_token=reservation,
        start_state={
            "job_id": "job-1",
            "status": "completed",
            "result_ref": result_ref,
            "error": None,
            "terminal_at_ts": 123.5,
        },
    )

    assert fallback["controller_actor_name"] == "controller-1"
    assert fallback["controller_namespace"] == "namespace"
    assert fallback["controller_token"] == reservation
    assert fallback["status"] is RegistryStatus.COMPLETED
    assert fallback["result_ref"] == result_ref
    assert fallback["completed_at_ts"] == 123.5


def test_close_failed_submission_marks_attached_or_reserved(monkeypatch) -> None:
    calls = []

    class _RemoteMethod:
        def __init__(self, name: str) -> None:
            self.name = name

        def remote(self, *args):
            calls.append((self.name, args))
            return f"{self.name}-ref"

    class _Registry:
        update_terminal = _RemoteMethod("update_terminal")
        fail_submission = _RemoteMethod("fail_submission")

    monkeypatch.setattr(job_backend_module.ray, "get", lambda ref, timeout=None: None)

    backend = object.__new__(RayJobBackend)
    backend._registry = _Registry()
    backend._idempotency_scope = "scope"
    backend._registry_update_timeout_s = 1.0
    reservation = "reservation"

    backend._close_failed_submission(
        attached=True,
        job_id="job-1",
        reservation_token=reservation,
        controller_name="controller-1",
        error="boom",
    )
    backend._close_failed_submission(
        attached=False,
        job_id="job-2",
        reservation_token=reservation,
        controller_name="controller-2",
        error="boom",
    )

    assert [name for name, _args in calls] == ["update_terminal", "fail_submission"]
    assert calls[0][1][2] is RegistryStatus.FAILED
    assert calls[1][1] == ("scope", "job-2", reservation, "boom")


def test_ray_job_ref_from_snapshot_validates_success_payload() -> None:
    job = RayJob(
        registry=object(), scope="scope", job_id="job-1", created_at=job_backend_module._created_at_from_submitted_ts(0)
    )

    ref = job._ref_from_snapshot(_record(status=RegistryStatus.COMPLETED, result_ref=_ref_payload("ok")))
    assert ref.run_uid == "run-ok"

    with pytest.raises(JobFailedError, match="completed without result payload"):
        job._ref_from_snapshot(_record(status=RegistryStatus.COMPLETED, result_ref=None))

    with pytest.raises(JobFailedError, match="invalid result payload"):
        job._ref_from_snapshot(_record(status=RegistryStatus.COMPLETED, result_ref={"bad": "payload"}))


def test_ray_job_raise_for_snapshot_maps_terminal_errors_to_public_exceptions() -> None:
    job = RayJob(
        registry=object(), scope="scope", job_id="job-1", created_at=job_backend_module._created_at_from_submitted_ts(0)
    )

    job._raise_for_snapshot(_record(status=RegistryStatus.COMPLETED, result_ref=_ref_payload()))

    with pytest.raises(JobCancelledError):
        job._raise_for_snapshot(_record(status=RegistryStatus.CANCELLED))

    with pytest.raises(JobFailedError, match="boom"):
        job._raise_for_snapshot(_record(status=RegistryStatus.FAILED, error="boom"))

    with pytest.raises(JobFailedError, match="failed"):
        job._raise_for_snapshot(_record(status=RegistryStatus.FAILED, error=None))


def test_ray_job_raise_timeout_uses_public_timeout_error() -> None:
    job = RayJob(
        registry=object(), scope="scope", job_id="job-1", created_at=job_backend_module._created_at_from_submitted_ts(0)
    )

    with pytest.raises(job_backend_module.JobTimeoutError, match="timed out"):
        job._raise_timeout(None)
