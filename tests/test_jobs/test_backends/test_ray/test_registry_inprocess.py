from __future__ import annotations

import time

import pytest

from checkmaite.core.report import ArtifactReport, InlineTextReport
from checkmaite.jobs import BackpressureError, CapabilityRunRef
from checkmaite.jobs.backends.ray.registry import (
    REGISTRY_COMPATIBILITY_VERSION,
    JobRegistry,
    RegistryCompatibilityError,
    RegistryStatus,
    _coerce_registry_status,
    _registry_configuration,
    _registry_descriptor,
    _validate_registry_descriptor,
)


def test_registry_descriptor_reports_compatibility_version_and_typed_configuration() -> None:
    configuration = _registry_configuration(registry_num_cpus=0.0)
    descriptor = JobRegistry(deployment_configuration=configuration).describe()

    assert set(descriptor) == {"compatibility_version", "configuration"}
    assert descriptor["compatibility_version"] == REGISTRY_COMPATIBILITY_VERSION
    assert descriptor["configuration"] == configuration


def test_registry_descriptor_validation_rejects_compatibility_version_mismatch() -> None:
    expected = _registry_descriptor(_registry_configuration())
    actual = _registry_descriptor(_registry_configuration())
    actual["compatibility_version"] += 1

    with pytest.raises(RegistryCompatibilityError, match="compatibility_version"):
        _validate_registry_descriptor(actual, expected)


def test_registry_descriptor_validation_rejects_configuration_mismatch() -> None:
    actual = _registry_descriptor(_registry_configuration(registry_num_cpus=0.0))
    expected = _registry_descriptor(_registry_configuration(registry_num_cpus=1.0))

    with pytest.raises(RegistryCompatibilityError, match="configuration"):
        _validate_registry_descriptor(actual, expected)


def test_registration_persists_bounded_discovery_and_resource_metadata() -> None:
    registry = JobRegistry()

    registration = registry.register_or_get(
        "scope",
        "key",
        job_name="September evaluation",
        requested_resources={"num_cpus": 4.0, "num_gpus": 1.0},
    )

    assert registration["job_name"] == "September evaluation"
    assert registration["requested_resources"] == {"num_cpus": 4.0, "num_gpus": 1.0}

    duplicate = registry.register_or_get("scope", "key", job_name="different")
    assert duplicate["job_id"] == registration["job_id"]
    assert duplicate["job_name"] == "September evaluation"


@pytest.mark.parametrize("job_name", ["", "   ", "x" * 257])
def test_registration_rejects_invalid_job_names(job_name: str) -> None:
    with pytest.raises(ValueError, match="job_name"):
        JobRegistry().register_or_get("scope", "key", job_name=job_name)


def test_scheduling_and_worker_start_are_distinct_registry_states() -> None:
    registry = JobRegistry()
    registration = registry.register_or_get(
        "scope",
        "key",
        job_name="tiny",
        requested_resources={"num_cpus": 2.0},
    )
    job_id = registration["job_id"]
    token = registration["reservation_token"]
    assert token is not None
    assert registry.attach_controller("scope", job_id, token, "controller", "namespace")

    assert registry.mark_scheduling("scope", job_id, token, "controller", "namespace", 30.0)
    scheduling = registry.get_job("scope", job_id)
    assert scheduling is not None
    assert scheduling["status"] is RegistryStatus.SCHEDULING
    assert scheduling["scheduling_started_at_ts"] is not None
    assert scheduling["scheduling_deadline_at_ts"] is not None
    assert scheduling["worker_started_at_ts"] is None

    worker_started_at = time.time()
    assert registry.mark_worker_running(
        "scope",
        job_id,
        "controller",
        token,
        {"node_id": "node-1", "pod_name": "worker-pod"},
        now_ts=worker_started_at,
    ) == (True, RegistryStatus.RUNNING)
    running = registry.get_job("scope", job_id)
    assert running is not None
    assert running["status"] is RegistryStatus.RUNNING
    assert running["worker_started_at_ts"] == worker_started_at
    assert running["worker_node_id"] == "node-1"
    assert running["worker_pod_name"] == "worker-pod"


def test_worker_start_retry_refreshes_nonempty_placement_diagnostics() -> None:
    registry = JobRegistry()
    registration = registry.register_or_get("scope", "key")
    job_id = registration["job_id"]
    token = registration["reservation_token"]
    assert token is not None
    assert registry.attach_controller("scope", job_id, token, "controller", "namespace")
    assert registry.mark_scheduling("scope", job_id, token, "controller", "namespace")
    first_worker_started_at = time.time()
    retried_worker_started_at = first_worker_started_at + 1.0

    assert registry.mark_worker_running(
        "scope",
        job_id,
        "controller",
        token,
        {
            "node_id": "node-a",
            "pod_name": "worker-pod-a",
            "kubernetes_node_name": "kubernetes-node-a",
        },
        now_ts=first_worker_started_at,
    ) == (True, RegistryStatus.RUNNING)
    assert registry.mark_worker_running(
        "scope",
        job_id,
        "controller",
        token,
        {
            "node_id": "node-b",
            "pod_name": "worker-pod-b",
            "kubernetes_node_name": "kubernetes-node-b",
        },
        now_ts=retried_worker_started_at,
    ) == (True, RegistryStatus.RUNNING)

    retried = registry.get_job("scope", job_id)
    assert retried is not None
    assert retried["worker_started_at_ts"] == retried_worker_started_at
    assert retried["worker_node_id"] == "node-b"
    assert retried["worker_pod_name"] == "worker-pod-b"
    assert retried["worker_kubernetes_node_name"] == "kubernetes-node-b"

    assert registry.mark_worker_running(
        "scope",
        job_id,
        "controller",
        token,
        {},
        now_ts=retried_worker_started_at + 1.0,
    ) == (True, RegistryStatus.RUNNING)
    reentered = registry.get_job("scope", job_id)
    assert reentered is not None
    assert reentered["worker_started_at_ts"] == retried_worker_started_at
    assert reentered["worker_node_id"] == "node-b"
    assert reentered["worker_pod_name"] == "worker-pod-b"
    assert reentered["worker_kubernetes_node_name"] == "kubernetes-node-b"


def test_registry_admission_limits_bound_nonterminal_and_scheduling_jobs() -> None:
    active_limited = JobRegistry(max_active_jobs_per_scope=1)
    first = active_limited.register_or_get("scope", "key-1")
    with pytest.raises(BackpressureError, match="active limit"):
        active_limited.register_or_get("scope", "key-2")
    active_limited.update_terminal(
        "scope",
        first["job_id"],
        RegistryStatus.FAILED,
        controller_token=first["reservation_token"],
    )
    assert active_limited.register_or_get("scope", "key-2")["decision"] == "new"

    scheduling_limited = JobRegistry(max_scheduling_jobs_per_scope=1)
    first = scheduling_limited.register_or_get("scope", "key-1")
    first_token = first["reservation_token"]
    assert first_token is not None
    assert scheduling_limited.attach_controller("scope", first["job_id"], first_token, "controller-1", "namespace")
    assert scheduling_limited.mark_scheduling("scope", first["job_id"], first_token, "controller-1", "namespace")

    with pytest.raises(BackpressureError, match="scheduling queue limit"):
        scheduling_limited.register_or_get("scope", "key-2")
    assert len(scheduling_limited.list_jobs("scope")) == 1

    # Keep the mark_scheduling check for two reservations admitted concurrently
    # before either one has entered the scheduling queue.
    racing = JobRegistry(max_scheduling_jobs_per_scope=1)
    first = racing.register_or_get("scope", "key-1")
    second = racing.register_or_get("scope", "key-2")
    first_token = first["reservation_token"]
    second_token = second["reservation_token"]
    assert first_token is not None
    assert second_token is not None
    assert racing.attach_controller("scope", first["job_id"], first_token, "controller-1", "namespace")
    assert racing.attach_controller("scope", second["job_id"], second_token, "controller-2", "namespace")
    assert racing.mark_scheduling("scope", first["job_id"], first_token, "controller-1", "namespace")
    with pytest.raises(BackpressureError, match="scheduling queue limit"):
        racing.mark_scheduling("scope", second["job_id"], second_token, "controller-2", "namespace")

    racing.fail_submission("scope", second["job_id"], second_token, "scheduling admission rejected")
    rejected = racing.get_job("scope", second["job_id"])
    assert rejected is not None
    assert rejected["status"] is RegistryStatus.FAILED
    racing.update_terminal(
        "scope",
        first["job_id"],
        RegistryStatus.FAILED,
        controller_actor_name="controller-1",
        controller_token=first_token,
    )
    assert racing.register_or_get("scope", "key-2")["decision"] == "new"


def _ref_payload(text: str = "ok") -> dict[str, object]:
    return CapabilityRunRef(
        run_uid=f"run-{text}",
        capability_id="tiny",
        store_uri=f"memory://{text}",
        outputs_uri=None,
        report=InlineTextReport(media_type="text/plain", content=text, filename="report.txt"),
    ).model_dump(mode="json")


def _registration(registry: JobRegistry, scope: str = "scope", key: str = "key"):
    registration = registry.register_or_get(scope, key)
    assert registration["reservation_token"] is not None
    return registration


def _running_job(registry: JobRegistry, scope: str = "scope", key: str = "key"):
    registration = _registration(registry, scope, key)
    job_id = registration["job_id"]
    token = registration["reservation_token"]
    controller = f"controller-{job_id}"
    assert registry.attach_controller(scope, job_id, token, controller, "namespace") is True
    assert registry.mark_running(scope, job_id, token, controller, "namespace") is True
    return registration, job_id, token, controller


def _new_started_job(registry: JobRegistry, scope: str = "scope", key: str = "key"):
    registration, job_id, token, _controller = _running_job(registry, scope, key)
    return registration, job_id, token


def test_register_or_get_dedupes_active_and_completed_jobs_but_releases_failed_jobs() -> None:
    registry = JobRegistry()

    first = registry.register_or_get("scope", "key")
    assert first["decision"] == "new"
    assert first["status"] is RegistryStatus.SUBMITTING
    assert first["reservation_token"] is not None

    duplicate = registry.register_or_get("scope", "key")
    assert duplicate["decision"] == "existing"
    assert duplicate["job_id"] == first["job_id"]

    assert registry.update_terminal(
        "scope",
        first["job_id"],
        RegistryStatus.FAILED,
        error="retryable",
        controller_token=first["reservation_token"],
    )
    retried = registry.register_or_get("scope", "key")
    assert retried["decision"] == "new"
    assert retried["job_id"] != first["job_id"]

    assert registry.update_terminal(
        "scope",
        retried["job_id"],
        RegistryStatus.COMPLETED,
        result_ref=_ref_payload("complete"),
        controller_token=retried["reservation_token"],
    )
    completed_duplicate = registry.register_or_get("scope", "key")
    assert completed_duplicate["decision"] == "existing"
    assert completed_duplicate["job_id"] == retried["job_id"]


def test_returned_records_are_copies() -> None:
    registry = JobRegistry()
    registration = registry.register_or_get("scope", "key")
    registration["status"] = RegistryStatus.FAILED

    stored = registry.get_job("scope", registration["job_id"])

    assert stored is not None
    assert stored["status"] is RegistryStatus.SUBMITTING


def test_attach_and_mark_running_require_matching_reservation_and_controller_owner() -> None:
    registry = JobRegistry()
    registration = registry.register_or_get("scope", "key")
    job_id = registration["job_id"]
    token = registration["reservation_token"]
    assert token is not None

    with pytest.raises(ValueError, match="Invalid reservation token"):
        registry.attach_controller("scope", job_id, "wrong-token", "controller", "namespace")

    assert registry.attach_controller("scope", job_id, token, "controller", "namespace") is True
    assert registry.attach_controller("scope", job_id, token, "controller", "namespace") is True

    with pytest.raises(ValueError, match="already has controller"):
        registry.attach_controller("scope", job_id, token, "other-controller", "namespace")

    with pytest.raises(ValueError, match="owned by another controller"):
        registry.mark_running("scope", job_id, token, "other-controller", "namespace")

    assert registry.mark_running("scope", job_id, token, "controller", "namespace") is True
    running = registry.get_job("scope", job_id)
    assert running is not None
    assert running["status"] is RegistryStatus.RUNNING
    assert running["reservation_token"] is None
    assert running["controller_lease_expires_at_ts"] is not None


def test_worker_start_reports_registry_first_cancellation() -> None:
    registry = JobRegistry()
    registration = registry.register_or_get("scope", "key")
    job_id = registration["job_id"]
    token = registration["reservation_token"]
    assert token is not None
    assert registry.attach_controller("scope", job_id, token, "controller", "namespace")
    assert registry.mark_scheduling("scope", job_id, token, "controller", "namespace")

    accepted, cancelling = registry.request_cancellation("scope", job_id, "controller", token)
    assert accepted is True
    assert cancelling is not None
    assert cancelling["status"] is RegistryStatus.CANCELLING
    assert registry.mark_worker_running("scope", job_id, "wrong-controller", token) == (False, None)
    assert registry.mark_worker_running("scope", job_id, "controller", token) == (
        False,
        RegistryStatus.CANCELLING,
    )


def test_cancellation_wins_later_controller_failure_after_ambiguous_response() -> None:
    registry = JobRegistry()
    registration = registry.register_or_get("scope", "key")
    job_id = registration["job_id"]
    token = registration["reservation_token"]
    assert token is not None
    assert registry.attach_controller("scope", job_id, token, "controller", "namespace")
    assert registry.mark_scheduling("scope", job_id, token, "controller", "namespace")
    accepted, cancelling = registry.request_cancellation("scope", job_id, "controller", token)
    assert accepted is True
    assert cancelling is not None
    assert cancelling["status"] is RegistryStatus.CANCELLING

    assert (
        registry.commit_controller_terminal(
            "scope",
            job_id,
            RegistryStatus.FAILED,
            error="worker rejected startup after the response was lost",
            controller_actor_name="controller",
            controller_token=token,
        )
        is RegistryStatus.CANCELLED
    )
    cancelled = registry.get_job("scope", job_id)
    assert cancelled is not None
    assert cancelled["status"] is RegistryStatus.CANCELLED
    assert cancelled["error"] is None
    assert (
        registry.update_terminal(
            "scope",
            job_id,
            RegistryStatus.FAILED,
            error="retry arrived too late",
            controller_actor_name="controller",
            controller_token=token,
        )
        is False
    )


def test_scheduling_timeout_is_ordered_against_registry_cancellation() -> None:
    registry = JobRegistry()

    def scheduling_job(key: str) -> tuple[str, str, float]:
        registration = registry.register_or_get("scope", key)
        job_id = registration["job_id"]
        token = registration["reservation_token"]
        assert token is not None
        assert registry.attach_controller("scope", job_id, token, f"controller-{key}", "namespace")
        assert registry.mark_scheduling(
            "scope",
            job_id,
            token,
            f"controller-{key}",
            "namespace",
            scheduling_timeout_s=10.0,
        )
        record = registry.get_job("scope", job_id)
        assert record is not None
        deadline = record["scheduling_deadline_at_ts"]
        assert deadline is not None
        return job_id, token, deadline

    cancelled_job_id, cancelled_token, cancelled_deadline = scheduling_job("cancelled")
    assert (
        registry.resolve_scheduling_timeout(
            "scope",
            cancelled_job_id,
            "wrong-controller",
            cancelled_token,
            now_ts=cancelled_deadline,
        )
        is None
    )
    assert (
        registry.resolve_scheduling_timeout(
            "scope",
            cancelled_job_id,
            "controller-cancelled",
            cancelled_token,
            now_ts=cancelled_deadline - 1.0,
        )
        is RegistryStatus.SCHEDULING
    )
    accepted, cancelling = registry.request_cancellation(
        "scope",
        cancelled_job_id,
        "controller-cancelled",
        cancelled_token,
    )
    assert accepted is True
    assert cancelling is not None
    assert cancelling["status"] is RegistryStatus.CANCELLING
    assert (
        registry.resolve_scheduling_timeout(
            "scope",
            cancelled_job_id,
            "controller-cancelled",
            cancelled_token,
            now_ts=cancelled_deadline,
        )
        is RegistryStatus.CANCELLING
    )

    timed_out_job_id, timed_out_token, timed_out_deadline = scheduling_job("timed-out")
    assert (
        registry.resolve_scheduling_timeout(
            "scope",
            timed_out_job_id,
            "controller-timed-out",
            timed_out_token,
            now_ts=timed_out_deadline,
        )
        is RegistryStatus.FAILED
    )
    timed_out = registry.get_job("scope", timed_out_job_id)
    assert timed_out is not None
    assert timed_out["status"] is RegistryStatus.FAILED
    assert timed_out["error"] == "worker was not scheduled within 10.000s"


def test_heartbeat_and_cancellation_follow_controller_ownership() -> None:
    registry = JobRegistry(controller_heartbeat_ttl_s=10)
    _registration, job_id, token = _new_started_job(registry)
    controller = f"controller-{job_id}"

    assert registry.heartbeat_controller("scope", job_id, "wrong-controller", token, now_ts=100.0) == (
        False,
        RegistryStatus.RUNNING,
    )
    assert registry.heartbeat_controller("scope", job_id, controller, "wrong-token", now_ts=100.0) == (
        False,
        RegistryStatus.RUNNING,
    )
    assert registry.heartbeat_controller("scope", job_id, controller, token, now_ts=100.0) == (
        True,
        RegistryStatus.RUNNING,
    )
    assert registry._job_index[("scope", job_id)]["controller_lease_expires_at_ts"] == 110.0

    accepted, unchanged = registry.request_cancellation("scope", job_id, "wrong-controller", token)
    assert accepted is False
    assert unchanged is not None
    assert unchanged["status"] is RegistryStatus.RUNNING

    accepted, cancelling = registry.request_cancellation("scope", job_id, controller, token)
    assert accepted is True
    assert cancelling is not None
    assert cancelling["status"] is RegistryStatus.CANCELLING
    assert cancelling["cancellation_requested_at_ts"] is not None

    accepted, cancelled = registry.request_cancellation("scope", "missing")
    assert accepted is False
    assert cancelled is None


def test_cancellation_before_running_is_terminal_and_retryable() -> None:
    registry = JobRegistry()
    registration = registry.register_or_get("scope", "key")

    accepted, denied = registry.request_cancellation("scope", registration["job_id"])
    assert accepted is False
    assert denied is not None
    assert denied["status"] is RegistryStatus.SUBMITTING

    accepted, cancelled = registry.request_cancellation(
        "scope",
        registration["job_id"],
        controller_token=registration["reservation_token"],
    )

    assert accepted is True
    assert cancelled is not None
    assert cancelled["status"] is RegistryStatus.CANCELLED
    retry = registry.register_or_get("scope", "key")
    assert retry["decision"] == "new"
    assert retry["job_id"] != registration["job_id"]


def test_update_terminal_validates_result_payload_and_controller_ownership() -> None:
    registry = JobRegistry(controller_retention_s=50)
    _registration, job_id, token = _new_started_job(registry)
    controller = f"controller-{job_id}"

    assert not registry.update_terminal(
        "scope",
        job_id,
        RegistryStatus.COMPLETED,
        result_ref=_ref_payload("wrong-owner"),
        controller_actor_name="wrong-controller",
        controller_token=token,
    )

    assert registry.update_terminal(
        "scope",
        job_id,
        RegistryStatus.COMPLETED,
        result_ref={"bad": "payload"},
        controller_actor_name=controller,
        controller_token=token,
    )
    failed = registry.get_job("scope", job_id)
    assert failed is not None
    assert failed["status"] is RegistryStatus.FAILED
    assert "invalid result_ref" in str(failed["error"])
    assert failed["result_ref"] is None
    assert failed["controller_retain_until_ts"] is not None

    retried = registry.register_or_get("scope", "key")
    assert retried["decision"] == "new"


@pytest.mark.parametrize(
    "report",
    [
        InlineTextReport(media_type="text/markdown", content="# Done", filename="report.md"),
        ArtifactReport(media_type="application/pdf", uri="s3://reports/report.pdf", filename="report.pdf"),
    ],
)
def test_registry_round_trips_both_report_variants(report) -> None:
    registry = JobRegistry()
    _registration, job_id, token = _new_started_job(registry)
    controller = f"controller-{job_id}"
    result_ref = CapabilityRunRef(
        run_uid="run-report",
        capability_id="tiny",
        store_uri="memory://report",
        report=report,
    ).model_dump(mode="json")

    assert registry.update_terminal(
        "scope",
        job_id,
        RegistryStatus.COMPLETED,
        result_ref=result_ref,
        controller_actor_name=controller,
        controller_token=token,
    )
    completed = registry.get_job("scope", job_id)
    assert completed is not None
    restored = CapabilityRunRef.model_validate(completed["result_ref"])
    assert type(restored.report) is type(report)
    assert restored.report == report


def test_completed_terminal_update_stores_result_and_rejects_later_state_changes() -> None:
    registry = JobRegistry()
    _registration, job_id, token = _new_started_job(registry)
    controller = f"controller-{job_id}"
    result_ref = _ref_payload("done")

    assert registry.update_terminal(
        "scope",
        job_id,
        "completed",
        result_ref=result_ref,
        controller_actor_name=controller,
        controller_token=token,
    )
    completed = registry.get_job("scope", job_id)
    assert completed is not None
    assert completed["status"] is RegistryStatus.COMPLETED
    assert completed["result_ref"] == result_ref

    assert registry.update_terminal("scope", job_id, RegistryStatus.COMPLETED, result_ref=result_ref) is True
    assert registry.update_terminal("scope", job_id, RegistryStatus.FAILED, error="too late") is False
    assert registry.update_terminal("scope", "missing", RegistryStatus.FAILED) is False


def test_list_jobs_orders_filters_limits_and_applies_lazy_expiry() -> None:
    registry = JobRegistry(reservation_ttl_s=1)
    old = registry.register_or_get("scope", "old")
    middle = registry.register_or_get("scope", "middle")
    new = registry.register_or_get("scope", "new")
    other_scope = registry.register_or_get("other", "other")

    registry._job_index[("scope", old["job_id"])]["submitted_at_ts"] = 10.0
    registry._job_index[("scope", middle["job_id"])]["submitted_at_ts"] = 20.0
    registry._job_index[("scope", new["job_id"])]["submitted_at_ts"] = 30.0
    registry._job_index[("other", other_scope["job_id"])]["submitted_at_ts"] = 40.0
    registry.update_terminal(
        "scope", old["job_id"], RegistryStatus.FAILED, error="old", controller_token=old["reservation_token"]
    )

    assert [record["job_id"] for record in registry.list_jobs("scope", limit=None)] == [
        new["job_id"],
        middle["job_id"],
        old["job_id"],
    ]
    assert [record["job_id"] for record in registry.list_jobs("scope", limit=2)] == [new["job_id"], middle["job_id"]]
    assert [record["job_id"] for record in registry.list_jobs("scope", status_filter="failed")] == [old["job_id"]]
    assert [record["job_id"] for record in registry.list_jobs("scope", before_submitted_at_ts=25.0)] == [
        middle["job_id"],
        old["job_id"],
    ]
    assert registry.list_jobs("scope", limit=-1) == []


def test_expired_submission_and_stale_running_sweeps_release_dedupe_for_retry() -> None:
    registry = JobRegistry(reservation_ttl_s=1, controller_heartbeat_ttl_s=1)
    stale_submission = registry.register_or_get("scope", "stale-submission")

    assert registry.sweep_expired_submissions(now_ts=stale_submission["reservation_expires_at_ts"] + 1) == 1
    expired = registry.get_job("scope", stale_submission["job_id"])
    assert expired is not None
    assert expired["status"] is RegistryStatus.FAILED
    assert registry.register_or_get("scope", "stale-submission")["decision"] == "new"

    _registration, running_job_id, _token = _new_started_job(registry, key="stale-running")
    running_record = registry.get_job("scope", running_job_id)
    assert running_record is not None
    registry._job_index[("scope", running_job_id)]["controller_actor_name"] = None
    assert (
        registry.sweep_stale_running_jobs(
            scope="scope",
            job_id=running_job_id,
            now_ts=running_record["controller_lease_expires_at_ts"] + 1,
        )
        == 1
    )
    failed = registry.get_job("scope", running_job_id)
    assert failed is not None
    assert failed["status"] is RegistryStatus.FAILED
    assert registry.register_or_get("scope", "stale-running")["decision"] == "new"


def test_stale_cancelling_job_becomes_cancelled_instead_of_failed() -> None:
    registry = JobRegistry(controller_heartbeat_ttl_s=1)
    _registration, job_id, token, controller = _running_job(registry, key="stale-cancelling")
    accepted, cancelling = registry.request_cancellation("scope", job_id, controller, token)
    assert accepted is True
    assert cancelling is not None
    assert cancelling["status"] is RegistryStatus.CANCELLING

    assert (
        registry.sweep_stale_running_jobs(
            scope="scope",
            job_id=job_id,
            now_ts=cancelling["controller_lease_expires_at_ts"] + 1,
        )
        == 1
    )
    cancelled = registry.get_job("scope", job_id)
    assert cancelled is not None
    assert cancelled["status"] is RegistryStatus.CANCELLED
    assert cancelled["error"] is None
    assert cancelled["controller_actor_name"] == controller
    assert cancelled["controller_retain_until_ts"] is not None


def test_retained_job_record_sweep_purges_terminal_records_without_controllers() -> None:
    registry = JobRegistry(terminal_job_retention_s=10, max_retained_terminal_jobs_per_scope=1)
    old = registry.register_or_get("scope", "old")
    new = registry.register_or_get("scope", "new")

    registry.update_terminal(
        "scope", old["job_id"], RegistryStatus.FAILED, error="old", controller_token=old["reservation_token"]
    )
    registry.update_terminal(
        "scope", new["job_id"], RegistryStatus.FAILED, error="new", controller_token=new["reservation_token"]
    )
    registry._job_index[("scope", old["job_id"])]["completed_at_ts"] = 10.0
    registry._job_index[("scope", new["job_id"])]["completed_at_ts"] = 20.0

    assert registry.sweep_retained_job_records(scope="scope", now_ts=25.0, terminal_job_retention_s=10, limit=1) == 1
    assert registry.get_job("scope", old["job_id"]) is None
    assert registry.get_job("scope", new["job_id"]) is not None

    replacement = registry.register_or_get("scope", "old")
    assert replacement["decision"] == "new"


def test_mark_controller_unavailable_fails_matching_non_terminal_job() -> None:
    registry = JobRegistry()
    _registration, job_id, token = _new_started_job(registry)

    unchanged = registry.mark_controller_unavailable("scope", job_id, "wrong-controller", token)
    assert unchanged is not None
    assert unchanged["status"] is RegistryStatus.RUNNING

    wrong_token = registry.mark_controller_unavailable("scope", job_id, f"controller-{job_id}", "wrong-token")
    assert wrong_token is not None
    assert wrong_token["status"] is RegistryStatus.RUNNING

    failed = registry.mark_controller_unavailable("scope", job_id, f"controller-{job_id}", token, "controller lost")
    assert failed is not None
    assert failed["status"] is RegistryStatus.FAILED
    assert failed["error"] == "controller lost"

    assert registry.mark_controller_unavailable("scope", "missing", None, None) is None


def test_coerce_registry_status_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Invalid registry status"):
        _coerce_registry_status("paused")


def test_copy_new_registration_requires_reservation_token() -> None:
    registry = JobRegistry()
    registration = _registration(registry)
    record = registry._job_index[("scope", registration["job_id"])]
    record["reservation_token"] = None

    with pytest.raises(RuntimeError, match="missing reservation_token"):
        registry._copy_new_registration(record)


def test_attach_controller_unknown_expired_terminal_and_invalid_state_paths() -> None:
    registry = JobRegistry(reservation_ttl_s=1)

    with pytest.raises(KeyError, match="Unknown job_id"):
        registry.attach_controller("scope", "missing", "token", "controller", "namespace")

    expired = _registration(registry, key="expired")
    assert (
        registry.attach_controller(
            "scope",
            expired["job_id"],
            expired["reservation_token"],
            "controller-expired",
            "namespace",
        )
        is True
    )
    registry._job_index[("scope", expired["job_id"])]["reservation_expires_at_ts"] = 0.0
    assert (
        registry.attach_controller(
            "scope",
            expired["job_id"],
            expired["reservation_token"],
            "controller-expired",
            "namespace",
        )
        is False
    )

    terminal = _registration(registry, key="terminal")
    assert registry.update_terminal(
        "scope",
        terminal["job_id"],
        RegistryStatus.FAILED,
        error="done",
        controller_token=terminal["reservation_token"],
    )
    assert (
        registry.attach_controller(
            "scope",
            terminal["job_id"],
            terminal["reservation_token"],
            "controller-terminal",
            "namespace",
        )
        is False
    )

    running = _registration(registry, key="running")
    record = registry._job_index[("scope", running["job_id"])]
    record["status"] = RegistryStatus.RUNNING
    with pytest.raises(ValueError, match="Cannot attach controller"):
        registry.attach_controller("scope", running["job_id"], running["reservation_token"], "controller", "namespace")


def test_mark_running_unknown_terminal_already_running_and_cancelled_before_start() -> None:
    registry = JobRegistry()

    with pytest.raises(KeyError, match="Unknown job_id"):
        registry.mark_running("scope", "missing", "token", "controller", "namespace")

    terminal = _registration(registry, key="terminal")
    assert registry.update_terminal(
        "scope",
        terminal["job_id"],
        RegistryStatus.FAILED,
        error="done",
        controller_token=terminal["reservation_token"],
    )
    assert (
        registry.mark_running("scope", terminal["job_id"], terminal["reservation_token"], "controller", "namespace")
        is False
    )

    _registration_row, job_id, token, controller = _running_job(registry, key="running")
    assert registry.mark_running("scope", job_id, token, controller, "namespace") is True

    cancelled = _registration(registry, key="cancel-before-start")
    record = registry._job_index[("scope", cancelled["job_id"])]
    record["controller_actor_name"] = "controller-cancel"
    record["controller_namespace"] = "namespace"
    record["controller_token"] = cancelled["reservation_token"]
    record["cancellation_requested_at_ts"] = 1.0
    assert (
        registry.mark_running(
            "scope",
            cancelled["job_id"],
            cancelled["reservation_token"],
            "controller-cancel",
            "namespace",
        )
        is False
    )
    assert record["status"] is RegistryStatus.CANCELLED


def test_heartbeat_controller_rejects_unknown_terminal_and_non_live_jobs() -> None:
    registry = JobRegistry()
    assert registry.heartbeat_controller("scope", "missing", "controller", "token") == (False, None)

    terminal = _registration(registry, key="terminal")
    assert registry.update_terminal(
        "scope",
        terminal["job_id"],
        RegistryStatus.FAILED,
        error="done",
        controller_token=terminal["reservation_token"],
    )
    assert registry.heartbeat_controller("scope", terminal["job_id"], "controller", "token") == (
        False,
        RegistryStatus.FAILED,
    )

    submitting = _registration(registry, key="submitting")
    record = registry._job_index[("scope", submitting["job_id"])]
    record["controller_actor_name"] = "controller"
    record["controller_token"] = submitting["reservation_token"]
    assert registry.heartbeat_controller(
        "scope",
        submitting["job_id"],
        "controller",
        submitting["reservation_token"],
    ) == (False, RegistryStatus.SUBMITTING)


def test_request_cancellation_terminal_expired_and_repeated_cancelling_paths() -> None:
    registry = JobRegistry(reservation_ttl_s=1)
    terminal = _registration(registry, key="terminal")
    registry.update_terminal(
        "scope",
        terminal["job_id"],
        RegistryStatus.FAILED,
        error="done",
        controller_token=terminal["reservation_token"],
    )
    accepted, terminal_cancel = registry.request_cancellation("scope", terminal["job_id"])
    assert accepted is False
    assert terminal_cancel is not None
    assert terminal_cancel["status"] is RegistryStatus.FAILED

    expired = _registration(registry, key="expired")
    registry._job_index[("scope", expired["job_id"])]["reservation_expires_at_ts"] = 0.0
    accepted, expired_cancel = registry.request_cancellation("scope", expired["job_id"])
    assert accepted is False
    assert expired_cancel is not None
    assert expired_cancel["status"] is RegistryStatus.FAILED

    _registration_row, job_id, token, controller = _running_job(registry, key="running")
    accepted, first = registry.request_cancellation("scope", job_id, controller, token)
    assert accepted is True
    assert first is not None
    assert first["status"] is RegistryStatus.CANCELLING
    first_requested_at = first["cancellation_requested_at_ts"]
    accepted, second = registry.request_cancellation("scope", job_id, controller, token)
    assert accepted is True
    assert second is not None
    assert second["status"] is RegistryStatus.CANCELLING
    assert second["cancellation_requested_at_ts"] >= first_requested_at


def test_fail_submission_ignores_non_matching_or_non_submitting_records() -> None:
    registry = JobRegistry()
    registry.fail_submission("scope", "missing", "token", "ignored")

    wrong_token = _registration(registry, key="wrong-token")
    registry.fail_submission("scope", wrong_token["job_id"], "wrong", "ignored")
    assert registry.get_job("scope", wrong_token["job_id"])["status"] is RegistryStatus.SUBMITTING  # type: ignore[index]

    scheduling = _registration(registry, key="scheduling")
    scheduling_token = scheduling["reservation_token"]
    assert scheduling_token is not None
    assert registry.attach_controller(
        "scope",
        scheduling["job_id"],
        scheduling_token,
        "scheduling-controller",
        "namespace",
    )
    assert registry.mark_scheduling(
        "scope",
        scheduling["job_id"],
        scheduling_token,
        "scheduling-controller",
        "namespace",
    )
    registry.fail_submission("scope", scheduling["job_id"], scheduling_token, "ignored")
    assert registry.get_job("scope", scheduling["job_id"])["status"] is RegistryStatus.SCHEDULING  # type: ignore[index]

    _registration_row, running_job_id, _token, _controller = _running_job(registry, key="running")
    registry.fail_submission("scope", running_job_id, _token, "ignored")
    assert registry.get_job("scope", running_job_id)["status"] is RegistryStatus.RUNNING  # type: ignore[index]

    terminal = _registration(registry, key="terminal")
    registry.update_terminal(
        "scope",
        terminal["job_id"],
        RegistryStatus.FAILED,
        error="done",
        controller_token=terminal["reservation_token"],
    )
    registry.fail_submission("scope", terminal["job_id"], terminal["reservation_token"], "ignored")
    assert registry.get_job("scope", terminal["job_id"])["error"] == "done"  # type: ignore[index]


def test_list_jobs_accepts_list_status_filter_and_rejects_invalid_status() -> None:
    registry = JobRegistry()
    failed = _registration(registry, key="failed")
    completed = _registration(registry, key="completed")
    _registration(registry, key="pending")
    registry.update_terminal(
        "scope", failed["job_id"], RegistryStatus.FAILED, error="failed", controller_token=failed["reservation_token"]
    )
    registry.update_terminal(
        "scope",
        completed["job_id"],
        RegistryStatus.COMPLETED,
        result_ref=_ref_payload("done"),
        controller_token=completed["reservation_token"],
    )

    listed = registry.list_jobs("scope", status_filter=["failed", RegistryStatus.COMPLETED], limit=None)
    assert {record["job_id"] for record in listed} == {failed["job_id"], completed["job_id"]}

    with pytest.raises(ValueError, match="Invalid registry status"):
        registry.list_jobs("scope", status_filter="paused")


def test_sweep_retained_job_records_count_based_policy_keeps_newest_records() -> None:
    registry = JobRegistry(terminal_job_retention_s=None, max_retained_terminal_jobs_per_scope=1)
    old = _registration(registry, key="old")
    new = _registration(registry, key="new")
    registry.update_terminal(
        "scope", old["job_id"], RegistryStatus.FAILED, error="old", controller_token=old["reservation_token"]
    )
    registry.update_terminal(
        "scope", new["job_id"], RegistryStatus.FAILED, error="new", controller_token=new["reservation_token"]
    )
    registry._job_index[("scope", old["job_id"])]["completed_at_ts"] = 10.0
    registry._job_index[("scope", new["job_id"])]["completed_at_ts"] = 20.0

    assert registry.sweep_retained_job_records(scope="scope", now_ts=30.0) == 1
    assert registry.get_job("scope", old["job_id"]) is None
    assert registry.get_job("scope", new["job_id"]) is not None
