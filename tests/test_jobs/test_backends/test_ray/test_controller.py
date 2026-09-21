from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import ray
from ray.actor import ActorHandle

from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend
from checkmaite.core.report import InlineTextReport
from checkmaite.jobs import BackpressureError, CapabilityRunRef, shutdown_job_backend
from checkmaite.jobs.backends.ray import controller as controller_module
from checkmaite.jobs.backends.ray.controller import (
    DEFAULT_SCHEDULING_TIMEOUT_S,
    ControllerStartupError,
    JobController,
    RayTaskResources,
    WorkerStartupUnavailableError,
    _execute_capability_ref,
    _heartbeat_registry_best_effort,
    _update_registry_terminal_best_effort,
    get_or_create_controller_actor,
)
from checkmaite.jobs.backends.ray.registry import RegistryStatus, get_or_create_registry_actor
from tests.test_jobs.fakes import EmptyTinyCapability, TinyCapability, TinyConfig, TinyDatasetCapability


def _ref_payload(text: str = "ok") -> dict[str, object]:
    return CapabilityRunRef(
        run_uid=f"run-{text}",
        capability_id="tiny",
        store_uri=f"memory://{text}",
        outputs_uri=None,
        report=InlineTextReport(media_type="text/plain", content=text, filename="report.txt"),
    ).model_dump(mode="json")


class RemoteCall:
    def __init__(self, result: object = True) -> None:
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return self.result


class InProcessController(JobController):
    def __init__(self, *args, terminal_push_result: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.terminal_push_result = terminal_push_result
        self.terminal_push_calls: list[dict[str, object]] = []
        self.retry_start_count = 0
        self.worker_cancellation_retry_start_count = 0
        self.heartbeat_loop_count = 0
        self.watch_start_count = 0

    def _push_terminal_best_effort(self) -> controller_module._RegistryCallResult:
        self.terminal_push_calls.append(self.get_state(self._controller_token, reconcile=False))
        if self.terminal_push_result:
            return controller_module._RegistryCallResult(
                controller_module._RegistryCallOutcome.ACCEPTED,
                self._status,
            )
        return controller_module._RegistryCallResult(controller_module._RegistryCallOutcome.UNAVAILABLE)

    def _start_terminal_retry_locked(self) -> None:
        if self._terminal_retry_started or self._terminal_committed:
            return
        self.retry_start_count += 1
        self._terminal_retry_started = True

    def _start_worker_cancellation_retry_locked(self) -> None:
        if self._worker_cancellation_retry_started or self._orphaned_obj_ref is None:
            return
        self.worker_cancellation_retry_start_count += 1
        self._worker_cancellation_retry_started = True

    def _heartbeat_loop(self) -> None:
        self.heartbeat_loop_count += 1

    def _start_retirement_locked(self) -> None:
        self._retirement_started = True

    def _watch_object_ref(self, obj_ref) -> None:
        self.watch_start_count += 1


def _controller(**kwargs) -> InProcessController:
    params = {
        "actor_name": "controller",
        "registry_name": "registry",
        "registry_namespace": "namespace",
        "scope": "scope",
        "job_id": "job-1",
        "registry_update_timeout_s": 0.1,
        "controller_heartbeat_interval_s": 0.1,
        "controller_terminal_retry_interval_s": 0.1,
        "controller_token": "token",
    }
    params.update(kwargs)
    return InProcessController(**params)


def test_controller_uses_a_bounded_default_scheduling_timeout() -> None:
    assert DEFAULT_SCHEDULING_TIMEOUT_S == 30 * 60.0
    assert _controller()._scheduling_timeout_s == DEFAULT_SCHEDULING_TIMEOUT_S


def test_controller_initial_state_and_terminal_status_helper() -> None:
    controller = _controller()

    assert controller.get_state("token", reconcile=False) == {
        "job_id": "job-1",
        "status": RegistryStatus.SUBMITTING,
        "result_ref": None,
        "error": None,
        "terminal_at_ts": None,
    }
    assert not controller._is_terminal(RegistryStatus.SUBMITTING)
    assert not controller._is_terminal(RegistryStatus.SCHEDULING)
    assert not controller._is_terminal(RegistryStatus.RUNNING)
    with pytest.raises(PermissionError, match="Invalid controller token"):
        controller.get_state("wrong-token", reconcile=False)
    assert controller._is_terminal(RegistryStatus.COMPLETED)
    assert controller._is_terminal(RegistryStatus.FAILED)
    assert controller._is_terminal(RegistryStatus.CANCELLED)


def test_controller_descriptor_validates_identity_token_and_configuration() -> None:
    controller = _controller(scheduling_timeout_s=20.0, controller_retention_s=30.0)

    descriptor = controller.describe("token")

    assert descriptor["compatibility_version"] == 1
    assert descriptor["actor_name"] == "controller"
    assert descriptor["scope"] == "scope"
    assert descriptor["job_id"] == "job-1"
    assert descriptor["configuration"]["scheduling_timeout_s"] == 20.0
    assert descriptor["configuration"]["controller_retention_s"] == 30.0
    with pytest.raises(PermissionError, match="Invalid controller token"):
        controller.describe("wrong-token")


def _slow_describe_controller_actor() -> Any:
    class SlowDescribeController:
        def __init__(self, **_kwargs) -> None:
            pass

        def describe(self, _token: str | None = None) -> dict[str, object]:
            time.sleep(1.0)
            return {}

    return ray.remote(max_restarts=0)(SlowDescribeController)


def _get_or_create_slow_controller(name: str, namespace: str) -> ActorHandle:
    return get_or_create_controller_actor(
        name=name,
        namespace=namespace,
        registry_name="registry",
        registry_namespace=namespace,
        scope="scope",
        job_id="job-1",
        registry_update_timeout_s=5.0,
        controller_heartbeat_interval_s=10.0,
        controller_terminal_retry_interval_s=1.0,
        controller_num_cpus=0.0,
        controller_memory=None,
        controller_resources=None,
        controller_token=f"token-{uuid4().hex}",
        startup_timeout_s=0.01,
    )


def _get_named_actor(name: str, namespace: str) -> ActorHandle | None:
    try:
        return ray.get_actor(name, namespace=namespace)
    except ValueError:
        return None


def _wait_for_named_actor_removal(name: str, namespace: str, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _get_named_actor(name, namespace) is None:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_new_controller_is_killed_when_startup_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    name = f"slow-controller-{uuid4().hex}"
    namespace = f"slow-controller-ns-{uuid4().hex}"
    slow_controller_actor = _slow_describe_controller_actor()
    monkeypatch.setattr(controller_module, "JobControllerActor", slow_controller_actor)

    try:
        with pytest.raises(ControllerStartupError, match="did not become ready"):
            _get_or_create_slow_controller(name, namespace)
        assert _wait_for_named_actor_removal(name, namespace)
    finally:
        controller = _get_named_actor(name, namespace)
        if controller is not None:
            ray.kill(controller, no_restart=True)


@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_existing_controller_is_not_killed_when_reattachment_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    name = f"slow-controller-{uuid4().hex}"
    namespace = f"slow-controller-ns-{uuid4().hex}"
    slow_controller_actor = _slow_describe_controller_actor()
    monkeypatch.setattr(controller_module, "JobControllerActor", slow_controller_actor)
    existing = slow_controller_actor.options(
        name=name,
        namespace=namespace,
        lifetime="detached",
        num_cpus=0.0,
    ).remote()

    try:
        with pytest.raises(ControllerStartupError, match="did not become ready"):
            _get_or_create_slow_controller(name, namespace)
        reattached = _get_named_actor(name, namespace)
        assert reattached is not None
        assert reattached._actor_id == existing._actor_id
    finally:
        ray.kill(existing, no_restart=True)


def test_worker_start_handshake_covers_controller_state_and_registry_outcomes(monkeypatch) -> None:
    controller = _controller()
    mark_running = RemoteCall((False, None))
    registry = type("Registry", (), {"mark_worker_running": mark_running})()
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)

    assert controller.worker_started("wrong-token", {}) is False
    assert controller.worker_started("token", {}) is False

    controller._status = RegistryStatus.FAILED
    assert controller.worker_started("token", {}) is False

    controller._status = RegistryStatus.SCHEDULING
    monkeypatch.setattr(controller, "_registry", lambda: None)
    assert controller.worker_started("token", {}) is None
    assert controller._status is RegistryStatus.SCHEDULING
    assert controller._registry_orphaned is False

    monkeypatch.setattr(controller, "_registry", lambda: registry)
    assert controller.worker_started("token", {"node_id": "node-1"}) is False
    assert controller._status is RegistryStatus.FAILED
    assert controller._registry_orphaned is True

    running = _controller()
    running._status = RegistryStatus.SCHEDULING
    monkeypatch.setattr(running, "_registry", lambda: registry)
    mark_running.result = (True, RegistryStatus.RUNNING)
    assert running.worker_started("token", {"node_id": "node-1"}) is True
    assert running.worker_started("token", {}) is True
    assert running._status is RegistryStatus.RUNNING
    assert running._worker_started.is_set()
    assert mark_running.calls[-2][0][-1] == {"node_id": "node-1"}
    assert mark_running.calls[-1][0][-1] == {}


def test_retried_worker_rechecks_registry_cancellation(monkeypatch) -> None:
    controller = _controller()
    controller._status = RegistryStatus.SCHEDULING
    mark_running = RemoteCall((True, RegistryStatus.RUNNING))
    registry = type("Registry", (), {"mark_worker_running": mark_running})()
    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)

    assert controller.worker_started("token", {"attempt": 1}) is True
    mark_running.result = (False, RegistryStatus.CANCELLING)
    assert controller.worker_started("token", {"attempt": 2}) is False

    assert len(mark_running.calls) == 2
    assert controller._status is RegistryStatus.CANCELLED
    assert controller._terminal_committed is True


def test_lost_worker_start_response_remains_retryable(monkeypatch) -> None:
    controller = _controller()
    controller._status = RegistryStatus.SCHEDULING
    mark_running = RemoteCall(object())
    registry = type("Registry", (), {"mark_worker_running": mark_running})()
    monkeypatch.setattr(controller, "_registry", lambda: registry)

    def lose_response(_value, timeout):
        raise controller_module.GetTimeoutError

    monkeypatch.setattr(controller_module.ray, "get", lose_response)

    assert controller.worker_started("token", {"attempt": 1}) is None
    assert controller._status is RegistryStatus.SCHEDULING
    assert controller._registry_orphaned is False
    assert controller._terminal_committed is False

    mark_running.result = (True, RegistryStatus.RUNNING)
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)
    assert controller.worker_started("token", {"attempt": 2}) is True
    assert controller._status is RegistryStatus.RUNNING


def test_worker_start_rpc_failure_remains_retryable(monkeypatch) -> None:
    controller = _controller()
    controller._status = RegistryStatus.RUNNING

    class FailedRemote:
        def remote(self, *_args, **_kwargs):
            raise RuntimeError("registry unavailable")

    registry = type("Registry", (), {"mark_worker_running": FailedRemote()})()
    monkeypatch.setattr(controller, "_registry", lambda: registry)

    assert controller.worker_started("token", {}) is None
    assert controller._status is RegistryStatus.RUNNING
    assert controller._registry_orphaned is False
    assert controller._terminal_committed is False
    assert controller._retirement_started is False
    assert not controller._heartbeat_stop.is_set()


def test_worker_start_rejects_existing_registry_terminal_state(monkeypatch) -> None:
    controller = _controller()
    obj_ref = object()
    controller._status = RegistryStatus.SCHEDULING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    mark_running = RemoteCall((False, RegistryStatus.CANCELLED))
    registry = type("Registry", (), {"mark_worker_running": mark_running})()
    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)
    cancelled: list[object] = []
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda ref, *, force: cancelled.append(ref),
    )

    assert controller.worker_started("token", {}) is False
    assert controller._status is RegistryStatus.FAILED
    assert controller._registry_orphaned is True
    assert controller._terminal_committed is True
    assert controller._retirement_started is True
    assert cancelled == [obj_ref]


def test_worker_start_cannot_overwrite_concurrent_heartbeat_cancellation(monkeypatch) -> None:
    controller = _controller()
    controller._status = RegistryStatus.SCHEDULING

    def cancel_during_registry_call(*_args, **_kwargs):
        controller._accept_cancellation_intent()
        return controller_module._RegistryCallResult(
            controller_module._RegistryCallOutcome.ACCEPTED,
            RegistryStatus.RUNNING,
        )

    monkeypatch.setattr(controller, "_mark_worker_running_best_effort", cancel_during_registry_call)

    assert controller.worker_started("token", {}) is False
    assert controller._status is RegistryStatus.CANCELLED
    assert not controller._worker_started.is_set()


def test_worker_start_commits_registry_first_cancellation(monkeypatch) -> None:
    controller = _controller()
    controller._status = RegistryStatus.SCHEDULING
    mark_running = RemoteCall((False, RegistryStatus.CANCELLING))
    registry = type("Registry", (), {"mark_worker_running": mark_running})()
    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)

    assert controller.worker_started("token", {}) is False
    assert controller._status is RegistryStatus.CANCELLED
    assert controller._terminal_committed is True


def test_worker_start_commits_controller_first_cancellation() -> None:
    controller = _controller()
    controller._status = RegistryStatus.CANCELLING

    assert controller.worker_started("wrong-token", {}) is False
    assert controller._status is RegistryStatus.CANCELLING
    assert not controller.terminal_push_calls

    assert controller.worker_started("token", {}) is False
    assert controller._status is RegistryStatus.CANCELLED
    assert controller._terminal_committed is True


def test_scheduling_watchdog_marks_registry_timeout_failed_and_cancels(monkeypatch) -> None:
    controller = _controller(scheduling_timeout_s=0.001)
    obj_ref = object()
    resolve_timeout = RemoteCall(RegistryStatus.FAILED)
    registry = type("Registry", (), {"resolve_scheduling_timeout": resolve_timeout})()
    cancelled: list[tuple[object, bool]] = []
    controller._status = RegistryStatus.SCHEDULING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda ref, *, force: cancelled.append((ref, force)),
    )

    controller._scheduling_timeout_loop(obj_ref)  # type: ignore[arg-type]

    assert controller._status is RegistryStatus.FAILED
    assert controller._error == "worker was not scheduled within 0.001s"
    assert cancelled == [(obj_ref, True)]
    assert resolve_timeout.calls


def test_scheduling_watchdog_retains_worker_when_initial_cancel_fails(monkeypatch) -> None:
    controller = _controller(scheduling_timeout_s=0.0)
    obj_ref = object()
    controller._status = RegistryStatus.SCHEDULING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    events: list[str] = []
    monkeypatch.setattr(
        controller,
        "_resolve_scheduling_timeout",
        lambda _obj_ref: RegistryStatus.CANCELLING,
    )

    def fail_cancel(_ref, *, force):
        events.append("cancel")
        raise RuntimeError("transient cancellation failure")

    def publish() -> None:
        events.append("publish")
        controller._terminal_committed = True
        controller._start_retirement_locked()

    monkeypatch.setattr(controller_module.ray, "cancel", fail_cancel)
    monkeypatch.setattr(controller, "_publish_terminal", publish)

    controller._scheduling_timeout_loop(obj_ref)  # type: ignore[arg-type]

    assert controller._status is RegistryStatus.CANCELLED
    assert controller._orphaned_obj_ref is obj_ref
    assert controller.worker_cancellation_retry_start_count == 1
    assert controller._retirement_started is True
    assert events == ["cancel", "publish"]


def test_scheduling_watchdog_preserves_registry_cancellation(monkeypatch) -> None:
    controller = _controller(scheduling_timeout_s=0.0)
    obj_ref = object()
    resolve_timeout = RemoteCall(RegistryStatus.CANCELLING)
    registry = type("Registry", (), {"resolve_scheduling_timeout": resolve_timeout})()
    cancelled: list[tuple[object, bool]] = []
    controller._status = RegistryStatus.SCHEDULING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda ref, *, force: cancelled.append((ref, force)),
    )

    controller._scheduling_timeout_loop(obj_ref)  # type: ignore[arg-type]

    assert controller._status is RegistryStatus.CANCELLED
    assert controller._error is None
    assert cancelled == [(obj_ref, True)]


@pytest.mark.parametrize(
    ("local_status", "expected_status"),
    [
        (RegistryStatus.SCHEDULING, RegistryStatus.FAILED),
        (RegistryStatus.CANCELLING, RegistryStatus.CANCELLED),
    ],
)
def test_scheduling_watchdog_bounds_registry_timeout_and_retires_orphan(
    monkeypatch,
    local_status: RegistryStatus,
    expected_status: RegistryStatus,
) -> None:
    controller = _controller(scheduling_timeout_s=0.0)
    obj_ref = object()
    resolution_ref = object()
    resolve_timeout = RemoteCall(resolution_ref)
    registry = type("Registry", (), {"resolve_scheduling_timeout": resolve_timeout})()
    cancelled: list[tuple[object, bool]] = []
    controller._status = local_status
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    monkeypatch.setattr(controller, "_registry", lambda: registry)

    def time_out_resolution(value, timeout):
        assert value is resolution_ref
        raise controller_module.GetTimeoutError

    monkeypatch.setattr(controller_module.ray, "get", time_out_resolution)
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda ref, *, force=False: cancelled.append((ref, force)),
    )

    controller._scheduling_timeout_loop(obj_ref)  # type: ignore[arg-type]

    assert len(resolve_timeout.calls) == 1
    assert cancelled == [(resolution_ref, False), (obj_ref, True)]
    assert controller._status is expected_status
    assert controller._registry_orphaned is True
    assert controller._terminal_committed is True
    assert controller._retirement_started is True
    assert controller._heartbeat_stop.is_set()


@pytest.mark.parametrize("failure", ["dispatch", "malformed-response"])
def test_scheduling_watchdog_handles_dispatch_and_response_errors(monkeypatch, failure: str) -> None:
    controller = _controller(scheduling_timeout_s=0.0)
    obj_ref = object()
    cancelled: list[tuple[object, bool]] = []
    controller._status = RegistryStatus.SCHEDULING
    controller._obj_ref = obj_ref  # type: ignore[assignment]

    if failure == "dispatch":

        class FailedRemote:
            def remote(self, *_args, **_kwargs):
                raise RuntimeError("dispatch failed")

        resolve_timeout = FailedRemote()
    else:
        resolve_timeout = RemoteCall("not-a-registry-status")
        monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)

    registry = type("Registry", (), {"resolve_scheduling_timeout": resolve_timeout})()
    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda ref, *, force=False: cancelled.append((ref, force)),
    )

    controller._scheduling_timeout_loop(obj_ref)  # type: ignore[arg-type]

    assert controller._status is RegistryStatus.FAILED
    assert controller._registry_orphaned is True
    assert controller._terminal_committed is True
    assert controller._retirement_started is True
    assert cancelled == [(obj_ref, True)]


def test_scheduling_watchdog_retires_controller_after_registry_record_is_lost(monkeypatch) -> None:
    controller = _controller(scheduling_timeout_s=0.0)
    obj_ref = object()
    resolve_timeout = RemoteCall(None)
    registry = type("Registry", (), {"resolve_scheduling_timeout": resolve_timeout})()
    cancelled: list[tuple[object, bool]] = []
    controller._status = RegistryStatus.SCHEDULING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda ref, *, force: cancelled.append((ref, force)),
    )

    controller._scheduling_timeout_loop(obj_ref)  # type: ignore[arg-type]

    assert len(resolve_timeout.calls) == 1
    assert cancelled == [(obj_ref, True)]
    assert controller._status is RegistryStatus.FAILED
    assert controller._error == "job registry state or controller ownership was unavailable"
    assert controller._registry_orphaned is True
    assert controller._terminal_retry_started is False


def test_registry_discovery_is_bounded_and_late_result_is_cached(monkeypatch) -> None:
    controller = _controller(registry_update_timeout_s=0.02)
    lookup_started = threading.Event()
    release_lookup = threading.Event()
    registry = object()
    calls = 0

    def slow_get_actor(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        lookup_started.set()
        release_lookup.wait(timeout=1.0)
        return registry

    monkeypatch.setattr(controller_module.ray, "get_actor", slow_get_actor)

    started_at = time.monotonic()
    assert controller._registry() is None
    assert time.monotonic() - started_at < 0.5
    assert lookup_started.is_set()
    assert calls == 1

    release_lookup.set()
    deadline = time.monotonic() + 1.0
    while controller._registry_actor is None and time.monotonic() < deadline:
        time.sleep(0.001)

    assert controller._registry() is registry
    assert calls == 1


def test_terminal_retry_commits_state_and_starts_retirement(monkeypatch) -> None:
    controller = _controller(terminal_push_result=True)
    controller._status = RegistryStatus.FAILED
    monkeypatch.setattr(controller_module.time, "sleep", lambda _delay: None)

    controller._terminal_retry_loop()

    assert controller._terminal_committed is True
    assert controller._retirement_started is True
    assert controller._heartbeat_stop.is_set()


def test_watcher_and_reconcile_commit_ready_worker_results(monkeypatch) -> None:
    ref = CapabilityRunRef.model_validate(_ref_payload("ready"))
    watched = _controller()
    monkeypatch.setattr(controller_module.ray, "get", lambda *_args, **_kwargs: ref)

    JobController._watch_object_ref(watched, object())  # type: ignore[arg-type]
    assert watched._status is RegistryStatus.COMPLETED
    assert watched._result_ref == ref.model_dump(mode="json")

    reconciled = _controller()
    obj_ref = object()
    reconciled._obj_ref = obj_ref  # type: ignore[assignment]
    monkeypatch.setattr(controller_module.ray, "wait", lambda refs, timeout: (refs, []))

    state = reconciled.reconcile()

    assert state["status"] is RegistryStatus.COMPLETED
    assert state["result_ref"] == ref.model_dump(mode="json")


def test_cancel_running_worker_records_request_and_cancels_ray_task(monkeypatch) -> None:
    controller = _controller()
    obj_ref = object()
    request_cancellation = RemoteCall((True, None))
    registry = type("Registry", (), {"request_cancellation": request_cancellation})()
    cancelled: list[tuple[object, bool]] = []
    controller._status = RegistryStatus.RUNNING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)
    monkeypatch.setattr(controller_module.ray, "wait", lambda _refs, timeout: ([], []))
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda ref, *, force: cancelled.append((ref, force)),
    )

    assert controller.cancel("token") is True
    assert controller._status is RegistryStatus.CANCELLING
    assert request_cancellation.calls
    assert cancelled == [(obj_ref, True)]


def test_cancel_retires_worker_after_definitive_registry_rejection(monkeypatch) -> None:
    controller = _controller()
    obj_ref = object()
    request_cancellation = RemoteCall((False, None))
    registry = type("Registry", (), {"request_cancellation": request_cancellation})()
    cancelled: list[tuple[object, bool]] = []
    controller._status = RegistryStatus.RUNNING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda ref, *, force: cancelled.append((ref, force)),
    )

    assert controller.cancel("token") is False
    assert controller._status is RegistryStatus.FAILED
    assert controller._registry_orphaned is True
    assert cancelled == [(obj_ref, True)]


def test_cancel_stops_worker_when_registry_already_finalized_cancellation(monkeypatch) -> None:
    controller = _controller()
    obj_ref = object()
    request_cancellation = RemoteCall((False, {"status": RegistryStatus.CANCELLED}))
    registry = type("Registry", (), {"request_cancellation": request_cancellation})()
    cancelled: list[object] = []
    controller._status = RegistryStatus.RUNNING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda ref, *, force: cancelled.append(ref),
    )

    assert controller.cancel("token") is False
    assert controller._status is RegistryStatus.CANCELLED
    assert cancelled == [obj_ref]


@pytest.mark.parametrize("proposed_status", [RegistryStatus.FAILED, RegistryStatus.COMPLETED])
def test_local_cancellation_dominates_later_terminal_state(proposed_status: RegistryStatus) -> None:
    controller = _controller()
    controller._status = RegistryStatus.CANCELLING

    state = controller._set_terminal(
        proposed_status,
        error="worker failed after cancellation",
        result_ref=_ref_payload("late-result") if proposed_status is RegistryStatus.COMPLETED else None,
    )

    assert state["status"] is RegistryStatus.CANCELLED
    assert state["error"] is None
    assert state["result_ref"] is None
    assert controller.terminal_push_calls[-1]["status"] is RegistryStatus.CANCELLED


def test_scheduling_watchdog_cannot_overwrite_local_cancellation(monkeypatch) -> None:
    controller = _controller(scheduling_timeout_s=0.0)
    obj_ref = object()
    controller._status = RegistryStatus.CANCELLING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    resolve_timeout = RemoteCall(RegistryStatus.FAILED)
    registry = type("Registry", (), {"resolve_scheduling_timeout": resolve_timeout})()
    cancelled: list[tuple[object, bool]] = []
    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda ref, *, force: cancelled.append((ref, force)),
    )

    controller._scheduling_timeout_loop(obj_ref)  # type: ignore[arg-type]

    assert controller._status is RegistryStatus.CANCELLED
    assert controller._error is None
    assert cancelled == [(obj_ref, True)]


def test_heartbeat_result_distinguishes_rejection_from_unavailability(monkeypatch) -> None:
    heartbeat = RemoteCall((False, None))
    registry = type("Registry", (), {"heartbeat_controller": heartbeat})()
    owner_credential = uuid4().hex
    monkeypatch.setattr(controller_module.ray, "get", lambda value, timeout: value)

    rejected = _heartbeat_registry_best_effort(
        registry,  # type: ignore[arg-type]
        scope="scope",
        job_id="job-1",
        controller_actor_name="controller",
        controller_token=owner_credential,
    )
    assert rejected.outcome is controller_module._RegistryCallOutcome.REJECTED

    def unavailable(_value, timeout):
        raise controller_module.GetTimeoutError

    monkeypatch.setattr(controller_module.ray, "get", unavailable)
    unknown = _heartbeat_registry_best_effort(
        registry,  # type: ignore[arg-type]
        scope="scope",
        job_id="job-1",
        controller_actor_name="controller",
        controller_token=owner_credential,
    )
    assert unknown.outcome is controller_module._RegistryCallOutcome.UNAVAILABLE


def test_heartbeat_applies_registry_first_cancellation(monkeypatch) -> None:
    controller = _controller()
    obj_ref = object()
    controller._status = RegistryStatus.RUNNING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    cancelled: list[tuple[object, bool]] = []
    monkeypatch.setattr(controller, "_registry", lambda: object())
    monkeypatch.setattr(
        controller_module,
        "_heartbeat_registry_best_effort",
        lambda *_args, **_kwargs: controller_module._RegistryCallResult(
            controller_module._RegistryCallOutcome.ACCEPTED,
            RegistryStatus.CANCELLING,
        ),
    )

    def cancel(ref, *, force):
        cancelled.append((ref, force))
        controller._heartbeat_stop.set()

    monkeypatch.setattr(controller_module.ray, "cancel", cancel)

    JobController._heartbeat_loop(controller)

    assert controller._status is RegistryStatus.CANCELLING
    assert cancelled == [(obj_ref, True)]


def test_registry_cancellation_retries_failed_worker_cancel(monkeypatch) -> None:
    controller = _controller()
    obj_ref = object()
    controller._status = RegistryStatus.RUNNING
    controller._obj_ref = obj_ref  # type: ignore[assignment]

    def fail_cancel(_ref, *, force):
        raise RuntimeError("transient cancellation failure")

    monkeypatch.setattr(controller_module.ray, "cancel", fail_cancel)

    controller._accept_cancellation_intent()

    assert controller._status is RegistryStatus.CANCELLING
    assert controller._orphaned_obj_ref is obj_ref
    assert controller.worker_cancellation_retry_start_count == 1


def test_heartbeat_rejection_cancels_and_retires_orphaned_work(monkeypatch) -> None:
    controller = _controller()
    obj_ref = object()
    controller._status = RegistryStatus.RUNNING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    registry = object()
    cancelled: list[tuple[object, bool]] = []
    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(
        controller_module,
        "_heartbeat_registry_best_effort",
        lambda *_args, **_kwargs: controller_module._RegistryCallResult(
            controller_module._RegistryCallOutcome.REJECTED
        ),
    )
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda ref, *, force: cancelled.append((ref, force)),
    )

    JobController._heartbeat_loop(controller)

    assert controller._status is RegistryStatus.FAILED
    assert controller._registry_orphaned is True
    assert controller._terminal_committed is True
    assert controller._retirement_started is True
    assert controller._heartbeat_stop.is_set()
    assert controller._orphaned_obj_ref is None
    assert cancelled == [(obj_ref, True)]


def test_orphaned_worker_is_cancelled_before_retirement_starts(monkeypatch) -> None:
    controller = _controller(controller_retention_s=0.0)
    obj_ref = object()
    controller._status = RegistryStatus.RUNNING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    events: list[str] = []
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda _ref, *, force: events.append("cancel"),
    )
    monkeypatch.setattr(controller, "_start_retirement_locked", lambda: events.append("retire"))

    assert controller._set_orphaned_live_terminal(cancel_worker=True) is True

    assert events == ["cancel", "retire"]
    assert controller._orphaned_obj_ref is None


def test_retirement_retries_orphaned_worker_cancellation_before_retention(monkeypatch) -> None:
    controller = _controller(controller_retention_s=3600.0)
    obj_ref = object()
    controller._registry_orphaned = True
    controller._orphaned_obj_ref = obj_ref  # type: ignore[assignment]
    attempts = 0
    sleeps: list[float] = []

    def cancel(_ref, *, force):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient cancellation failure")

    monkeypatch.setattr(controller_module.ray, "cancel", cancel)
    monkeypatch.setattr(controller_module.ray, "get_actor", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError))
    monkeypatch.setattr(controller_module.time, "sleep", lambda delay: sleeps.append(delay))

    controller._retirement_loop()

    assert attempts == 2
    assert sleeps[:2] == [controller._controller_terminal_retry_interval_s, 3600.0]
    assert controller._orphaned_obj_ref is None


def test_retirement_rechecks_worker_cancellation_after_retention(monkeypatch) -> None:
    controller = _controller(controller_retention_s=3600.0)
    obj_ref = object()
    controller._registry_orphaned = True
    cancelled: list[object] = []

    def sleep(delay: float) -> None:
        if delay == 3600.0:
            controller._orphaned_obj_ref = obj_ref  # type: ignore[assignment]

    monkeypatch.setattr(controller_module.time, "sleep", sleep)
    monkeypatch.setattr(
        controller_module.ray,
        "cancel",
        lambda ref, *, force: cancelled.append(ref),
    )
    monkeypatch.setattr(controller_module.ray, "get_actor", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError))

    controller._retirement_loop()

    assert cancelled == [obj_ref]
    assert controller._orphaned_obj_ref is None


def test_jittered_retry_delay_spreads_base_delay(monkeypatch) -> None:
    monkeypatch.setattr("checkmaite.jobs.backends.ray.controller.secrets.randbelow", lambda _upper: 0)
    assert JobController._jittered_retry_delay_s(8.0) == 6.0

    monkeypatch.setattr("checkmaite.jobs.backends.ray.controller.secrets.randbelow", lambda _upper: 500)
    assert JobController._jittered_retry_delay_s(8.0) == 10.0


def test_set_terminal_publishes_success_and_stops_heartbeats() -> None:
    controller = _controller()
    result_ref = _ref_payload("done")

    state = controller._set_terminal(RegistryStatus.COMPLETED, result_ref=result_ref)

    assert state["status"] is RegistryStatus.COMPLETED
    assert state["result_ref"] == result_ref
    assert state["terminal_at_ts"] is not None
    assert controller._terminal_committed is True
    assert controller._heartbeat_stop.is_set()
    assert controller.terminal_push_calls[-1]["status"] is RegistryStatus.COMPLETED


def test_set_terminal_without_completed_payload_becomes_failed() -> None:
    controller = _controller()

    state = controller._set_terminal(RegistryStatus.COMPLETED)

    assert state["status"] is RegistryStatus.FAILED
    assert state["error"] == "completed job missing result_ref"
    assert state["result_ref"] is None


def test_set_terminal_is_immutable_after_first_terminal_state() -> None:
    controller = _controller()

    first = controller._set_terminal(RegistryStatus.FAILED, error="first")
    second = controller._set_terminal(RegistryStatus.CANCELLED)

    assert first["status"] is RegistryStatus.FAILED
    assert second["status"] is RegistryStatus.FAILED
    assert second["error"] == "first"


def test_set_terminal_starts_retry_when_registry_push_fails() -> None:
    controller = _controller(terminal_push_result=False)

    controller._set_terminal(RegistryStatus.FAILED, error="registry unavailable")

    assert controller._terminal_committed is False
    assert controller.retry_start_count == 1
    assert controller._terminal_retry_started is True
    assert not controller._heartbeat_stop.is_set()


def test_cancel_before_worker_launch_marks_cancelled_and_terminal_cancel_returns_false() -> None:
    controller = _controller()

    assert controller.cancel("wrong-token") is False
    assert controller.cancel("token") is True
    assert controller.get_state("token", reconcile=False)["status"] is RegistryStatus.CANCELLED
    assert controller.cancel("token") is False


def test_reconcile_returns_current_state_when_terminal_or_no_worker_ref() -> None:
    controller = _controller()
    assert controller.reconcile()["status"] is RegistryStatus.SUBMITTING

    controller._set_terminal(RegistryStatus.FAILED, error="done")
    assert controller.reconcile()["status"] is RegistryStatus.FAILED
    assert controller.get_state("token")["status"] is RegistryStatus.FAILED


def test_start_rejects_wrong_controller_token_before_registry_or_ray_work() -> None:
    expected_token = f"expected-{uuid4().hex}"
    controller = _controller(controller_token=expected_token)

    with pytest.raises(ValueError, match="Invalid controller token"):
        controller.start(TinyCapability(), {"config": TinyConfig()}, {}, 0, f"wrong-{uuid4().hex}")


def test_start_leaves_reservation_for_submitter_cleanup_on_scheduling_backpressure(monkeypatch) -> None:
    controller = _controller()
    registry = type("Registry", (), {"mark_scheduling": RemoteCall()})()
    monkeypatch.setattr(controller, "_registry", lambda: registry)

    def reject_scheduling(*_args, **_kwargs):
        raise BackpressureError("scheduling queue limit reached")

    monkeypatch.setattr(controller_module.ray, "get", reject_scheduling)

    with pytest.raises(BackpressureError, match="scheduling queue limit"):
        controller.start(TinyCapability(), {"config": TinyConfig()}, {}, 0, "token")

    assert controller._status is RegistryStatus.SUBMITTING
    assert controller.terminal_push_calls == []


def test_start_returns_current_state_when_already_terminal_or_worker_ref_exists() -> None:
    terminal = _controller()
    terminal._set_terminal(RegistryStatus.CANCELLED)
    assert (
        terminal.start(TinyCapability(), {"config": TinyConfig()}, {}, 0, "token")["status"] is RegistryStatus.CANCELLED
    )

    running = _controller()
    running._obj_ref = object()  # type: ignore[assignment]
    assert (
        running.start(TinyCapability(), {"config": TinyConfig()}, {}, 0, "token")["status"] is RegistryStatus.SUBMITTING
    )


def test_start_reserves_retry_for_ambiguous_worker_handshake(monkeypatch) -> None:
    controller = _controller(scheduling_timeout_s=None)
    registry = type("Registry", (), {"mark_scheduling": RemoteCall(True)})()
    obj_ref = object()
    remote_options: dict[str, object] = {}

    class WorkerRemote:
        @staticmethod
        def remote(*_args, **_kwargs):
            return obj_ref

    class RuntimeContext:
        current_actor = object()

    def remote(**options):
        remote_options.update(options)
        return lambda _function: WorkerRemote()

    monkeypatch.setattr(controller, "_registry", lambda: registry)
    monkeypatch.setattr(controller_module.ray, "get", lambda value, **_kwargs: value)
    monkeypatch.setattr(controller_module.ray, "remote", remote)
    monkeypatch.setattr(controller_module.ray, "get_runtime_context", RuntimeContext)
    monkeypatch.setattr(controller, "_start_heartbeat_locked", lambda: None)
    monkeypatch.setattr(controller, "_start_watcher_locked", lambda _obj_ref: None)

    state = controller.start(TinyCapability(), {"config": TinyConfig()}, {}, 0, "token")

    assert state["status"] is RegistryStatus.SCHEDULING
    assert remote_options["max_retries"] == 1
    assert remote_options["retry_exceptions"] == [WorkerStartupUnavailableError]


def test_thread_start_helpers_are_idempotent_without_running_real_loops() -> None:
    controller = _controller()

    with controller._lock:
        controller._start_heartbeat_locked()
        controller._start_heartbeat_locked()
        controller._start_terminal_retry_locked()
        controller._start_terminal_retry_locked()
        controller._start_watcher_locked(object())
        controller._start_watcher_locked(object())

    deadline = time.time() + 2
    while controller.heartbeat_loop_count == 0 and time.time() < deadline:
        time.sleep(0.01)

    assert controller._heartbeat_started is True
    assert controller.heartbeat_loop_count == 1
    assert controller.retry_start_count == 1
    assert controller.watch_start_count == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0.0), (1, 1.0), ("2.5", 2.5)],
)
def test_ray_task_resources_normalizes_quantities(value, expected) -> None:
    assert RayTaskResources.normalize_quantity("num_cpus", value) == expected


@pytest.mark.parametrize("value", [True, object(), "not-a-number"])
def test_ray_task_resources_rejects_non_numeric_quantities(value) -> None:
    with pytest.raises(TypeError, match="non-negative numeric"):
        RayTaskResources.normalize_quantity("num_cpus", value)


def test_ray_task_resources_rejects_negative_quantities() -> None:
    with pytest.raises(ValueError, match="non-negative numeric"):
        RayTaskResources.normalize_quantity("num_cpus", -1)


def test_ray_task_resources_from_mapping_accepts_nested_and_top_level_custom_resources() -> None:
    resources = RayTaskResources.from_mapping(
        {
            "num_cpus": "2",
            "num_gpus": 0.5,
            "memory": "1024",
            "resources": {"nested": "3"},
            "top_level": 4,
        }
    )

    assert resources.as_dict() == {
        "num_cpus": 2.0,
        "num_gpus": 0.5,
        "memory": 1024.0,
        "resources": {"nested": 3.0, "top_level": 4.0},
    }
    assert RayTaskResources.from_mapping(resources) is resources


def test_ray_task_resources_from_mapping_rejects_invalid_input() -> None:
    with pytest.raises(TypeError, match="resources must be a mapping"):
        RayTaskResources.from_mapping("not-a-mapping")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"resources\['resources'\]"):
        RayTaskResources.from_mapping({"resources": "not-a-mapping"})


def test_worker_translates_non_timeout_controller_rpc_failure(monkeypatch) -> None:
    from checkmaite.jobs.backends.ray import worker as worker_module

    class RuntimeContext:
        @staticmethod
        def get_node_id() -> str:
            return "node"

    controller = type("Controller", (), {"worker_started": RemoteCall()})()
    monkeypatch.setattr(worker_module.ray, "get_runtime_context", RuntimeContext)

    def fail_rpc(*_args, **_kwargs):
        raise RuntimeError("controller unavailable")

    monkeypatch.setattr(worker_module.ray, "get", fail_rpc)

    with pytest.raises(WorkerStartupUnavailableError, match="could not verify worker startup"):
        worker_module.execute_capability_ref(
            TinyCapability(),
            {},
            controller=controller,  # type: ignore[arg-type]
            controller_token=uuid4().hex,
        )


@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_worker_task_retries_unavailable_startup_handshake(tmp_path: Path) -> None:
    @ray.remote
    class RetryStartupController:
        def __init__(self) -> None:
            self.attempts = 0

        def worker_started(self, _token, _worker_info):
            self.attempts += 1
            if self.attempts == 1:
                time.sleep(0.1)
                return None
            return True

        def attempt_count(self) -> int:
            return self.attempts

    controller = RetryStartupController.remote()
    worker = ray.remote(
        max_retries=1,
        retry_exceptions=[WorkerStartupUnavailableError],
    )(_execute_capability_ref)
    result = ray.get(
        worker.remote(
            TinyCapability(),
            {
                "config": TinyConfig(text="retried-startup"),
                "use_cache": False,
                "_analytics_store": {"backend": "parquet", "uri": str(tmp_path / "store")},
            },
            controller=controller,
            controller_token=uuid4().hex,
            startup_timeout_s=0.08,
        ),
        timeout=30,
    )

    assert result.capability_id == TinyCapability().id
    assert ray.get(controller.attempt_count.remote(), timeout=5) == 2


def test_execute_capability_ref_runs_capability_writes_store_and_returns_reference(tmp_path: Path) -> None:
    marker = tmp_path / "worker-started.txt"
    ref = _execute_capability_ref(
        TinyCapability(),
        {
            "config": TinyConfig(text="worker", start_marker_path=str(marker)),
            "use_cache": False,
            "report_threshold": 0.75,
            "_analytics_store": {"backend": "parquet", "uri": str(tmp_path / "store")},
        },
    )

    assert marker.read_text() == "started"
    assert ref.capability_id == TinyCapability().id
    assert ref.store_uri.endswith(".parquet")
    assert ref.report.model_dump() == {
        "kind": "inline_text",
        "media_type": "text/markdown",
        "content": "worker:0.75",
        "filename": "tiny-report.md",
    }


def test_execute_capability_ref_completes_with_empty_analytics(tmp_path: Path) -> None:
    ref = _execute_capability_ref(
        EmptyTinyCapability(),
        {
            "config": TinyConfig(text="no rows"),
            "use_cache": False,
            "report_threshold": 0.5,
            "_analytics_store": {"backend": "parquet", "uri": str(tmp_path / "store")},
        },
    )

    assert ref.store_uri is None
    assert ref.report.content == "no rows:0.5"


def test_execute_capability_ref_writes_provenance_to_runs_table(tmp_path: Path, fake_ic_dataset_default) -> None:
    store_path = tmp_path / "store"
    ref = _execute_capability_ref(
        TinyDatasetCapability(),
        {
            "datasets": [fake_ic_dataset_default],
            "config": TinyConfig(text="worker"),
            "use_cache": False,
            "_analytics_store": {"backend": "parquet", "uri": str(store_path)},
            "_provenance": {
                "user_id": "alice",
                "workspace_id": "workspace-a",
                "job_id": "job-1",
                "backend": "ray",
                "submitted_at": "2024-01-01T00:00:00+00:00",
                "run_event_id": "job-1",
            },
        },
    )

    result = AnalyticsStore(ParquetBackend(str(store_path))).query_sql(
        "SELECT user_id, workspace_id, job_id, backend, submitted_at, completed_at, run_event_id FROM runs"
    )

    assert ref.run_uid
    assert result.shape[0] == 1
    assert result["user_id"].to_list() == ["alice"]
    assert result["workspace_id"].to_list() == ["workspace-a"]
    assert result["job_id"].to_list() == ["job-1"]
    assert result["backend"].to_list() == ["ray"]
    assert result["submitted_at"].drop_nulls().len() == 1
    assert result["completed_at"].drop_nulls().len() == 1
    assert result["run_event_id"].to_list() == ["job-1"]


def test_execute_capability_ref_rejects_cache_in_job_submission(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="use_cache=True is not supported"):
        _execute_capability_ref(
            TinyCapability(),
            {
                "config": TinyConfig(text="worker"),
                "use_cache": True,
                "_analytics_store": {"backend": "parquet", "uri": str(tmp_path / "store")},
            },
        )


def _wait_for_controller_status(controller, controller_token: str, expected: RegistryStatus, timeout_s: float = 30.0):
    deadline = time.time() + timeout_s
    last_state = None
    while time.time() < deadline:
        last_state = ray.get(controller.get_state.remote(controller_token), timeout=5)
        if last_state["status"] is expected or last_state["status"] == expected:
            return last_state
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {expected}; last_state={last_state}")


@dataclass(frozen=True)
class RegistryContext:
    registry: object
    registry_name: str
    namespace: str
    scope: str


@dataclass(frozen=True)
class ReservedJob:
    job_id: str
    token: str
    controller_name: str


def _new_registry_context(prefix: str) -> RegistryContext:
    namespace = f"{prefix}-{uuid4().hex}"
    registry_name = f"registry-{uuid4().hex}"
    scope = f"scope-{uuid4().hex}"
    registry = get_or_create_registry_actor(
        name=registry_name,
        namespace=namespace,
        reservation_ttl_s=30.0,
        registry_num_cpus=0.0,
    )
    return RegistryContext(registry=registry, registry_name=registry_name, namespace=namespace, scope=scope)


def _reserve_job(context: RegistryContext, scoped_run_key: str) -> ReservedJob:
    registration = ray.get(context.registry.register_or_get.remote(context.scope, scoped_run_key), timeout=5)
    job_id = registration["job_id"]
    token = registration["reservation_token"]
    return ReservedJob(job_id=job_id, token=token, controller_name=f"controller-{job_id}")


def _attach_controller_record(context: RegistryContext, job: ReservedJob) -> None:
    assert ray.get(
        context.registry.attach_controller.remote(
            context.scope,
            job.job_id,
            job.token,
            job.controller_name,
            context.namespace,
        ),
        timeout=5,
    )


def _mark_running(context: RegistryContext, job: ReservedJob) -> None:
    assert ray.get(
        context.registry.mark_running.remote(
            context.scope,
            job.job_id,
            job.token,
            job.controller_name,
            context.namespace,
        ),
        timeout=5,
    )


def _new_controller_actor(context: RegistryContext, job: ReservedJob):
    return get_or_create_controller_actor(
        name=job.controller_name,
        namespace=context.namespace,
        registry_name=context.registry_name,
        registry_namespace=context.namespace,
        scope=context.scope,
        job_id=job.job_id,
        registry_update_timeout_s=5.0,
        controller_heartbeat_interval_s=0.1,
        controller_terminal_retry_interval_s=0.1,
        controller_num_cpus=0.0,
        controller_memory=None,
        controller_resources=None,
        controller_token=job.token,
    )


@ray.remote
def _rejected_worker_start() -> None:
    raise RuntimeError("job controller rejected worker startup acknowledgement")


@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_controller_registry_best_effort_helpers_update_real_registry_actor() -> None:
    shutdown_job_backend(wait=False)
    context = _new_registry_context("helper")
    job = _reserve_job(context, "key")
    _attach_controller_record(context, job)
    _mark_running(context, job)

    heartbeat = _heartbeat_registry_best_effort(
        context.registry,
        scope=context.scope,
        job_id=job.job_id,
        controller_actor_name=job.controller_name,
        controller_token=job.token,
        timeout_s=5.0,
    )
    assert heartbeat.outcome is controller_module._RegistryCallOutcome.ACCEPTED

    terminal = _update_registry_terminal_best_effort(
        context.registry,
        scope=context.scope,
        job_id=job.job_id,
        status=RegistryStatus.COMPLETED,
        result_ref=_ref_payload("helper"),
        controller_actor_name=job.controller_name,
        controller_token=job.token,
        timeout_s=5.0,
    )
    assert terminal.outcome is controller_module._RegistryCallOutcome.ACCEPTED
    assert terminal.status is RegistryStatus.COMPLETED

    stored = ray.get(context.registry.get_job.remote(context.scope, job.job_id), timeout=5)
    assert stored["status"] == RegistryStatus.COMPLETED
    assert stored["result_ref"]["run_uid"] == "run-helper"


@pytest.mark.parametrize(
    "controller_status",
    [
        pytest.param(RegistryStatus.SCHEDULING, id="registry-first"),
        pytest.param(RegistryStatus.CANCELLING, id="controller-first"),
    ],
)
@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_scheduling_cancellation_survives_worker_rejection_and_watcher(
    controller_status: RegistryStatus,
) -> None:
    context = _new_registry_context("registry-first-cancel")
    job = _reserve_job(context, "key")
    _attach_controller_record(context, job)
    assert ray.get(
        context.registry.mark_scheduling.remote(
            context.scope,
            job.job_id,
            job.token,
            job.controller_name,
            context.namespace,
        ),
        timeout=5,
    )

    controller = JobController(
        actor_name=job.controller_name,
        registry_name=context.registry_name,
        registry_namespace=context.namespace,
        scope=context.scope,
        job_id=job.job_id,
        controller_token=job.token,
    )
    controller._status = controller_status

    if controller_status is RegistryStatus.SCHEDULING:
        accepted, cancelling = ray.get(
            context.registry.request_cancellation.remote(
                context.scope,
                job.job_id,
                job.controller_name,
                job.token,
            ),
            timeout=5,
        )
        assert accepted is True
        assert cancelling["status"] == RegistryStatus.CANCELLING
    else:
        scheduling = ray.get(context.registry.get_job.remote(context.scope, job.job_id), timeout=5)
        assert scheduling["status"] == RegistryStatus.SCHEDULING

    assert controller.worker_started(job.token, {}) is False
    controller._watch_object_ref(_rejected_worker_start.remote())

    state = controller.get_state(job.token, reconcile=False)
    stored = ray.get(context.registry.get_job.remote(context.scope, job.job_id), timeout=5)
    assert state["status"] is RegistryStatus.CANCELLED
    assert state["error"] is None
    assert stored["status"] == RegistryStatus.CANCELLED
    assert stored["error"] is None


@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_scheduling_watchdog_preserves_real_registry_cancellation() -> None:
    context = _new_registry_context("watchdog-cancel")
    job = _reserve_job(context, "key")
    _attach_controller_record(context, job)
    assert ray.get(
        context.registry.mark_scheduling.remote(
            context.scope,
            job.job_id,
            job.token,
            job.controller_name,
            context.namespace,
            0.0,
        ),
        timeout=5,
    )

    obj_ref = ray.put(None)
    controller = JobController(
        actor_name=job.controller_name,
        registry_name=context.registry_name,
        registry_namespace=context.namespace,
        scope=context.scope,
        job_id=job.job_id,
        scheduling_timeout_s=0.0,
        controller_token=job.token,
    )
    controller._status = RegistryStatus.SCHEDULING
    controller._obj_ref = obj_ref  # type: ignore[assignment]
    accepted, cancelling = ray.get(
        context.registry.request_cancellation.remote(
            context.scope,
            job.job_id,
            job.controller_name,
            job.token,
        ),
        timeout=5,
    )
    assert accepted is True
    assert cancelling["status"] == RegistryStatus.CANCELLING

    controller._scheduling_timeout_loop(obj_ref)  # type: ignore[arg-type]

    state = controller.get_state(job.token, reconcile=False)
    stored = ray.get(context.registry.get_job.remote(context.scope, job.job_id), timeout=5)
    assert state["status"] is RegistryStatus.CANCELLED
    assert stored["status"] == RegistryStatus.CANCELLED


@pytest.mark.ray
@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_controller_actor_smoke_completes_and_cancels_with_real_ray(tmp_path: Path) -> None:
    shutdown_job_backend(wait=False)

    context = _new_registry_context("controller-smoke")
    store_config = {"backend": "parquet", "uri": str(tmp_path / "store")}

    try:
        job = _reserve_job(context, "complete-key")
        controller = _new_controller_actor(context, job)
        _attach_controller_record(context, job)

        started = ray.get(
            controller.start.remote(
                TinyCapability(),
                {"config": TinyConfig(text="controller-smoke"), "use_cache": False, "_analytics_store": store_config},
                {"num_cpus": 1, "num_gpus": 0.0},
                0,
                job.token,
            ),
            timeout=10,
        )
        assert started["status"] in {
            RegistryStatus.SCHEDULING,
            RegistryStatus.RUNNING,
            RegistryStatus.COMPLETED,
        }

        completed = _wait_for_controller_status(controller, job.token, RegistryStatus.COMPLETED)
        assert completed["result_ref"]["report"]["content"] == "controller-smoke:0.5"
        stored = ray.get(context.registry.get_job.remote(context.scope, job.job_id), timeout=5)
        assert stored["status"] == RegistryStatus.COMPLETED

        cancel_job = _reserve_job(context, "cancel-key")
        cancel_controller = _new_controller_actor(context, cancel_job)
        _attach_controller_record(context, cancel_job)
        cancel_started = tmp_path / "controller-cancel-started.txt"
        ray.get(
            cancel_controller.start.remote(
                TinyCapability(),
                {
                    "config": TinyConfig(text="cancel", sleep_s=3.0, start_marker_path=str(cancel_started)),
                    "use_cache": False,
                    "_analytics_store": store_config,
                },
                {"num_cpus": 1, "num_gpus": 0.0},
                0,
                cancel_job.token,
            ),
            timeout=10,
        )
        deadline = time.time() + 10
        while not cancel_started.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert cancel_started.exists()
        assert ray.get(cancel_controller.cancel.remote("wrong-token"), timeout=5) is False
        assert ray.get(cancel_controller.cancel.remote(cancel_job.token), timeout=5) is True
        cancelled = _wait_for_controller_status(cancel_controller, cancel_job.token, RegistryStatus.CANCELLED)
        assert cancelled["error"] is None
    finally:
        shutdown_job_backend(wait=False)
