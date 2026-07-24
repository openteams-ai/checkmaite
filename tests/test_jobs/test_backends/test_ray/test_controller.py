from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
import ray

from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend
from checkmaite.core.report import InlineTextReport
from checkmaite.jobs import CapabilityRunRef, shutdown_job_backend
from checkmaite.jobs.backends.ray.controller import (
    JobController,
    RayTaskResources,
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


class InProcessController(JobController):
    def __init__(self, *args, terminal_push_result: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.terminal_push_result = terminal_push_result
        self.terminal_push_calls: list[dict[str, object]] = []
        self.retry_start_count = 0
        self.heartbeat_loop_count = 0
        self.watch_start_count = 0

    def _push_terminal_best_effort(self) -> bool:
        self.terminal_push_calls.append(self.get_state(self._controller_token, reconcile=False))
        return self.terminal_push_result

    def _start_terminal_retry_locked(self) -> None:
        if self._terminal_retry_started or self._terminal_committed:
            return
        self.retry_start_count += 1
        self._terminal_retry_started = True

    def _heartbeat_loop(self) -> None:
        self.heartbeat_loop_count += 1

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
    assert not controller._is_terminal(RegistryStatus.RUNNING)
    with pytest.raises(PermissionError, match="Invalid controller token"):
        controller.get_state("wrong-token", reconcile=False)
    assert controller._is_terminal(RegistryStatus.COMPLETED)
    assert controller._is_terminal(RegistryStatus.FAILED)
    assert controller._is_terminal(RegistryStatus.CANCELLED)


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
            "resources": {"nested": "3"},
            "top_level": 4,
        }
    )

    assert resources.as_dict() == {
        "num_cpus": 2.0,
        "num_gpus": 0.5,
        "resources": {"nested": 3.0, "top_level": 4.0},
    }
    assert RayTaskResources.from_mapping(resources) is resources


def test_ray_task_resources_from_mapping_rejects_invalid_input() -> None:
    with pytest.raises(TypeError, match="resources must be a mapping"):
        RayTaskResources.from_mapping("not-a-mapping")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"resources\['resources'\]"):
        RayTaskResources.from_mapping({"resources": "not-a-mapping"})


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


@pytest.mark.ray
@pytest.mark.usefixtures("_jobs_smoke_ray_runtime")
def test_controller_registry_best_effort_helpers_update_real_registry_actor() -> None:
    shutdown_job_backend(wait=False)
    context = _new_registry_context("helper")
    job = _reserve_job(context, "key")
    _attach_controller_record(context, job)
    _mark_running(context, job)

    assert (
        _heartbeat_registry_best_effort(
            context.registry,
            scope=context.scope,
            job_id=job.job_id,
            controller_actor_name=job.controller_name,
            controller_token=job.token,
            timeout_s=5.0,
        )
        is True
    )

    assert (
        _update_registry_terminal_best_effort(
            context.registry,
            scope=context.scope,
            job_id=job.job_id,
            status=RegistryStatus.COMPLETED,
            result_ref=_ref_payload("helper"),
            controller_actor_name=job.controller_name,
            controller_token=job.token,
            timeout_s=5.0,
        )
        is True
    )

    stored = ray.get(context.registry.get_job.remote(context.scope, job.job_id), timeout=5)
    assert stored["status"] == RegistryStatus.COMPLETED
    assert stored["result_ref"]["run_uid"] == "run-helper"


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
        assert started["status"] in {RegistryStatus.RUNNING, RegistryStatus.COMPLETED}

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
