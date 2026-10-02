import json
from pathlib import Path

import pytest
import yaml

from checkmaite import cache_path
from checkmaite_container import _runner, run_plan
from checkmaite_container._plan import RunPlanConfigurationError


def _write_plan(tmp_path: Path) -> Path:
    def reference(name: str) -> str:
        return f"file:plugin.py:{name}"

    payload = {
        "version": 1,
        "resources": {"threads": 2, "device": "cpu"},
        "datasets": {"data": {"class": reference("FakeDataset"), "args": {"dataset_id": "test-data"}}},
        "models": {"model": {"class": reference("FakeModel"), "args": {}}},
        "metrics": {"metric": {"class": reference("FakeMetric"), "args": {}}},
        "tasks": [
            {
                "name": "evaluation",
                "capability": reference("FakeCapability"),
                "dataset": "data",
                "model": "model",
                "metrics": "metric",
                "config": {"batch_size": 4},
                "report_threshold": 0.75,
            }
        ],
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_run_plan_writes_outputs_and_applies_overrides(
    tmp_path: Path,
    plugin_file: Path,
) -> None:
    assert plugin_file.is_file()
    output = tmp_path / "output"
    cache = tmp_path / "cache"

    result = run_plan(
        _write_plan(tmp_path),
        output_directory=output,
        cache_directory=cache,
        batch_size=8,
    )

    assert result.tasks[0].name == "evaluation"
    assert result.tasks[0].run_uid
    assert result.analytics_directory == str(output / "analytics")
    assert cache_path() == cache
    assert (output / "analytics").is_dir()
    assert (output / "tasks/evaluation/report.md").read_text() == "# Result\n\nThreshold: 0.75"

    run_payload = json.loads((output / "tasks/evaluation/run.json").read_text())
    assert run_payload["config"]["batch_size"] == 8
    assert run_payload["outputs"] == {"device": "cpu", "batch_size": 8}

    summary = json.loads((output / "run-results.json").read_text())
    assert summary["version"] == 1
    assert summary["tasks"][0]["capability_id"].endswith("FakeCapability")


def test_run_plan_rejects_unknown_object(tmp_path: Path, plugin_file: Path) -> None:
    plan_path = _write_plan(tmp_path)
    payload = yaml.safe_load(plan_path.read_text())
    payload["tasks"][0]["dataset"] = "missing"
    plan_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(RunPlanConfigurationError, match="unknown dataset"):
        run_plan(plan_path, output_directory=tmp_path / "output", cache_directory=tmp_path / "cache")


def test_run_plan_builds_only_referenced_objects(
    tmp_path: Path,
    plugin_file: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    plan_path = _write_plan(tmp_path)
    payload = yaml.safe_load(plan_path.read_text())
    # Building this object would fail, so the run only succeeds if it is skipped.
    payload["models"]["unused"] = {"class": "file:plugin.py:Missing"}
    plan_path.write_text(yaml.safe_dump(payload))

    run_plan(plan_path, output_directory=tmp_path / "output", cache_directory=tmp_path / "cache")

    assert "Skipping unused model(s): unused" in caplog.text


def test_run_plan_reports_configuration_errors_before_building_objects(
    tmp_path: Path,
    plugin_file: Path,
) -> None:
    plan_path = _write_plan(tmp_path)
    payload = yaml.safe_load(plan_path.read_text())
    # Building this dataset would fail, so the capability error must come first.
    payload["datasets"]["data"] = {"class": "file:plugin.py:CorruptDataset"}
    payload["tasks"][0]["capability"] = "file:plugin.py:MissingCapability"
    plan_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(RunPlanConfigurationError, match="does not export 'MissingCapability'"):
        run_plan(plan_path, output_directory=tmp_path / "output", cache_directory=tmp_path / "cache")


def test_run_plan_rejects_wrong_object_count_before_running_tasks(
    tmp_path: Path,
    plugin_file: Path,
) -> None:
    plan_path = _write_plan(tmp_path)
    payload = yaml.safe_load(plan_path.read_text())
    payload["tasks"].append({**payload["tasks"][0], "name": "second", "model": ["model", "model"]})
    plan_path.write_text(yaml.safe_dump(payload))
    output = tmp_path / "output"

    with pytest.raises(RunPlanConfigurationError, match="task 'second'.*exactly 1 model"):
        run_plan(plan_path, output_directory=output, cache_directory=tmp_path / "cache")
    assert not (output / "tasks").exists()


def test_run_plan_does_not_report_host_faults_as_configuration_errors(
    tmp_path: Path,
    plugin_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def malformed_cgroup() -> int:
        raise ValueError("invalid literal for int() with base 10: 'bad'")

    monkeypatch.setattr(_runner, "_cgroup_cpu_count", malformed_cgroup)

    with pytest.raises(ValueError, match="invalid literal") as exc_info:
        run_plan(_write_plan(tmp_path), output_directory=tmp_path / "output", cache_directory=tmp_path / "cache")
    assert not isinstance(exc_info.value, RunPlanConfigurationError)


def test_resolve_resources_caps_threads_by_host_capacity() -> None:
    plan = _runner.RunPlan.model_validate(
        {"version": 1, "resources": {"threads": 20, "device": "cpu"}, "tasks": [{"name": "task", "capability": "x.y"}]}
    )
    host = _runner.HostResources(available_cpus=6, cuda_device_count=0)

    resources = _runner.resolve_resources(plan, host)

    assert resources.available_cpus == 6
    assert resources.threads == 6
    assert resources.device == "cpu"


def test_resolve_resources_auto_selects_cuda() -> None:
    plan = _runner.RunPlan.model_validate({"version": 1, "tasks": [{"name": "task", "capability": "x.y"}]})
    host = _runner.HostResources(available_cpus=2, cuda_device_count=1)

    assert _runner.resolve_resources(plan, host).device == "cuda:0"


def test_resolve_resources_rejects_invalid_device_override() -> None:
    plan = _runner.RunPlan.model_validate({"version": 1, "tasks": [{"name": "task", "capability": "x.y"}]})
    host = _runner.HostResources(available_cpus=2, cuda_device_count=0)

    with pytest.raises(ValueError, match="device must be"):
        _runner.resolve_resources(plan, host, device="gpu")


def test_resolve_resources_normalizes_and_checks_cuda_device() -> None:
    plan = _runner.RunPlan.model_validate(
        {"version": 1, "resources": {"device": "cuda"}, "tasks": [{"name": "task", "capability": "x.y"}]}
    )
    host = _runner.HostResources(available_cpus=2, cuda_device_count=1)

    assert _runner.resolve_resources(plan, host).device == "cuda:0"
    with pytest.raises(RunPlanConfigurationError, match=r"only 1 device\(s\) are available"):
        _runner.resolve_resources(plan, host, device="cuda:1")


def test_resolve_resources_rejects_unavailable_cuda() -> None:
    plan = _runner.RunPlan.model_validate(
        {"version": 1, "resources": {"device": "cuda:1"}, "tasks": [{"name": "task", "capability": "x.y"}]}
    )
    host = _runner.HostResources(available_cpus=2, cuda_device_count=0)

    with pytest.raises(RunPlanConfigurationError, match="CUDA is not available"):
        _runner.resolve_resources(plan, host)
