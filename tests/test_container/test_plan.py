from pathlib import Path

import pytest
from pydantic import ValidationError

from checkmaite_container import RunPlan, load_plan


def test_run_plan_applies_resource_and_object_defaults() -> None:
    plan = RunPlan.model_validate(
        {
            "version": 1,
            "datasets": {"data": {"class": "package.Dataset"}},
            "tasks": [{"name": "task", "capability": "package.Capability"}],
        }
    )

    assert plan.resources.threads == "auto"
    assert plan.resources.device == "auto"
    assert plan.datasets["data"].class_path == "package.Dataset"
    assert plan.datasets["data"].args == {}


@pytest.mark.parametrize("device", ["gpu", "mps", "cuda-ish"])
def test_run_plan_rejects_unknown_device(device: str) -> None:
    with pytest.raises(ValidationError, match="device must be"):
        RunPlan.model_validate(
            {
                "version": 1,
                "resources": {"device": device},
                "tasks": [{"name": "task", "capability": "package.Capability"}],
            }
        )


def test_run_plan_rejects_duplicate_task_names() -> None:
    with pytest.raises(ValidationError, match="task names must be unique"):
        RunPlan.model_validate(
            {
                "version": 1,
                "tasks": [
                    {"name": "same", "capability": "package.First"},
                    {"name": "same", "capability": "package.Second"},
                ],
            }
        )


def test_load_plan_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_plan(path)
