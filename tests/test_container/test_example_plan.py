"""Keep the committed example run plan runnable against the current CheckMAITE API."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from checkmaite_container import cli, load_plan
from checkmaite_container._loading import load_capability, load_symbol

_REPOSITORY_ROOT = Path(__file__).parents[2]
_EXAMPLE_PLAN = _REPOSITORY_ROOT / "docker/example-run.yaml"


def _class_references(value: Any) -> Iterator[str]:
    """Yield every nested ``_class`` reference in constructor arguments."""
    if isinstance(value, list):
        for item in value:
            yield from _class_references(item)
    elif isinstance(value, dict):
        if "_class" in value and set(value).issubset({"_class", "args"}):
            yield value["_class"]
            yield from _class_references(value.get("args", {}))
        else:
            for item in value.values():
                yield from _class_references(item)


def test_example_plan_references_callable_objects() -> None:
    plan = load_plan(_EXAMPLE_PLAN)
    specs = plan.datasets | plan.models | plan.metrics

    references = [
        reference for spec in specs.values() for reference in [spec.class_path, *_class_references(spec.args)]
    ]
    assert references
    for reference in references:
        assert callable(load_symbol(reference, _EXAMPLE_PLAN.parent)), reference
    for task in plan.tasks:
        assert callable(load_capability(task.capability, task, plan, _EXAMPLE_PLAN.parent)), task.capability


def test_example_plan_runs_on_fixture_data(tmp_path: Path, coco_input_directory: Path) -> None:
    """Run the example unchanged on tiny repository fixtures laid out as the container input mount."""
    pytest.importorskip("onnxruntime")
    shutil.copy(_EXAMPLE_PLAN, coco_input_directory / "run.yaml")

    output = tmp_path / "output"
    status = cli.main(
        [
            "run",
            "--config",
            str(coco_input_directory / "run.yaml"),
            "--output",
            str(output),
            "--cache",
            str(tmp_path / "cache"),
            "--threads",
            "1",
            "--device",
            "cpu",
            "--batch-size",
            "1",
        ]
    )

    assert status == 0
    summary = json.loads((output / "run-results.json").read_text(encoding="utf-8"))
    (task,) = summary["tasks"]
    assert task["name"] == "baseline"
    assert task["capability_id"].endswith("MaiteEvaluation")
    assert Path(task["report_file"]).is_file()

    run = json.loads((output / "tasks/baseline/run.json").read_text(encoding="utf-8"))
    assert run["config"]["batch_size"] == 1
    assert 0.0 <= run["outputs"]["metrics"]["map50"]["scalar_values"]["map_50"] <= 1.0
    assert any((output / "analytics").rglob("*.parquet"))
