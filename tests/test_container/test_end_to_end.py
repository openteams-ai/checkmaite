"""Run realistic plans through the container CLI on tiny repository fixtures."""

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from checkmaite_container import cli

_DATASETS = {
    "evaluation": {
        "class": "checkmaite.core.object_detection.dataset_loaders.load_coco_detection_dataset",
        "args": {"root": "data/evaluation/images", "ann_file": "data/evaluation/annotations.json"},
    }
}
_ONNX_MODEL = {
    "class": "checkmaite.core.object_detection.models.OnnxODModel",
    "args": {"weights_path": "models/candidate/model.onnx", "config_path": "models/candidate/config.json"},
}

_DETECTOR_PLUGIN = """
import numpy as np
from modelmaite.object_detection import DetectionTarget


class FixedBoxDetector:
    \"\"\"Predict one fixed box per image, as a trained detector would predict one per object.\"\"\"

    def __init__(self, label, **kwargs):
        self.label = label
        # The runner supplies the resolved device through **kwargs.
        self.metadata = {"id": f"fixed-box-{kwargs['device']}", "index2label": {label: "dog"}}

    def __call__(self, batch):
        return [
            DetectionTarget(
                boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
                labels=np.array([self.label]),
                scores=np.array([0.9]),
            )
            for _ in batch
        ]
"""


def _map_metric(metric_id: str, return_key: str) -> dict[str, Any]:
    return {
        "class": "checkmaite.core.object_detection.metrics.TorchODMetric",
        "args": {
            "od_metric": {
                "_class": "torchmetrics.detection.MeanAveragePrecision",
                "args": {"box_format": "xyxy", "iou_type": "bbox"},
            },
            "return_key": return_key,
            "metric_id": metric_id,
        },
    }


def _run(tmp_path: Path, input_directory: Path, plan: dict[str, Any]) -> tuple[int, Path]:
    plan_path = input_directory / "run.yaml"
    plan_path.write_text(yaml.safe_dump({"version": 1, "resources": {"device": "cpu"}, **plan}), encoding="utf-8")
    output = tmp_path / "output"
    status = cli.main(
        ["run", "--config", str(plan_path), "--output", str(output), "--cache", str(tmp_path / "cache")],
    )
    return status, output


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_mounted_plugin_model_runs_with_builtin_objects(tmp_path: Path, coco_input_directory: Path) -> None:
    plugins = coco_input_directory / "plugins"
    plugins.mkdir()
    (plugins / "detector.py").write_text(_DETECTOR_PLUGIN, encoding="utf-8")

    status, output = _run(
        tmp_path,
        coco_input_directory,
        {
            "datasets": _DATASETS,
            "models": {"candidate": {"class": "file:plugins/detector.py:FixedBoxDetector", "args": {"label": 18}}},
            "metrics": {"map50": _map_metric("map50", "map_50")},
            "tasks": [
                {
                    "name": "baseline",
                    # Inferred from the built-in dataset even though the model is a plugin.
                    "capability": "builtin:MaiteEvaluation",
                    "dataset": "evaluation",
                    "model": "candidate",
                    "metrics": "map50",
                    "config": {"batch_size": 2},
                }
            ],
        },
    )

    assert status == 0
    run = _read_json(output / "tasks/baseline/run.json")
    assert run["capability_id"].endswith("object_detection.maite_evaluation_capability.MaiteEvaluation")
    assert run["model_metadata"][0]["id"] == "fixed-box-cpu"
    assert 0.0 <= run["outputs"]["metrics"]["map50"]["scalar_values"]["map_50"] <= 1.0


def test_multi_task_plan_shares_objects_across_capabilities(tmp_path: Path, coco_input_directory: Path) -> None:
    pytest.importorskip("onnxruntime")

    status, output = _run(
        tmp_path,
        coco_input_directory,
        {
            "datasets": _DATASETS,
            "models": {"candidate": _ONNX_MODEL},
            "metrics": {"map50": _map_metric("map50", "map_50"), "map": _map_metric("map", "map")},
            "tasks": [
                {
                    "name": "evaluation",
                    "capability": "builtin:MaiteEvaluation",
                    "dataset": "evaluation",
                    "model": "candidate",
                    "metrics": ["map50", "map"],
                    "config": {"batch_size": 1},
                },
                {"name": "cleaning", "capability": "builtin:DataevalCleaning", "dataset": "evaluation"},
            ],
        },
    )

    assert status == 0
    summary = _read_json(output / "run-results.json")
    assert [task["name"] for task in summary["tasks"]] == ["evaluation", "cleaning"]

    evaluation = _read_json(output / "tasks/evaluation/run.json")
    cleaning = _read_json(output / "tasks/cleaning/run.json")
    assert set(evaluation["outputs"]["metrics"]) == {"map50", "map"}
    assert cleaning["capability_id"].endswith("DataevalCleaning")
    assert cleaning["model_metadata"] == []
    assert cleaning["dataset_metadata"] == evaluation["dataset_metadata"]

    # The runs table has one row per dataset, model, and metric in each run.
    runs = pl.read_parquet(output / "analytics/runs/*.parquet")
    assert set(runs["run_uid"]) == {task["run_uid"] for task in summary["tasks"]}
