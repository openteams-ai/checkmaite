"""IR-3.1-S-4: run-plan model arguments override the ONNX metadata's input height, width, and batch size."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from checkmaite_container._loading import instantiate_object
from checkmaite_container._plan import ObjectSpec

_FIXTURE = Path(__file__).parents[1] / "data_for_tests" / "jatic_onnx_od"


def _model_dir(tmp_path: Path, *, height: int, width: int, batch_size: int) -> Path:
    """Copy the 10x20, batch-1 test detector with metadata that declares a different input shape."""
    shutil.copy(_FIXTURE / "constant_detector.onnx", tmp_path / "model.onnx")
    metadata = json.loads((_FIXTURE / "model-metadata.json").read_text())
    metadata["io"]["input"].update(height=height, width=width)
    metadata["io"]["batchSize"] = batch_size
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    return tmp_path


def _onnx_model(plan_root: Path, **overrides: int) -> Any:
    spec = ObjectSpec.model_validate(
        {
            "class": "checkmaite.core.object_detection.models.OnnxODModel",
            "args": {"weights_path": "model.onnx", "config_path": "config.json", **overrides},
        }
    )
    return instantiate_object(spec, plan_root=plan_root, inject_device=False)


def test_run_plan_image_size_overrides_metadata_and_resizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("onnxruntime")
    monkeypatch.chdir(tmp_path)  # the runner resolves plan paths from the plan's directory
    plan_root = _model_dir(tmp_path, height=5, width=7, batch_size=1)
    image = np.random.default_rng(0).random((3, 30, 40), dtype=np.float32)

    # The metadata's 5x7 does not match the model's fixed 10x20 input.
    with pytest.raises(Exception, match="dimensions"):
        _onnx_model(plan_root)([image])

    detections = _onnx_model(plan_root, image_height=10, image_width=20)([image])
    assert len(detections) == 1


def test_run_plan_batch_size_overrides_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("onnxruntime")
    monkeypatch.chdir(tmp_path)
    plan_root = _model_dir(tmp_path, height=10, width=20, batch_size=-1)
    images = [np.zeros((3, 10, 20), dtype=np.float32)] * 2

    with pytest.raises(ValueError, match="Input batch has 2 images"):
        _onnx_model(plan_root, batch_size=1)(images)
