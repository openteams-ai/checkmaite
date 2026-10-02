import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from threadpoolctl import threadpool_limits

_FIXTURES = Path(__file__).parents[1] / "data_for_tests"


@pytest.fixture(autouse=True)
def _restore_process_state() -> Iterator[None]:
    """Undo the environment and thread limits that a container run applies to the process."""
    environment = dict(os.environ)
    torch_threads = torch.get_num_threads()
    # ``limits=None`` changes nothing and records the current limits for restoring.
    pools = threadpool_limits(limits=None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(environment)
        torch.set_num_threads(torch_threads)
        pools.restore_original_limits()


@pytest.fixture
def plugin_file(tmp_path: Path) -> Path:
    path = tmp_path / "plugin.py"
    path.write_text(
        """
from checkmaite.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
)
from checkmaite.core.report import InlineTextReport


class FakeDataset:
    def __init__(self, dataset_id="dataset"):
        self.metadata = {"id": dataset_id}


class CorruptDataset:
    def __init__(self):
        raise ValueError("corrupt annotations")


class FakeModel:
    def __init__(self, device="missing", model_id="model"):
        self.device = device
        self.metadata = {"id": model_id}


class FakeMetric:
    def __init__(self, metric_id="metric"):
        self.metadata = {"id": metric_id}


class FakeWrapper:
    def __init__(self, child):
        self.child = child


class FakeConfig(CapabilityConfigBase):
    batch_size: int = 1


class FakeOutputs(CapabilityOutputsBase):
    device: str
    batch_size: int


class FakeRun(CapabilityRunBase[FakeConfig, FakeOutputs]):
    def collect_md_report(self, threshold):
        return InlineTextReport(
            media_type="text/markdown",
            filename="report.md",
            content=f"# Result\\n\\nThreshold: {threshold}",
        )


class FakeCapability(Capability):
    _RUN_TYPE = FakeRun

    @classmethod
    def _create_config(cls):
        return FakeConfig()

    @property
    def supports_datasets(self):
        return Number.ONE

    @property
    def supports_models(self):
        return Number.ONE

    @property
    def supports_metrics(self):
        return Number.ONE

    def _run(self, models, datasets, metrics, config, use_prediction_and_evaluation_cache):
        assert datasets and metrics
        return FakeOutputs(device=models[0].device, batch_size=config.batch_size)


class FailingCapability(FakeCapability):
    def _run(self, models, datasets, metrics, config, use_prediction_and_evaluation_cache):
        raise RuntimeError("execution failed")


class ValueFailingCapability(FakeCapability):
    def _run(self, models, datasets, metrics, config, use_prediction_and_evaluation_cache):
        raise ValueError("execution value error")
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def coco_input_directory(tmp_path: Path) -> Path:
    """Lay out tiny repository COCO and ONNX fixtures as the container input mount."""
    input_directory = tmp_path / "input"
    images = input_directory / "data/evaluation/images"
    model_directory = input_directory / "models/candidate"
    images.mkdir(parents=True)
    model_directory.mkdir(parents=True)

    coco = _FIXTURES / "coco_resized_val2017"
    for image in coco.glob("*.jpg"):
        shutil.copy(image, images)
    shutil.copy(coco / "instances_val2017_resized_6.json", input_directory / "data/evaluation/annotations.json")
    # A constant-output detector whose metadata accepts one image per batch.
    shutil.copy(_FIXTURES / "jatic_onnx_od/constant_detector.onnx", model_directory / "model.onnx")
    shutil.copy(_FIXTURES / "jatic_onnx_od/model-metadata.json", model_directory / "config.json")
    return input_directory
