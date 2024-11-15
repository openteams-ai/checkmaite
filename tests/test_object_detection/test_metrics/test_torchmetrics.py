from jatic_ri.object_detection.metrics import map50_torch_metric_factory, TorchODMetric
from typing import Any, Sequence
from dataclasses import dataclass
from maite.protocols import object_detection as od
import torch
import pytest
from maite.protocols import ArrayLike
import numpy as np


@pytest.fixture
def dummy_data():
    """Dummy predictions and targets that satisfy MAITE ObjectDetectionTarget interface."""

    @dataclass
    class DetectionTarget:
        boxes: ArrayLike
        labels: ArrayLike
        scores: ArrayLike

    pred = DetectionTarget(
        boxes=np.array([[10, 10, 50, 50]]),
        scores=np.array([0.9]),
        labels=np.array([1]),
    )
    target = DetectionTarget(
        boxes=np.array([[10, 10, 50, 50]]),
        scores=np.array([1.0]),
        labels=np.array([1]),
    )
    
    return [pred], [target]


class FakeTorchMetric:
    """A simplified implementation of the torchmetric Metric ABC."""

    def __init__(self) -> None:
        self._fake_data: dict[str, Any] = {}

    def compute(self) -> dict[str, Any]:
        return self._fake_data 

    def update(self, preds: Sequence[dict[str, torch.Tensor]], targets: Sequence[dict[str, torch.Tensor]]) -> None:        
        self._fake_data.update({"preds": preds, "targets":targets})

    def reset(self) -> None:
        self._fake_data.clear()


def test_torch_metric_wrapper(dummy_data):
    """Check that wrapper gives equivalent results as original metric object"""
    
    preds, targets = dummy_data
    fake_torch_metric = FakeTorchMetric()
    metric_wrapper = TorchODMetric(fake_torch_metric, return_key=None)

    # wrapper should always return an identical result as original metric object
    initial_result = metric_wrapper.compute()
    assert  initial_result == fake_torch_metric.compute()

    # wrapper should always convert input to type dict[str, torch.Tensor]
    # as this is required input format for torchmetric
    metric_wrapper.update(preds, targets)
    for result_valz in metric_wrapper._od_metric._fake_data.values():
        for detections in result_valz:
            for torch_valz in detections.values():
                assert isinstance(torch_valz, torch.Tensor)

    # resetting the wrapper should restore original result
    metric_wrapper.reset()
    assert metric_wrapper.compute() == initial_result


def test_map50_torch_metric_update(dummy_data):
    """Test update method adds data correctly."""
    
    preds, targets = dummy_data
    map50_torch_metric = map50_torch_metric_factory()
    map50_torch_metric.update(preds, targets)
    # No assertions; we just want to make sure no errors occur here.

def test_map50_torch_metric_compute(dummy_data):
    """Test compute method returns the expected mAP@0.5 value."""

    preds, targets = dummy_data
    map50_torch_metric = map50_torch_metric_factory()
    map50_torch_metric.update(preds, targets)
    result = map50_torch_metric.compute()
    assert "map_50" in result, "Expected 'map_50' key in result"
    assert 0.0 <= result["map_50"] <= 1.0, "Expected mAP@0.5 to be between 0 and 1"

def test_map50_torch_metric_reset(dummy_data):
    """Test reset method clears cached data."""

    preds, targets = dummy_data
    map50_torch_metric = map50_torch_metric_factory()
    map50_torch_metric.update(preds, targets)
    map50_torch_metric.reset()
    # calling compute before update raises warning that this could be an error,
    # but this is deliberate in this test and so we suppress
    warn_msg = ("The ``compute`` method of metric MeanAveragePrecision was called "
    "before the ``update`` method which may lead to errors, as metric states have "
    "not yet been updated.")
    with pytest.warns(UserWarning, match=warn_msg):
        result = map50_torch_metric.compute()
    
    # no data returns -1
    assert result["map_50"] == -1, "Expected 'map_50' to be -1 after reset"
