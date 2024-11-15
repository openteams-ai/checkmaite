"""metrics"""

from collections.abc import Sequence
from typing import Any, Optional, cast

import torch
from maite.protocols import object_detection as od
from torchmetrics import Metric as TorchMetric
from torchmetrics.detection import MeanAveragePrecision

__all__ = ["TorchODMetric", "map50_torch_metric_factory"]


class MissingTorchMetricKeyError(Exception):
    pass


class TorchODMetric:
    """
    A MAITE-compliant wrapper for object-detection metrics from the torchmetric package.

    The purpose of this wrapper is primarily to convert ObjectDetectionTarget MAITE classes
    into `dict[str, torch.Tensor]` that are the required input to the torchmetric package.
    See the `to_tensor_dict` method for details. It also adds a convenience feature for only
    returning a specific metric (if not specified, then all metrics will be returned).
    """

    def __init__(self, od_metric: TorchMetric, return_key: Optional[str] = None) -> None:
        self._od_metric = od_metric
        self.return_key = return_key

    @staticmethod
    def to_tensor_dict(tgt: od.ObjectDetectionTarget) -> dict[str, torch.Tensor]:
        """Convert an ObjectDetectionTarget_impl into dict expected required by torchmetrics."""
        return {
            "boxes": torch.as_tensor(tgt.boxes),
            "scores": torch.as_tensor(tgt.scores),
            "labels": torch.as_tensor(tgt.labels),
        }

    def update(
        self,
        preds: Sequence[od.ObjectDetectionTarget],
        targets: Sequence[od.ObjectDetectionTarget],
    ) -> None:
        "Add predictions and targets to metric's cache for later calculation."
        preds_tm = [self.to_tensor_dict(pred) for pred in preds]
        targets_tm = [self.to_tensor_dict(tgt) for tgt in targets]
        self._od_metric.update(preds_tm, targets_tm)

    def compute(self) -> dict[str, Any]:
        "Compute metric value(s) for currently cached predictions and targets."
        all_results = cast(dict[str, Any], self._od_metric.compute())
        if self.return_key:
            if self.return_key not in all_results:
                raise MissingTorchMetricKeyError(f"key '{self.return_key}' not in Torchmetrics results")
            return {self.return_key: all_results[self.return_key]}
        return all_results

    def reset(self) -> None:
        "Clear contents of current metric's cache of predictions and targets."
        self._od_metric.reset()


def map50_torch_metric_factory() -> od.Metric:
    "Factory for create a MAITE-compliant wrapper of the MAP-50 torchmetric."
    _tm_map50 = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.5],
        rec_thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        max_detection_thresholds=[1, 10, 100],
        class_metrics=False,
        extended_summary=False,
        average="macro",
    )
    return TorchODMetric(_tm_map50, return_key="map_50")
