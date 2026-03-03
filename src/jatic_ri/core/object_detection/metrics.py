from collections.abc import Sequence
from typing import Any, cast

import torch
from maite.protocols import MetricMetadata
from maite.protocols import object_detection as od
from torchmetrics import Metric as TorchMetric
from torchmetrics.detection import MeanAveragePrecision

from jatic_ri.core._utils import id_hash

__all__ = [
    "TorchODMetric",
    "TorchODMultiClassMap50",
    "map50_torch_metric_factory",
    "multiclass_map50_torch_metric_factory",
]


class MissingTorchMetricKeyError(Exception):
    pass


class TorchODMetric:
    """A MAITE-compliant wrapper for object-detection metrics from the torchmetric package.

    The purpose of this wrapper is primarily to convert ObjectDetectionTarget MAITE classes
    into `dict[str, torch.Tensor]` that are the required input to the torchmetric package.
    See the `to_tensor_dict` method for details. It also adds a convenience feature for only
    returning a specific metric (if not specified, then all metrics will be returned).

    Parameters
    ----------
    od_metric : TorchMetric
        The torchmetric object detection metric.
    return_key : str | None, optional
        If specified, only this key from the metric results will be returned.
        Default is None, which returns all metrics.
    metric_id : str
        Identifier for the metric.

    Attributes
    ----------
    metadata : MetricMetadata
        Metadata for the metric.
    """

    def __init__(self, od_metric: TorchMetric, return_key: str | None = None, *, metric_id: str) -> None:
        """Initialize the Torch object detection metric wrapper.

        Parameters
        ----------
        od_metric : TorchMetric
            The torchmetrics object detection metric to wrap.
        return_key : str, optional
            The specific metric key to return. Default is None.
        metric_id : str
            Identifier for metric.
        """
        self._od_metric = od_metric
        self.return_key: str | None = return_key
        self.metadata = MetricMetadata(id=metric_id)

    @staticmethod
    def to_tensor_dict(tgt: od.ObjectDetectionTarget) -> dict[str, torch.Tensor]:
        """Convert an ObjectDetectionTarget into dict expected by torchmetrics.

        Parameters
        ----------
        tgt : od.ObjectDetectionTarget
            The object detection target to convert.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary with "boxes", "scores", and "labels" keys,
            and corresponding torch tensors as values.
        """
        return {
            "boxes": torch.as_tensor(tgt.boxes),
            "scores": torch.as_tensor(tgt.scores),
            "labels": torch.as_tensor(tgt.labels),
        }

    def update(
        self,
        preds: Sequence[od.ObjectDetectionTarget],
        targets: Sequence[od.ObjectDetectionTarget],
        metadatas: Sequence[od.DatumMetadataType] = [],  # noqa: ARG002
    ) -> None:
        """Add predictions and targets to metric's cache for later calculation.

        Parameters
        ----------
        preds : Sequence[od.ObjectDetectionTarget]
            A sequence of predicted object detection targets.
        targets : Sequence[od.ObjectDetectionTarget]
            A sequence of ground truth object detection targets.
        """
        preds_tm = [self.to_tensor_dict(pred) for pred in preds]
        targets_tm = [self.to_tensor_dict(tgt) for tgt in targets]
        self._od_metric.update(preds_tm, targets_tm)

    def compute(self) -> dict[str, Any]:
        """Compute metric value(s) for currently cached predictions and targets.

        Returns
        -------
        dict[str, Any]
            A dictionary of metric results. If `return_key` was specified during
            initialization, only that metric is returned. Otherwise, all computed
            metrics are returned.

        Raises
        ------
        MissingTorchMetricKeyError
            If `return_key` was specified but is not found in the torchmetrics
            results.
        """
        all_results = cast(dict[str, Any], self._od_metric.compute())
        if self.return_key:
            if self.return_key not in all_results:
                raise MissingTorchMetricKeyError(f"key '{self.return_key}' not in Torchmetrics results")
            return {self.return_key: all_results[self.return_key]}
        return all_results

    def reset(self) -> None:
        """Clear contents of current metric's cache of predictions and targets."""
        self._od_metric.reset()


class TorchODMultiClassMap50(TorchODMetric):
    """A MAITE-compliant wrapper for the Torchmetrics MeanAveragePrecision metric with multi-class scores.

    In order to return values in a format compliant with the RI standards, each
    element returned by compute must be safely convertable to a float.
    Therefore, the compute() method is overridden.

    See the RI conventional for more details:
    https://jatic.pages.jatic.net/reference-implementation/reference-implementation/reference/conventions.html

    Also note that the Metric object does not have access to class names (i.e.
    index2label), so returning class numbers for keys is the best that can be
    done.

    Parameters
    ----------
    _class_map50 : MeanAveragePrecision
        The Torchmetrics MeanAveragePrecision metric instance.
    return_key : str | None, optional
        The primary key to return from the computed metrics.
        Default is "map_50_classwise".
    metric_id : str
        Identifier for the metric.
    """

    def __init__(
        self, _class_map50: MeanAveragePrecision, return_key: str | None = "map_50_classwise", *, metric_id: str
    ) -> None:
        super().__init__(od_metric=_class_map50, return_key=return_key, metric_id=metric_id)

    def compute(self) -> dict[str, Any]:
        """Compute metric value(s) for currently cached predictions and targets.

        MeanAveragePrecision.compute() returns a key 'map_per_class' which is a
        list of mAP per class and another key 'classes' which is a list of
        corresponding class IDs. The dict comprehension in the return statement
        restructures these into an RI-compliant format of dict[str, num].

        Returns
        -------
        dict[str, Any]
            A dictionary of metric results. This includes the primary `return_key`
            (e.g., "map_50_classwise"), per-class mAP scores keyed by class ID
            (as a string), and a "per_class_flag" set to 1.
        """
        all_results = cast(dict[str, Any], self._od_metric.compute())
        if not self.return_key:
            return all_results
        return (
            {self.return_key: all_results["map_50"]}
            | {
                # Added to the return key are entries with the class id as key and the mAP for that class as value
                # Class values are Tensor((1,), dtype=torch.int32) so must cast as int then str to get desired output
                str(int(all_results["classes"][i])): all_results["map_per_class"][i]
                for i in range(len(all_results["map_per_class"]))
                if all_results["map_per_class"][i] >= 0.0  # Metric returns -1.0 for classes not present in the dataset
            }
            | {"per_class_flag": 1}
        )  # Added to indicate that the per-class mAP is present in the output


def map50_torch_metric_factory() -> od.Metric:
    """Create a MAITE-compliant wrapper of the MAP-50 torchmetric.

    Uses reasonable defaults.

    Returns
    -------
    od.Metric
        A MAITE-compliant object detection metric.
    """
    map50_params = {
        "box_format": "xyxy",
        "iou_type": "bbox",
        "iou_thresholds": [0.5],
        "rec_thresholds": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "max_detection_thresholds": [1, 10, 100],
        "class_metrics": False,
        "extended_summary": False,
        "average": "macro",
    }
    _tm_map50 = MeanAveragePrecision(**map50_params)
    return TorchODMetric(_tm_map50, return_key="map_50", metric_id=f"mAP_50_{id_hash(**map50_params)}")


def multiclass_map50_torch_metric_factory() -> od.Metric:
    """Create a MAITE-compliant wrapper of the multi-class MAP-50 torchmetric.

    Uses reasonable defaults.

    Returns
    -------
    od.Metric
        A MAITE-compliant object detection metric.
    """
    mc_map50_params = {
        "box_format": "xyxy",
        "iou_type": "bbox",
        "iou_thresholds": [0.5],
        "rec_thresholds": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "max_detection_thresholds": [1, 10, 100],
        "class_metrics": True,
        "extended_summary": False,
        "average": "macro",
    }
    _class_map50 = MeanAveragePrecision(**mc_map50_params)
    return TorchODMultiClassMap50(
        _class_map50, return_key="map_50_classwise", metric_id=f"mAP_50_classwise_{id_hash(**mc_map50_params)}"
    )
