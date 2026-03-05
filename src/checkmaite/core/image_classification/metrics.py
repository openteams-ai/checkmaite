from collections.abc import Sequence
from typing import Literal

import torch
from maite.protocols import MetricMetadata
from maite.protocols import image_classification as ic
from torchmetrics import Accuracy, F1Score
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification.stat_scores import MulticlassStatScores

from checkmaite.core._utils import id_hash

__all__: list[str] = [
    "TorchICMulticlassMetric",
    "accuracy_multiclass_torch_metric_factory",
    "f1score_multiclass_torch_metric_factory",
]


class MetricInputDataError(Exception):
    pass


class InvalidMetricTypeError(Exception):
    pass


class TorchICMulticlassMetric(ic.Metric):
    """A MAITE-compliant wrapper for image classification metrics from the torchmetric package.

    The purpose of this wrapper is primarily to convert MAITE-compliant batches
    of prediction target data for the image classification sub-problem into
    data structures compatible with the instantiated TorchMetric.

    This Class wrapper only handles the _(one-dimensional) multi-class_ type
    of the IC subproblem, i.e. each image is assigned exactly one of two or
    more classes (e.g. Dog, Cat, Fish...). Other metric types require different
    shapes and dtypes which, when also considering that the class must handle
    any MAITE-compliant parameter structures, would add additional complexity
    to handle which is not necessary for our use case.
    (Reference: https://torchmetrics.readthedocs.io/en/v0.10.2/pages/classification.html#input-types)

    Parameters
    ----------
    ic_metric : TorchMetric
        The torchmetric object to wrap.
    return_key : str
        The key to use for the metric result in the output dictionary.
    metric_id : str, optional
        The ID for the metric metadata, by default "torchICMulticlass".

    Raises
    ------
    InvalidMetricTypeError
        If the provided TorchMetric is not a multiclass metric.
    """

    def __init__(self, ic_metric: TorchMetric, return_key: str, *, metric_id: str) -> None:
        """Initialize a TorchICMulticlassMetric.

        Parameters:
            ic_metric: A TorchMetric instance for multiclass classification metrics
            return_key: The key to use in the returned dictionary from compute()
            metric_id: Identifier for metric.

        Raises:
            InvalidMetricTypeError: If the provided metric is not a multiclass metric
        """
        if not issubclass(type(ic_metric), MulticlassStatScores):
            raise InvalidMetricTypeError(
                """
                Provided TorchMetric is not a multiclass metric (it is not a subclass
                of 'torchmetrics.classification.stat_scores.MulticlassStatScores').
                """
            )
        self._ic_metric: TorchMetric = ic_metric
        self.return_key: str = return_key
        self.metadata = MetricMetadata(id=metric_id)

    def update(
        self,
        preds: Sequence[ic.TargetType],
        targets: Sequence[ic.TargetType],
        metadatas: Sequence[ic.DatumMetadataType] = [],  # noqa: ARG002
    ) -> None:
        """Add predictions and targets to metric's cache for later calculation.

        Parameters
        ----------
        preds : Sequence[ic.TargetType]
            A batch of predictions.
        targets : Sequence[ic.TargetType]
            A batch of targets.

        Raises
        ------
        MetricInputDataError
            If prediction and target batches are not of equal length.
        """
        if not len(preds) == len(targets):
            raise MetricInputDataError(
                f"""
                Prediction and Target batches are not of equal length.
                Prediction has length {len(preds)} while Target has length {len(targets)}.
                """
            )

        # Convert each ArrayLike item within the prediction and target batch Sequences to Tensors
        preds_batch_list_of_tensors: Sequence[torch.Tensor] = [torch.as_tensor(x) for x in preds]
        targets_batch_list_of_tensors: Sequence[torch.Tensor] = [torch.as_tensor(x) for x in targets]

        # Assemble the Sequence of 1-dim Tensors as one 2-dim Tensor
        preds_batch_tensor: torch.Tensor = torch.stack(preds_batch_list_of_tensors)
        targets_batch_tensor: torch.Tensor = torch.stack(targets_batch_list_of_tensors)

        # Update metric - convert probabilities/logits to integer class indicies predictions and targets
        self._ic_metric.update(preds_batch_tensor.argmax(1), targets_batch_tensor.argmax(1))

    def compute(self) -> dict[str, float]:
        """Compute metric value(s) for currently cached predictions and targets.

        Returns
        -------
        dict[str, float]
            A dictionary containing the computed metric value.
        """
        return {self.return_key: self._ic_metric.compute()}

    def reset(self) -> None:
        """Clear contents of current metric's cache of predictions and targets."""
        self._ic_metric.reset()


def accuracy_multiclass_torch_metric_factory(
    num_classes: int,
    average: Literal["micro", "macro", "weighted"] = "micro",
) -> TorchICMulticlassMetric:
    """Factory for create a MAITE-compliant wrapper of the multiclass Accuracy TorchMetric.

    Parameters
    ----------
    num_classes : int
        Number of classes (e.g. Dog, Cat, Fish...) in the IC task.
    average : Literal["micro", "macro", "weighted"], optional
        Type of average to be taken, by default "micro".
        In short, 'micro' averages in the context of Accuracy refers to the
        simple calculation of "number of correct classifications over total
        attempts", while both 'macro' and 'weighted' adjust the statistic
        to consider the per-class distribution of targets in the test data.

        Note that TorchMetric's Accuracy subclasses will also accept
        'average="none"'. However, this will change the output of compute()
        from a scalar to a Tensor, so our type hints will not consider this
        value as it will add complexity to workflows which are not necessary
        for our use case. We also do not parameterize 'multidim_average'
        (it will default to "global").

    Returns
    -------
    TorchICMulticlassMetric
        A MAITE-compliant metric object.
    """
    _tm_accuracy = Accuracy(task="multiclass", num_classes=num_classes, average=average)

    return TorchICMulticlassMetric(
        _tm_accuracy,
        return_key="accuracy",
        metric_id=f"torch_ic_accuracy_{id_hash(num_classes=num_classes, average=average)}",
    )


def f1score_multiclass_torch_metric_factory(
    num_classes: int,
    average: Literal["micro", "macro", "weighted"] = "macro",
) -> TorchICMulticlassMetric:
    """Factory for create a MAITE-compliant wrapper of the multiclass F1Score TorchMetric.

    Parameters
    ----------
    num_classes : int
        Number of classes (e.g. Dog, Cat, Fish...) in the IC task.
    average : Literal["micro", "macro", "weighted"], optional
        Type of average to be taken, by default "macro".
        Average type 'macro' in the context of F1Score refers to calculating
        the per-class F1 score statistics rather than averaging their results.
        As F1 score is already a derivation of precision and recall
        (statistics which are inherently class-based), we consider 'macro'
        averaging to be the natural default for F1 Score. (TorchMetrics
        does the same.)

        Note that TorchMetric's F1Score subclasses will also accept
        'average="none"'. However, this will change the output of compute()
        from a scalar to a Tensor, so our type hints will not consider this
        value as it will add complexity to workflows which are not necessary
        for our use case. We also do not parameterize 'multidim_average'
        (it will default to "global").

    Returns
    -------
    TorchICMulticlassMetric
        A MAITE-compliant metric object.
    """

    _tm_f1score = F1Score(task="multiclass", num_classes=num_classes, average=average)
    return TorchICMulticlassMetric(
        _tm_f1score,
        return_key="f1_score",
        metric_id=f"torch_ic_f1score_{id_hash(num_classes=num_classes, average=average)}",
    )
