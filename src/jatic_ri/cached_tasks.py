"""Evaluation and Prediction tool for Test Stages"""

import functools
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import maite.tasks
import pydantic
from maite._internals.protocols import generic
from maite.protocols import ArrayLike
from maite.protocols.image_classification import DatumMetadataType as ImageClassificationDatumMetadataType
from maite.protocols.image_classification import InputType as ImageClassificationInputType
from maite.protocols.image_classification import TargetType as ImageClassificationTargetType
from maite.protocols.object_detection import DatumMetadataType as ObjectDetectionDatumMetadataType
from maite.protocols.object_detection import InputType as ObjectDetectionInputType

from jatic_ri import cache_path
from jatic_ri._common.test_stages.interfaces.test_stage import OutputsBase
from jatic_ri.util._cache import PydanticCache

__all__ = ["predict"]


TModel = TypeVar("TModel", bound=pydantic.BaseModel)


class ObjectDetectionTarget(OutputsBase):
    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike


ObjectDetectionTargetType = ObjectDetectionTarget
ObjectDetectionTargetBatchType = Sequence[ObjectDetectionTargetType]
ImageClassificationTargetBatchType = Sequence[ImageClassificationTargetType]
SomeTargetBatchType = ImageClassificationTargetBatchType | ObjectDetectionTargetBatchType

SomeInputBatchType: TypeAlias = Sequence[ImageClassificationInputType] | Sequence[ObjectDetectionInputType]
SomeMetadataBatchType: TypeAlias = (
    Sequence[ImageClassificationDatumMetadataType] | Sequence[ObjectDetectionDatumMetadataType]
)


def _make_task_cache(task: str, model_type: type[TModel]) -> PydanticCache[TModel]:
    class TaskCache(PydanticCache[model_type]):
        def path(self, key: str) -> Path:
            d = cache_path() / "cached-tasks" / task
            d.mkdir(parents=True, exist_ok=True)
            return d / key

    return TaskCache()


class _ConfigBase(pydantic.BaseModel):
    @functools.cached_property
    def uid(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


class _PredictConfig(_ConfigBase):
    model_id: str
    dataset_id: str
    augmentation_id: str | None
    return_augmented_data: bool


class _PredictCall(OutputsBase):
    config: _PredictConfig | None
    predictions: list[SomeTargetBatchType]
    augmented_data: list[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]]


_predict_cache = _make_task_cache("predict", _PredictCall)


def _predict(
    *,
    model: generic.Model,
    dataloader: generic.DataLoader | None,
    dataset: generic.Dataset | None,
    batch_size: int,
    augmentation: generic.Augmentation | None,
    return_augmented_data: bool,
    dataset_id: str | None,
) -> _PredictCall:
    model_id = model.metadata["id"]
    model_index2label = getattr(model.metadata, "index2label", {})
    dataset_id = dataset.metadata["id"] if dataset is not None else dataset_id
    dataset_index2label = getattr(dataset.metadata, "index2label", {}) if dataset is not None else {}

    error_msg = f"""
    The model and dataset index to label mappings do not match which is likely a user error.

    Model index to label mapping: {model_index2label}
    Dataset index to label mapping: {dataset_index2label}
    Model ID: {model_id}
    Dataset ID: {dataset_id}

    If this mismatch is not actually an error, please contact the RI team to discuss your use-case.
"""
    if model_index2label != dataset_index2label:
        raise ValueError(error_msg)

    augmentation_id = augmentation.metadata["id"] if augmentation else None
    config: _PredictConfig | None

    if dataset_id is None:
        config = key = None
    else:
        config = _PredictConfig(
            model_id=model_id,
            dataset_id=dataset_id,
            augmentation_id=augmentation_id,
            return_augmented_data=return_augmented_data,
        )
        key = config.uid

    if key is not None:
        call = _predict_cache.get(key)
        if call is not None:
            return call

    ps, ads = maite.tasks.predict(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        batch_size=batch_size,
        augmentation=augmentation,
        return_augmented_data=return_augmented_data,
    )

    call = _PredictCall.model_validate(
        {"config": config, "predictions": ps, "augmented_data": ads}, from_attributes=True
    )

    if key is not None:
        _predict_cache.set(key, call)

    return call


def predict(
    *,
    model: generic.Model,
    dataloader: generic.DataLoader | None = None,
    dataset: generic.Dataset | None = None,
    batch_size: int = 1,
    augmentation: generic.Augmentation | None = None,
    return_augmented_data: bool = False,
    dataset_id: str | None = None,
) -> tuple[
    Sequence[SomeTargetBatchType], Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]]
]:
    "Generate predictions using a model and dataset."

    call = _predict(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        batch_size=batch_size,
        augmentation=augmentation,
        return_augmented_data=return_augmented_data,
        dataset_id=dataset_id,
    )
    return call.predictions, call.augmented_data


class _EvaluateFromPredictionsConfig(_ConfigBase):
    model_id: str | None = None
    dataset_id: str | None = None
    augmentation_id: str | None = None
    metric_id: str


class _EvaluateFromPredictionsCall(OutputsBase):
    config: _EvaluateFromPredictionsConfig
    metric_results: dict[str, Any]


_evaluate_from_predictions_cache = _make_task_cache("evaluate-from-predictions", _EvaluateFromPredictionsCall)


def _evaluate_from_predictions(
    *,
    metric: generic.Metric,
    predictions: Sequence[SomeTargetBatchType],
    targets: Sequence[SomeTargetBatchType],
    key: str | _PredictConfig | None,
) -> _EvaluateFromPredictionsCall:
    metric_id = metric.metadata["id"]

    if isinstance(key, _PredictConfig):
        config = _EvaluateFromPredictionsConfig(
            model_id=key.model_id,
            dataset_id=key.dataset_id,
            augmentation_id=key.augmentation_id,
            metric_id=metric_id,
        )
        key = config.uid
    else:
        config = _EvaluateFromPredictionsConfig(metric_id=metric_id)

    if key is not None:
        call = _evaluate_from_predictions_cache.get(key)
        if call is not None:
            return call

    metric_results = maite.tasks.evaluate_from_predictions(
        metric=metric,
        predictions=predictions,
        targets=targets,
    )

    call = _EvaluateFromPredictionsCall.model_validate(
        {"config": config, "metric_results": metric_results}, from_attributes=True
    )

    if key is not None:
        _evaluate_from_predictions_cache.set(key, call)

    return call


def evaluate_from_predictions(
    *,
    metric: generic.Metric,
    predictions: Sequence[SomeTargetBatchType],
    targets: Sequence[SomeTargetBatchType],
    key: str | _PredictConfig | None = None,
) -> dict[str, Any]:
    call = _evaluate_from_predictions(metric=metric, predictions=predictions, targets=targets, key=key)
    return call.metric_results


def evaluate(
    *,
    model: generic.Model,
    metric: generic.Metric,
    dataloader: generic.DataLoader | None = None,
    dataset: generic.Dataset | None = None,
    batch_size: int = 1,
    augmentation: generic.Augmentation | None = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
    dataset_id: str | None = None,
) -> tuple[
    dict[str, Any],
    list[SomeTargetBatchType],
    Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]],
]:
    call = _predict(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        batch_size=batch_size,
        augmentation=augmentation,
        return_augmented_data=True,  # must be True as otherwise no targets available for evaluate_from_predicitions
        dataset_id=dataset_id,
    )
    metric_results = evaluate_from_predictions(
        metric=metric,
        predictions=call.predictions,
        targets=[d[1] for d in call.augmented_data],
        key=call.config.uid if call.config else None,
    )
    return (
        metric_results,
        call.predictions if return_preds else [],
        call.augmented_data if return_augmented_data else [],
    )
