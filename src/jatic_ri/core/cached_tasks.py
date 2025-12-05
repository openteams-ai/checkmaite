import functools
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias, TypeVar, cast

import maite.protocols.image_classification as ic
import maite.protocols.object_detection as od
import maite.tasks
import pydantic
from maite._internals.protocols import generic as gen
from maite.protocols import ArrayLike
from maite.protocols.image_classification import DatumMetadataType as ImageClassificationDatumMetadataType
from maite.protocols.image_classification import InputType as ImageClassificationInputType
from maite.protocols.image_classification import TargetType as ImageClassificationTargetType
from maite.protocols.object_detection import DatumMetadataType as ObjectDetectionDatumMetadataType
from maite.protocols.object_detection import InputType as ObjectDetectionInputType

from jatic_ri import (
    cache_path,
)
from jatic_ri.core._cache import PydanticCache
from jatic_ri.core.capability_core import CapabilityOutputsBase

__all__ = ["predict", "evaluate_from_predictions", "evaluate"]


PModel = TypeVar("PModel", bound=pydantic.BaseModel)

SomeInputType: TypeAlias = ic.InputType | od.InputType
SomeTargetType: TypeAlias = ic.TargetType | od.TargetType
SomeMetadataType: TypeAlias = ic.DatumMetadataType | od.DatumMetadataType
T_Input = TypeVar("T_Input", bound=SomeInputType)
T_Target = TypeVar("T_Target", bound=SomeTargetType)
T_Metadata = TypeVar("T_Metadata", bound=SomeMetadataType)


# this is required so that Pydantic knows how to de/serialize the object detection TargetType
class PydanticCompatObjectDetectionTarget(CapabilityOutputsBase):
    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike


PydanticCompatTargetBatchType = Sequence[PydanticCompatObjectDetectionTarget] | Sequence[ImageClassificationTargetType]


def _make_task_cache(task: str, model_type: type[PModel]) -> PydanticCache[PModel]:
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


class _PredictCall(CapabilityOutputsBase):
    config: _PredictConfig | None
    predictions: list[PydanticCompatTargetBatchType]
    augmented_data: list[
        tuple[
            Sequence[ObjectDetectionInputType] | Sequence[ImageClassificationInputType],
            PydanticCompatTargetBatchType,
            Sequence[ObjectDetectionDatumMetadataType] | Sequence[ImageClassificationDatumMetadataType],
        ]
    ]


_predict_cache = _make_task_cache("predict", _PredictCall)


def _predict(
    *,
    model: gen.Model[T_Input, T_Target],
    dataloader: gen.DataLoader[T_Input, T_Target, T_Metadata] | None = None,
    dataset: gen.Dataset[T_Input, T_Target, T_Metadata] | None = None,
    batch_size: int = 1,
    augmentation: gen.Augmentation[
        T_Input,
        T_Target,
        T_Metadata,
        T_Input,
        T_Target,
        T_Metadata,
    ]
    | None = None,
    return_augmented_data: bool = False,
    dataset_id: str | None = None,
    use_cache: bool = True,
) -> tuple[
    Sequence[Sequence[T_Target]],
    Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]],
    _PredictConfig | None,
]:
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

    if key is not None and use_cache:
        call = _predict_cache.get(key)
        if call is not None:
            # pydantic types don't play well with generics,
            ps = cast(Sequence[Sequence[T_Target]], call.predictions)
            ads = cast(
                Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]], call.augmented_data
            )
            return ps, ads, call.config

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

    if key is not None and use_cache:
        _predict_cache.set(key, call)

    return ps, ads, config


# TODO: add overloads so that we can make signature more precise for users
def predict(
    *,
    model: gen.Model[T_Input, T_Target],
    dataloader: gen.DataLoader[T_Input, T_Target, T_Metadata] | None = None,
    dataset: gen.Dataset[T_Input, T_Target, T_Metadata] | None = None,
    batch_size: int = 1,
    augmentation: gen.Augmentation[
        T_Input,
        T_Target,
        T_Metadata,
        T_Input,
        T_Target,
        T_Metadata,
    ]
    | None = None,
    return_augmented_data: bool = False,
    dataset_id: str | None = None,
    use_cache: bool = True,
) -> tuple[Sequence[Sequence[T_Target]], Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]]]:
    "Generate predictions using a model and dataset."

    ps, ads, _ = _predict(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        batch_size=batch_size,
        augmentation=augmentation,
        return_augmented_data=return_augmented_data,
        dataset_id=dataset_id,
        use_cache=use_cache,
    )
    return ps, ads


class _EvaluateFromPredictionsConfig(_ConfigBase):
    model_id: str | None = None
    dataset_id: str | None = None
    augmentation_id: str | None = None
    metric_id: str


class _EvaluateFromPredictionsCall(CapabilityOutputsBase):
    config: _EvaluateFromPredictionsConfig
    metric_results: dict[str, Any]


_evaluate_from_predictions_cache = _make_task_cache("evaluate-from-predictions", _EvaluateFromPredictionsCall)


def _evaluate_from_predictions(
    *,
    metric: gen.Metric[T_Target],
    predictions: Sequence[Sequence[T_Target]],
    targets: Sequence[Sequence[T_Target]],
    key: str | _PredictConfig | None,
    use_cache: bool = True,
) -> dict[str, Any]:
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

    if key is not None and use_cache:
        call = _evaluate_from_predictions_cache.get(key)
        if call is not None:
            return call.metric_results

    metric_results = maite.tasks.evaluate_from_predictions(
        metric=metric,
        predictions=predictions,
        targets=targets,
    )

    call = _EvaluateFromPredictionsCall.model_validate(
        {"config": config, "metric_results": metric_results}, from_attributes=True
    )

    if key is not None and use_cache:
        _evaluate_from_predictions_cache.set(key, call)

    return metric_results


def evaluate_from_predictions(
    *,
    metric: gen.Metric[T_Target],
    predictions: Sequence[Sequence[T_Target]],
    targets: Sequence[Sequence[T_Target]],
    key: str | _PredictConfig | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    "Evaluate pre-calculated predictions against target (truth) data for some specified metric."
    return _evaluate_from_predictions(
        metric=metric, predictions=predictions, targets=targets, key=key, use_cache=use_cache
    )


def evaluate(
    *,
    model: gen.Model[T_Input, T_Target],
    metric: gen.Metric[T_Target],
    dataloader: gen.DataLoader[T_Input, T_Target, T_Metadata] | None = None,
    dataset: gen.Dataset[T_Input, T_Target, T_Metadata] | None = None,
    batch_size: int = 1,
    augmentation: gen.Augmentation[
        T_Input,
        T_Target,
        T_Metadata,
        T_Input,
        T_Target,
        T_Metadata,
    ]
    | None = None,
    return_augmented_data: bool = False,
    return_preds: bool = False,
    dataset_id: str | None = None,
    use_cache: bool = True,
) -> tuple[
    dict[str, Any],
    Sequence[Sequence[T_Target]],
    Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]],
]:
    "Evaluate a model's performance on data according to some metric with optional augmentation."
    ps, ads, config = _predict(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        batch_size=batch_size,
        augmentation=augmentation,
        return_augmented_data=True,  # must be True as otherwise no targets available for evaluate_from_predicitions
        dataset_id=dataset_id,
        use_cache=use_cache,
    )
    metric_results = _evaluate_from_predictions(
        metric=metric,
        predictions=ps,
        targets=[d[1] for d in ads],
        key=config.uid if config else None,
        use_cache=use_cache,
    )
    return (
        metric_results,
        ps if return_preds else [],
        ads if return_augmented_data else [],
    )
