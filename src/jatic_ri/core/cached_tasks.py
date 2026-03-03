import abc
import functools
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar, cast

import maite.protocols.image_classification as ic
import maite.protocols.object_detection as od
import maite.tasks
import pydantic
from maite.protocols import ArrayLike
from maite.protocols import generic as gen
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


def _get_id(
    obj: gen.Model | gen.Dataset | gen.Augmentation | gen.Metric | None,
    *,
    id: str | None = None,  # noqa: A002
) -> str | None:
    if obj is not None and id is not None:
        raise ValueError
    if obj is not None:
        return obj.metadata["id"]
    if id is not None:
        return id
    return None


class _ConfigBase(pydantic.BaseModel, abc.ABC):
    @functools.cached_property
    def cache_key(self) -> str | None:
        if not self.is_valid_for_caching():
            return None

        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    @abc.abstractmethod
    def is_valid_for_caching(self) -> bool: ...


StrictReturnAugmentedData = Literal["all", "targets", "none"]
ReturnAugmentedData = bool | StrictReturnAugmentedData


def _parse_return_augmented_data(rad: ReturnAugmentedData) -> StrictReturnAugmentedData:
    if isinstance(rad, bool):
        return "all" if rad else "none"
    return rad


class _PredictConfig(_ConfigBase):
    model_id: str | None
    dataset_id: str | None
    augmentation_id: str | None
    return_augmented_data: StrictReturnAugmentedData

    def is_valid_for_caching(self) -> bool:
        # self.augmentation_id is None is a valid use case and should not result in an invalid cache key
        return not any(id is None for id in [self.model_id, self.dataset_id])  # noqa: A001


class _PredictCall(CapabilityOutputsBase):
    config: _PredictConfig
    predictions: list[PydanticCompatTargetBatchType]
    augmented_data: list[
        tuple[
            Sequence[ObjectDetectionInputType] | Sequence[ImageClassificationInputType],
            PydanticCompatTargetBatchType,
            Sequence[ObjectDetectionDatumMetadataType] | Sequence[ImageClassificationDatumMetadataType],
        ]
    ]


_predict_cache = _make_task_cache("predict", _PredictCall)


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
    return_augmented_data: ReturnAugmentedData = False,
    dataset_id: str | None = None,
    use_cache: bool = True,
) -> tuple[Sequence[Sequence[T_Target]], Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]]]:
    "Generate predictions using a model and dataset."

    model_id = _get_id(obj=model)
    model_index2label = getattr(model.metadata, "index2label", {})
    dataset_id = _get_id(dataset, id=dataset_id)
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

    strict_return_augmented_data = _parse_return_augmented_data(return_augmented_data)

    config = _PredictConfig(
        model_id=model_id,
        dataset_id=dataset_id,
        augmentation_id=_get_id(augmentation) if augmentation else None,
        return_augmented_data=strict_return_augmented_data,
    )

    if config.cache_key is not None and use_cache:
        call = _predict_cache.get(config.cache_key)
        if call is not None:
            # pydantic types don't play well with generics,
            ps = cast(Sequence[Sequence[T_Target]], call.predictions)
            ads = cast(
                Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]], call.augmented_data
            )
            return ps, ads

    ps, ads = maite.tasks.predict(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        batch_size=batch_size,
        augmentation=augmentation,
        return_augmented_data=strict_return_augmented_data != "none",
    )

    if strict_return_augmented_data == "targets":
        ads = [(cast(Sequence[T_Input], []), targets, metadata) for _, targets, metadata in ads]

    if config.cache_key is not None and use_cache:
        _predict_cache.set(
            config.cache_key,
            _PredictCall.model_validate(
                {"config": config, "predictions": ps, "augmented_data": ads}, from_attributes=True
            ),
        )

    return ps, ads


class _EvaluateFromPredictionsConfig(_ConfigBase):
    model_id: str | None
    dataset_id: str | None
    augmentation_id: str | None
    metric_id: str | None

    def is_valid_for_caching(self) -> bool:
        # self.augmentation_id is None is a valid use case and should not result in an invalid cache key
        return not any(id is None for id in [self.model_id, self.dataset_id, self.metric_id])  # noqa: A001


class _EvaluateFromPredictionsCall(CapabilityOutputsBase):
    config: _EvaluateFromPredictionsConfig
    metric_results: dict[str, Any]


_evaluate_from_predictions_cache = _make_task_cache("evaluate-from-predictions", _EvaluateFromPredictionsCall)


def evaluate_from_predictions(
    *,
    metric: gen.Metric[T_Target, T_Metadata],
    predictions: Sequence[Sequence[T_Target]],
    targets: Sequence[Sequence[T_Target]],
    model: gen.Model | None = None,
    model_id: str | None = None,
    dataset: gen.Dataset | None = None,
    dataset_id: str | None = None,
    augmentation: gen.Augmentation | None = None,
    augmentation_id: str | None = None,
    use_cache: bool = True,
) -> Mapping[str, Any]:
    "Evaluate pre-calculated predictions against target (truth) data for some specified metric."
    config = _EvaluateFromPredictionsConfig(
        model_id=_get_id(model, id=model_id),
        dataset_id=_get_id(dataset, id=dataset_id),
        augmentation_id=_get_id(augmentation, id=augmentation_id),
        metric_id=_get_id(metric),
    )

    if config.cache_key is not None and use_cache:
        call = _evaluate_from_predictions_cache.get(config.cache_key)
        if call is not None:
            return call.metric_results

    metric_results = maite.tasks.evaluate_from_predictions(
        metric=metric, pred_batches=predictions, target_batches=targets, metadata_batches=[[] for _ in targets]
    )

    if config.cache_key is not None and use_cache:
        _evaluate_from_predictions_cache.set(
            config.cache_key,
            _EvaluateFromPredictionsCall.model_validate(
                {"config": config, "metric_results": metric_results}, from_attributes=True
            ),
        )

    return metric_results


def evaluate(
    *,
    model: gen.Model[T_Input, T_Target],
    metric: gen.Metric[T_Target, T_Metadata],
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
    return_augmented_data: ReturnAugmentedData = False,
    return_preds: bool = False,
    dataset_id: str | None = None,
    use_cache: bool = True,
) -> tuple[
    Mapping[str, Any],
    Sequence[Sequence[T_Target]],
    Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]],
]:
    "Evaluate a model's performance on data according to some metric with optional augmentation."

    strict_return_augmented_data = _parse_return_augmented_data(return_augmented_data)
    ps, ads = predict(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        batch_size=batch_size,
        augmentation=augmentation,
        return_augmented_data="targets" if strict_return_augmented_data == "none" else strict_return_augmented_data,
        dataset_id=dataset_id,
        use_cache=use_cache,
    )
    metric_results = evaluate_from_predictions(
        metric=metric,
        predictions=ps,
        targets=[d[1] for d in ads],
        model=model,
        dataset=dataset,
        dataset_id=dataset_id,
        augmentation=augmentation,
        use_cache=use_cache,
    )
    return (
        metric_results,
        ps if return_preds else [],
        ads if strict_return_augmented_data != "none" else [],
    )
