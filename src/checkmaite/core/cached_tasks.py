import abc
import functools
import hashlib
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast

import maite.tasks
import pydantic
from maite.protocols import AugmentationMetadata, MetricMetadata
from maite.protocols import generic as gen

from checkmaite import (
    cache_path,
)
from checkmaite.core._cache import PydanticCache, warn_optional_cache_failure
from checkmaite.core._cached_metadata import (
    CachedDatumMetadata,
    require_unchanged_cache_value,
    validate_cache_value,
    validate_metadata_value,
)
from checkmaite.core.capability_core import CapabilityOutputsBase
from checkmaite.core.evaluation_tasks import EvaluationTask, get_task

__all__ = ["TaskCachedTasks", "evaluate", "evaluate_from_predictions", "for_task", "predict"]

PModel = TypeVar("PModel", bound=pydantic.BaseModel)

T_Input = TypeVar("T_Input")
T_Target = TypeVar("T_Target")
T_Metadata = TypeVar("T_Metadata")


# TODO: Remove this workaround once https://github.com/mit-ll-ai-technology/maite/issues/45
# is fixed in CheckMAITE's minimum supported MAITE version.
class _MaterializingAugmentation(Generic[T_Input, T_Target, T_Metadata]):
    def __init__(
        self,
        augmentation: gen.Augmentation[T_Input, T_Target, T_Metadata, T_Input, T_Target, T_Metadata] | None,
        task: EvaluationTask,
    ) -> None:
        self._augmentation = augmentation
        self._task = task
        self.metadata: AugmentationMetadata = (
            augmentation.metadata if augmentation is not None else {"id": "checkmaite-materialize-mot-streams"}
        )

    def __call__(
        self,
        batch: tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]],
    ) -> tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]:
        inputs, targets, metadata = self._augmentation(batch) if self._augmentation is not None else batch
        if self._task.materialize_inputs:
            inputs = [
                value if isinstance(value, Sequence) else cast(T_Input, list(cast(Any, value))) for value in inputs
            ]
        return inputs, targets, metadata


def _materializing_augmentation(
    augmentation: gen.Augmentation[T_Input, T_Target, T_Metadata, T_Input, T_Target, T_Metadata] | None,
    task: EvaluationTask,
) -> gen.Augmentation[T_Input, T_Target, T_Metadata, T_Input, T_Target, T_Metadata]:
    return _MaterializingAugmentation(augmentation, task)


def _make_task_cache(task: str, model_type: type[PModel]) -> PydanticCache[PModel]:
    class TaskCache(PydanticCache[model_type]):
        def path(self, key: str) -> Path:
            d = cache_path() / "cached-tasks" / task
            d.mkdir(parents=True, exist_ok=True)
            return d / key

    return TaskCache()


def _try_publish(cache: PydanticCache[PModel], key: str, model_type: type[PModel], data: Any) -> bool:
    try:
        value = model_type.model_validate(data, from_attributes=True)
    except Exception as error:  # noqa: BLE001 - caching is optional after completed work
        warn_optional_cache_failure(f"Cache publication is disabled for this value: {error}", stacklevel=2)
        return False
    return cache.try_set(key, value)


def _validate_prediction_batch(
    predictions: Sequence[Any],
    targets: Sequence[Any],
    metadata: Sequence[Any],
    *,
    strict: bool,
    location: str,
) -> None:
    if not strict:
        return
    for index, prediction in enumerate(predictions):
        validate_cache_value(prediction, location=f"{location}.predictions[{index}]")
    for index, target in enumerate(targets):
        validate_cache_value(target, location=f"{location}.targets[{index}]")
    for index, datum_metadata in enumerate(metadata):
        if type(datum_metadata) is not dict:
            raise TypeError("Strict metadata caching requires exact dictionaries.")
        validate_metadata_value(
            datum_metadata,
            location=f"{location}.metadata[{index}]",
            strict=True,
        )


def _validate_prediction_artifact(
    predictions: Sequence[Sequence[Any]],
    augmented_data: Sequence[tuple[Sequence[Any], Sequence[Any], Sequence[Any]]],
    *,
    strict: bool,
) -> None:
    for batch_index, (prediction_batch, (_, target_batch, metadata_batch)) in enumerate(
        zip(predictions, augmented_data, strict=True)
    ):
        _validate_prediction_batch(
            prediction_batch,
            target_batch,
            metadata_batch,
            strict=strict,
            location=f"batch[{batch_index}]",
        )


def _require_lossless_task_conversion(
    task: EvaluationTask,
    predictions: Sequence[Sequence[Any]],
    inference_context: Sequence[tuple[Sequence[Any], Sequence[Any], Sequence[Any]]],
) -> None:
    # The cache stores predictions and targets converted to the task's target
    # type. Strict mode checks the values before that, so also check that the
    # conversion changed nothing, such as an integer turned into a float.
    target_batches = [batch[1] for batch in inference_context]
    for name, batches in (("predictions", predictions), ("targets", target_batches)):
        converted = task.validate_target_batches(batches)
        for batch_index, (batch, converted_batch) in enumerate(zip(batches, converted, strict=True)):
            for index, (value, converted_value) in enumerate(zip(batch, converted_batch, strict=True)):
                require_unchanged_cache_value(value, converted_value, location=f"batch[{batch_index}].{name}[{index}]")


def _validate_result_artifact(metric_results: Mapping[str, Any], *, strict: bool) -> dict[str, Any]:
    if strict and type(metric_results) is not dict:
        raise TypeError("Strict metric-result caching requires an exact dict.")
    results = metric_results if type(metric_results) is dict else dict(metric_results)
    if strict:
        validate_cache_value(results, location="metric_results")
    return cast(dict[str, Any], results)


def _validate_serialization_policy(strict_cache_serialization: bool) -> None:
    if not isinstance(strict_cache_serialization, bool):
        raise TypeError("strict_cache_serialization must be a bool.")


def _resolve_id(
    obj: gen.Model | gen.Dataset | gen.Augmentation | gen.Metric | None,
    *,
    id: str | None = None,  # noqa: A002
) -> str | None:
    if obj is not None and id is not None:
        raise ValueError(f"Received object of type {type(obj)} and id; only one should be provided.")
    if obj is not None:
        return obj.metadata["id"]
    if id is not None:
        return id
    return None


class _ConfigBase(pydantic.BaseModel, abc.ABC):
    @functools.cached_property
    def cache_key(self) -> str | None:
        required_fields = self.cache_key_required_fields()
        if any(getattr(self, f) is None for f in required_fields):
            return None

        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    @abc.abstractmethod
    def cache_key_required_fields(self) -> set[str]: ...


class _CollectingMetric(Generic[T_Target, T_Metadata]):
    """Optionally evaluate online while retaining target and metadata batches."""

    def __init__(
        self,
        metric: gen.Metric[T_Target, T_Metadata] | None,
        *,
        task: EvaluationTask,
        collect_batches: bool,
        validate_batches: bool,
        require_batches: bool,
    ) -> None:
        self._metric = metric
        self._collect_batches = collect_batches
        self._validate_batches = validate_batches
        self._require_batches = require_batches
        self._task = task
        self._task_checked = False
        self.metadata: MetricMetadata = metric.metadata if metric is not None else {"id": "checkmaite-collector"}
        self.target_batches: list[Sequence[T_Target]] = []
        self.metadata_batches: list[Sequence[T_Metadata]] = []
        self._batch_count = 0

    def reset(self) -> None:
        self.target_batches.clear()
        self.metadata_batches.clear()
        self._batch_count = 0
        if self._metric is not None:
            self._metric.reset()

    def update(
        self,
        predictions: Sequence[T_Target],
        targets: Sequence[T_Target],
        metadata: Sequence[T_Metadata],
        /,
    ) -> None:
        if self._validate_batches:
            if len(predictions) != len(targets):
                raise ValueError("Corresponding prediction and target batches must have the same length.")
            if len(metadata) != len(predictions):
                raise ValueError("Corresponding prediction and metadata batches must have the same length.")
        if not self._task_checked and (len(predictions) > 0 or len(targets) > 0):
            # Checking the first non-empty batch catches a wrong task= before the run is cached.
            self._task._require_batch(predictions, targets)  # noqa: SLF001
            self._task_checked = True
        if self._collect_batches:
            self.target_batches.append(targets)
            self.metadata_batches.append(metadata)
        self._batch_count += 1
        if self._metric is not None:
            self._metric.update(predictions, targets, metadata)

    def compute(self) -> Mapping[str, Any]:
        if self._require_batches and self._batch_count == 0:
            raise ValueError("Predictions and targets must have at least one element.")
        return self._metric.compute() if self._metric is not None else {}


class _PredictConfig(_ConfigBase):
    cache_schema_version: Literal[2] = 2
    model_id: str | None
    dataset_id: str | None
    augmentation_id: str | None
    # The task named with task=, so an entry is never reused for another task.
    task: str
    # Batching can affect model outputs, padding, nondeterministic kernels, and
    # prediction batch boundaries, so it is part of the prediction identity.
    batch_size: int
    strict_cache_serialization: bool = False

    def cache_key_required_fields(self) -> set[str]:
        # self.augmentation_id is None is a valid use case and should not result in an invalid cache key
        return {"model_id", "dataset_id"}


class _PredictCall(CapabilityOutputsBase):
    """Reusable inference artifact containing raw predictions, targets, and metadata.

    Predictions and targets are validated with the ``target_adapter`` of the
    evaluation task named by ``task``.
    """

    config: _PredictConfig
    task: str
    predictions: list[Sequence[Any]]
    # Inputs are never cached, so the input batches are always empty.
    augmented_data: list[tuple[Sequence[Any], Sequence[Any], Sequence[CachedDatumMetadata]]]

    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_task_targets(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        predictions = data.get("predictions")
        augmented_data = data.get("augmented_data")
        if predictions is None or augmented_data is None:
            return data
        target_batches = [batch[1] for batch in augmented_data]

        task_name = data.get("task")
        if task_name is None:
            raise ValueError("A prediction cache entry must name its evaluation task.")
        task = get_task(task_name)

        validated_targets = task.validate_target_batches(target_batches)
        return {
            **data,
            "task": task.name,
            "predictions": task.validate_target_batches(predictions),
            "augmented_data": [
                (inputs, targets, metadata)
                for (inputs, _, metadata), targets in zip(augmented_data, validated_targets, strict=True)
            ],
        }


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
    return_augmented_data: bool = False,
    dataset_id: str | None = None,
    use_cache: bool = True,
    strict_cache_serialization: bool = False,
    task: str | EvaluationTask[T_Target],
) -> tuple[Sequence[Sequence[T_Target]], Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]]]:
    """Generate predictions through the fundamental :func:`evaluate` task.

    Setting ``return_augmented_data=True`` always runs fresh inference so the
    returned inputs and predictions share one realization. Complete inputs are
    materialized for the response but are not cached; one-shot video streams may
    therefore require substantial memory. Compatibility serialization
    is the default; ``strict_cache_serialization=True`` requires explicitly
    lossless predictions, targets, metadata, and metric results. ``task`` is
    described in :func:`evaluate`.
    """
    _, predictions, augmented_data = evaluate(
        model=model,
        metric=None,
        dataloader=dataloader,
        dataset=dataset,
        batch_size=batch_size,
        augmentation=augmentation,
        return_augmented_data=return_augmented_data,
        return_preds=True,
        dataset_id=dataset_id,
        use_cache=use_cache,
        strict_cache_serialization=strict_cache_serialization,
        task=task,
    )
    return predictions, augmented_data


class _EvaluateFromPredictionsConfig(_ConfigBase):
    cache_schema_version: Literal[2] = 2
    inference_id: str | None
    # The task requested for the predictions, as in _PredictConfig. Direct
    # evaluate_from_predictions() callers have none.
    task: str | None
    model_id: str | None
    dataset_id: str | None
    augmentation_id: str | None
    metric_id: str | None
    # Identifies the batching semantics of predictions produced by evaluate().
    # Direct evaluate_from_predictions() callers may leave this unset.
    prediction_batch_size: int | None
    # Distinguishes evaluation results produced from the same cached raw
    # predictions by different deterministic CPU postprocessing semantics.
    cpu_prediction_postprocessor_id: str | None
    strict_cache_serialization: bool = False

    def cache_key_required_fields(self) -> set[str]:
        return {"inference_id", "metric_id"}


class _EvaluateFromPredictionsCall(CapabilityOutputsBase):
    config: _EvaluateFromPredictionsConfig
    metric_results: dict[str, Any]


_evaluate_from_predictions_cache = _make_task_cache("evaluate-from-predictions", _EvaluateFromPredictionsCall)


def evaluate_from_predictions(
    *,
    metric: gen.Metric[T_Target, T_Metadata],
    predictions: Sequence[Sequence[T_Target]],
    targets: Sequence[Sequence[T_Target]],
    metadata_batches: Sequence[Sequence[T_Metadata]],
    model: gen.Model | None = None,
    model_id: str | None = None,
    dataset: gen.Dataset | None = None,
    dataset_id: str | None = None,
    augmentation: gen.Augmentation | None = None,
    augmentation_id: str | None = None,
    prediction_batch_size: int | None = None,
    cpu_prediction_postprocessor_id: str | None = None,
    inference_id: str | None = None,
    use_cache: bool = True,
    strict_cache_serialization: bool = False,
) -> Mapping[str, Any]:
    """Evaluate pre-calculated predictions against targets with one metric.

    ``metadata_batches`` supplies datum metadata paired with each prediction
    batch. ``prediction_batch_size`` identifies batching used to produce model
    predictions when this function is called through :func:`evaluate`.
    ``cpu_prediction_postprocessor_id`` is the stable identity of deterministic
    CPU postprocessing already applied to ``predictions``. Result caching is
    disabled unless ``inference_id`` identifies the supplied predictions, targets,
    metadata, and batching. The serialization policy is part of that identity.
    """
    if len(targets) != len(predictions) or len(metadata_batches) != len(predictions):
        raise ValueError("Predictions, targets, and metadata must have the same number of batches.")
    for prediction_batch, target_batch, metadata_batch in zip(predictions, targets, metadata_batches, strict=True):
        if len(target_batch) != len(prediction_batch) or len(metadata_batch) != len(prediction_batch):
            raise ValueError("Corresponding prediction, target, and metadata batches must have the same length.")

    _validate_serialization_policy(strict_cache_serialization)

    config = _EvaluateFromPredictionsConfig(
        inference_id=inference_id,
        task=None,
        model_id=_resolve_id(model, id=model_id),
        dataset_id=_resolve_id(dataset, id=dataset_id),
        augmentation_id=_resolve_id(augmentation, id=augmentation_id),
        metric_id=_resolve_id(metric),
        prediction_batch_size=prediction_batch_size,
        cpu_prediction_postprocessor_id=cpu_prediction_postprocessor_id,
        strict_cache_serialization=strict_cache_serialization,
    )

    if not config.cache_key and use_cache:
        warnings.warn(
            "use_cache was requested but caching is disabled because at least one of the following is None:"
            + str(config.cache_key_required_fields()),
            stacklevel=2,
        )

    if config.cache_key is not None and use_cache:
        call = _evaluate_from_predictions_cache.get(config.cache_key)
        if call is not None:
            return call.metric_results

    metric_results = maite.tasks.evaluate_from_predictions(
        metric=metric,
        pred_batches=predictions,
        target_batches=targets,
        metadata_batches=metadata_batches,
    )

    if config.cache_key is not None and use_cache:
        try:
            cached_metric_results = _validate_result_artifact(
                metric_results,
                strict=strict_cache_serialization,
            )
        except Exception as error:  # noqa: BLE001 - caching is optional after completed work
            warn_optional_cache_failure(f"Cache publication is disabled for this value: {error}", stacklevel=2)
        else:
            _try_publish(
                _evaluate_from_predictions_cache,
                config.cache_key,
                _EvaluateFromPredictionsCall,
                {"config": config, "metric_results": cached_metric_results},
            )

    return metric_results


@dataclass(frozen=True)
class _EvaluationCacheState:
    enabled: bool = False
    key: str | None = None
    config: _EvaluateFromPredictionsConfig | None = None
    call: _EvaluateFromPredictionsCall | None = None


def _prepare_evaluation_cache(
    *,
    metric: gen.Metric[Any, Any] | None,
    inference_id: str | None,
    task: str | None,
    model_id: str | None,
    dataset_id: str | None,
    augmentation_id: str | None,
    batch_size: int,
    cpu_prediction_postprocessor: Callable[[Sequence[Sequence[Any]]], Sequence[Sequence[Any]]] | None,
    cpu_prediction_postprocessor_id: str | None,
    return_augmented_data: bool,
    use_cache: bool,
    strict_cache_serialization: bool,
) -> _EvaluationCacheState:
    if metric is None:
        return _EvaluationCacheState()

    evaluation_use_cache = use_cache
    if cpu_prediction_postprocessor is not None and cpu_prediction_postprocessor_id is None and use_cache:
        warnings.warn(
            "use_cache was requested but evaluation-result caching is disabled because "
            "cpu_prediction_postprocessor_id was not provided for the cpu_prediction_postprocessor.",
            stacklevel=3,
        )
        evaluation_use_cache = False

    config = _EvaluateFromPredictionsConfig(
        inference_id=inference_id,
        task=task,
        model_id=model_id,
        dataset_id=dataset_id,
        augmentation_id=augmentation_id,
        metric_id=_resolve_id(metric),
        prediction_batch_size=batch_size,
        cpu_prediction_postprocessor_id=cpu_prediction_postprocessor_id,
        strict_cache_serialization=strict_cache_serialization,
    )
    key = config.cache_key
    if key is None and evaluation_use_cache:
        warnings.warn(
            "use_cache was requested but caching is disabled because at least one of the following is None:"
            + str(config.cache_key_required_fields()),
            stacklevel=3,
        )
    enabled = key is not None and evaluation_use_cache
    call = _evaluate_from_predictions_cache.get(key) if enabled and not return_augmented_data and key else None
    return _EvaluationCacheState(enabled=enabled, key=key, config=config, call=call)


def _store_evaluation_result(state: _EvaluationCacheState, metric_results: Mapping[str, Any]) -> None:
    if state.enabled and state.key is not None and state.config is not None:
        try:
            cached_metric_results = _validate_result_artifact(
                metric_results,
                strict=state.config.strict_cache_serialization,
            )
        except Exception as error:  # noqa: BLE001 - caching is optional after completed work
            warn_optional_cache_failure(f"Cache publication is disabled for this value: {error}", stacklevel=2)
            return
        _try_publish(
            _evaluate_from_predictions_cache,
            state.key,
            _EvaluateFromPredictionsCall,
            {"config": state.config, "metric_results": cached_metric_results},
        )


def _resolve_task(task: str | EvaluationTask) -> EvaluationTask:
    if isinstance(task, str):
        return get_task(task)
    if not isinstance(task, EvaluationTask):
        raise TypeError(f"task must be a task name or a registered EvaluationTask, got {type(task).__name__}.")
    # The cache stores entries by task name, so the name must lead back to this
    # task or one equal to it, as register_task() treats equal tasks as one.
    try:
        registered = get_task(task.name)
    except KeyError:
        registered = None
    if registered is None or registered != task:
        raise ValueError(
            f"Evaluation task {task.name!r} is not registered. Register it with register_task() before passing it."
        )
    return registered


def _validate_evaluate_request(
    *,
    return_augmented_data: bool,
    cpu_prediction_postprocessor: Callable[[Sequence[Sequence[Any]]], Sequence[Sequence[Any]]] | None,
    cpu_prediction_postprocessor_id: str | None,
    strict_cache_serialization: bool,
) -> None:
    if not isinstance(return_augmented_data, bool):
        raise TypeError("return_augmented_data must be a bool, matching maite.tasks.evaluate().")
    _validate_serialization_policy(strict_cache_serialization)
    if cpu_prediction_postprocessor is None and cpu_prediction_postprocessor_id is not None:
        raise ValueError("cpu_prediction_postprocessor_id was provided without a cpu_prediction_postprocessor.")


def _evaluate_cached_predictions(
    *,
    prediction_call: _PredictCall,
    metric: gen.Metric[T_Target, T_Metadata] | None,
    evaluation_cache: _EvaluationCacheState,
    cpu_prediction_postprocessor: Callable[[Sequence[Sequence[T_Target]]], Sequence[Sequence[T_Target]]] | None,
    return_preds: bool,
) -> tuple[
    Mapping[str, Any],
    Sequence[Sequence[T_Target]],
    list[Any],
]:
    raw_predictions = cast(Sequence[Sequence[T_Target]], prediction_call.predictions)
    inference_context = cast(
        Sequence[tuple[Sequence[Any], Sequence[T_Target], Sequence[T_Metadata]]],
        prediction_call.augmented_data,
    )
    processed_predictions = (
        cpu_prediction_postprocessor(raw_predictions) if cpu_prediction_postprocessor is not None else raw_predictions
    )

    if metric is None:
        metric_results: Mapping[str, Any] = {}
    elif evaluation_cache.call is not None:
        metric_results = evaluation_cache.call.metric_results
    else:
        metric_results = maite.tasks.evaluate_from_predictions(
            metric=metric,
            pred_batches=processed_predictions,
            target_batches=[batch[1] for batch in inference_context],
            metadata_batches=[batch[2] for batch in inference_context],
        )
        _store_evaluation_result(evaluation_cache, metric_results)

    return metric_results, processed_predictions if return_preds else [], []


def _admit_prediction_artifact(
    *,
    enabled: bool,
    predictions: Sequence[Sequence[Any]],
    inference_context: Sequence[tuple[Sequence[Any], Sequence[Any], Sequence[Any]]],
    strict: bool,
) -> bool:
    if not enabled:
        return False
    try:
        _validate_prediction_artifact(predictions, inference_context, strict=strict)
        return True
    except Exception as error:  # noqa: BLE001 - caching is optional after completed work
        warn_optional_cache_failure(f"Cache publication is disabled for this value: {error}", stacklevel=3)
        return False


def _publish_predictions(
    key: str,
    config: _PredictConfig,
    task: EvaluationTask,
    predictions: Sequence[Sequence[Any]],
    inference_context: Sequence[tuple[Sequence[Any], Sequence[Any], Sequence[Any]]],
) -> bool:
    if config.strict_cache_serialization:
        try:
            _require_lossless_task_conversion(task, predictions, inference_context)
        except Exception as error:  # noqa: BLE001 - caching is optional after completed work
            warn_optional_cache_failure(f"Cache publication is disabled for this value: {error}", stacklevel=3)
            return False
    return _try_publish(
        _predict_cache,
        key,
        _PredictCall,
        {
            "config": config,
            "task": task.name,
            "predictions": predictions,
            "augmented_data": inference_context,
        },
    )


def _run_fresh_evaluation(
    *,
    model: gen.Model[T_Input, T_Target],
    metric: gen.Metric[T_Target, T_Metadata] | None,
    dataloader: gen.DataLoader[T_Input, T_Target, T_Metadata] | None,
    dataset: gen.Dataset[T_Input, T_Target, T_Metadata] | None,
    batch_size: int,
    augmentation: gen.Augmentation[T_Input, T_Target, T_Metadata, T_Input, T_Target, T_Metadata] | None,
    return_augmented_data: bool,
    return_preds: bool,
    cpu_prediction_postprocessor: Callable[[Sequence[Sequence[T_Target]]], Sequence[Sequence[T_Target]]] | None,
    prediction_cache_write_enabled: bool,
    prediction_key: str | None,
    prediction_config: _PredictConfig,
    evaluation_cache: _EvaluationCacheState,
    task: EvaluationTask,
) -> tuple[
    Mapping[str, Any],
    Sequence[Sequence[T_Target]],
    Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]],
]:
    needs_deferred_scoring = metric is not None and cpu_prediction_postprocessor is not None
    collect_target_batches = needs_deferred_scoring or prediction_cache_write_enabled
    evaluate_online = metric is not None and cpu_prediction_postprocessor is None and evaluation_cache.call is None
    collector = _CollectingMetric(
        metric if evaluate_online else None,
        task=task,
        collect_batches=collect_target_batches,
        validate_batches=metric is not None or collect_target_batches,
        require_batches=metric is not None,
    )
    need_predictions = (
        return_preds or prediction_cache_write_enabled or (needs_deferred_scoring and evaluation_cache.call is None)
    )
    effective_augmentation = _materializing_augmentation(augmentation, task) if return_augmented_data else augmentation
    online_metric_results, raw_predictions, maite_augmented_data = maite.tasks.evaluate(
        model=model,
        metric=collector,
        dataloader=dataloader,
        dataset=dataset,
        batch_size=batch_size,
        augmentation=effective_augmentation,
        return_augmented_data=return_augmented_data,
        return_preds=need_predictions,
    )

    inference_context: Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]]
    if return_augmented_data:
        inference_context = [
            (cast(Sequence[T_Input], []), targets, metadata) for _, targets, metadata in maite_augmented_data
        ]
        returned_augmented_data = maite_augmented_data
    elif collect_target_batches:
        inference_context = [
            (cast(Sequence[T_Input], []), targets, metadata)
            for targets, metadata in zip(collector.target_batches, collector.metadata_batches, strict=True)
        ]
        returned_augmented_data = []
    else:
        inference_context = []
        returned_augmented_data = []

    prediction_cache_publish_enabled = _admit_prediction_artifact(
        enabled=prediction_cache_write_enabled,
        predictions=raw_predictions,
        inference_context=inference_context,
        strict=prediction_config.strict_cache_serialization,
    )
    if prediction_cache_publish_enabled and prediction_key is not None:
        prediction_cache_publish_enabled = _publish_predictions(
            prediction_key, prediction_config, task, raw_predictions, inference_context
        )

    processed_predictions = (
        cpu_prediction_postprocessor(raw_predictions) if cpu_prediction_postprocessor is not None else raw_predictions
    )
    if metric is None:
        metric_results: Mapping[str, Any] = {}
    elif evaluation_cache.call is not None:
        metric_results = evaluation_cache.call.metric_results
    elif evaluate_online:
        metric_results = online_metric_results
    else:
        metric_results = maite.tasks.evaluate_from_predictions(
            metric=metric,
            pred_batches=processed_predictions,
            target_batches=[batch[1] for batch in inference_context],
            metadata_batches=[batch[2] for batch in inference_context],
        )

    if (
        metric is not None
        and evaluation_cache.call is None
        and (not prediction_cache_write_enabled or prediction_cache_publish_enabled)
    ):
        _store_evaluation_result(evaluation_cache, metric_results)

    return metric_results, processed_predictions if return_preds else [], returned_augmented_data


def evaluate(
    *,
    model: gen.Model[T_Input, T_Target],
    metric: gen.Metric[T_Target, T_Metadata] | None = None,
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
    cpu_prediction_postprocessor: Callable[[Sequence[Sequence[T_Target]]], Sequence[Sequence[T_Target]]] | None = None,
    cpu_prediction_postprocessor_id: str | None = None,
    use_cache: bool = True,
    strict_cache_serialization: bool = False,
    task: str | EvaluationTask[T_Target],
) -> tuple[
    Mapping[str, Any],
    Sequence[Sequence[T_Target]],
    Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]],
]:
    """Evaluate a model through MAITE's fundamental task with cache-aware reuse.

    Cold evaluations run the metric in MAITE's inference loop. Cached
    predictions are evaluated with :func:`evaluate_from_predictions`, while
    :func:`predict` delegates here with ``metric=None``.

    Setting ``return_augmented_data=True`` always runs fresh inference so the
    returned inputs and predictions share one realization. Complete inputs are
    materialized for the response but are not cached; one-shot video streams may
    therefore require substantial memory. ``strict_cache_serialization``
    applies one policy to every persisted task artifact and is included in cache
    identity, preventing strict calls from reading compatibility-mode entries.

    ``task`` is required. It names the evaluation task, or alias, of the model
    and dataset (see :mod:`checkmaite.core.evaluation_tasks`), or is the
    registered :class:`EvaluationTask` itself. An unknown name raises
    ``KeyError`` and an unregistered task ``ValueError``. It decides how
    predictions and targets are cached and whether one-shot input streams are
    materialized, and it is part of the cache key. If the first batch of
    predictions or targets does not match the task, ``ValueError`` is raised.
    """
    resolved_task = _resolve_task(task)
    if dataloader is not None and use_cache:
        warnings.warn(
            "Caching is disabled for dataloader calls because their batching has no stable identity.",
            stacklevel=2,
        )
        use_cache = False

    model_id = _resolve_id(model)
    resolved_dataset_id = _resolve_id(dataset, id=dataset_id)
    augmentation_id = _resolve_id(augmentation) if augmentation is not None else None
    if augmentation is not None and augmentation_id is None and use_cache:
        warnings.warn(
            "Caching is disabled because the supplied augmentation has no metadata ID.",
            stacklevel=2,
        )
        use_cache = False
    _validate_evaluate_request(
        return_augmented_data=return_augmented_data,
        cpu_prediction_postprocessor=cpu_prediction_postprocessor,
        cpu_prediction_postprocessor_id=cpu_prediction_postprocessor_id,
        strict_cache_serialization=strict_cache_serialization,
    )
    prediction_config = _PredictConfig(
        model_id=model_id,
        dataset_id=resolved_dataset_id,
        augmentation_id=augmentation_id,
        task=resolved_task.name,
        batch_size=batch_size,
        strict_cache_serialization=strict_cache_serialization,
    )
    prediction_key = prediction_config.cache_key
    if prediction_key is None and use_cache:
        warnings.warn(
            "use_cache was requested but caching is disabled because at least one of the following is None:"
            + str(prediction_config.cache_key_required_fields()),
            stacklevel=2,
        )
    prediction_cache_enabled = prediction_key is not None and use_cache
    evaluation_cache = _prepare_evaluation_cache(
        metric=metric,
        inference_id=prediction_key,
        task=prediction_config.task,
        model_id=model_id,
        dataset_id=resolved_dataset_id,
        augmentation_id=augmentation_id,
        batch_size=batch_size,
        cpu_prediction_postprocessor=cpu_prediction_postprocessor,
        cpu_prediction_postprocessor_id=cpu_prediction_postprocessor_id,
        return_augmented_data=return_augmented_data,
        use_cache=use_cache and not return_augmented_data,
        strict_cache_serialization=strict_cache_serialization,
    )

    if evaluation_cache.call is not None and not return_preds and not return_augmented_data:
        return evaluation_cache.call.metric_results, [], []

    prediction_call = (
        _predict_cache.get(prediction_key)
        if prediction_cache_enabled and not return_augmented_data and prediction_key is not None
        else None
    )
    if prediction_call is not None:
        return _evaluate_cached_predictions(
            prediction_call=prediction_call,
            metric=metric,
            evaluation_cache=evaluation_cache,
            cpu_prediction_postprocessor=cpu_prediction_postprocessor,
            return_preds=return_preds,
        )
    if evaluation_cache.call is not None:
        evaluation_cache = replace(evaluation_cache, call=None)

    return _run_fresh_evaluation(
        model=model,
        metric=metric,
        dataloader=dataloader,
        dataset=dataset,
        batch_size=batch_size,
        augmentation=augmentation,
        return_augmented_data=return_augmented_data,
        return_preds=return_preds,
        cpu_prediction_postprocessor=cpu_prediction_postprocessor,
        prediction_cache_write_enabled=prediction_cache_enabled and not return_augmented_data,
        prediction_key=prediction_key,
        prediction_config=prediction_config,
        evaluation_cache=evaluation_cache,
        task=resolved_task,
    )


class TaskCachedTasks(Generic[T_Target]):
    """:func:`evaluate` and :func:`predict` with one evaluation task bound.

    Their signatures match the module-level functions without ``task``, so
    they keep the shape of ``maite.tasks.evaluate`` and ``maite.tasks.predict``.
    Create one with :func:`for_task`. The task is looked up on every call, so it
    only has to be registered by the time the functions are called.
    """

    def __init__(self, task: EvaluationTask[T_Target]) -> None:
        if not isinstance(task, EvaluationTask):
            raise TypeError(f"Expected an EvaluationTask, got {type(task).__name__}.")
        self.task = task

    def evaluate(
        self,
        *,
        model: gen.Model[T_Input, T_Target],
        metric: gen.Metric[T_Target, T_Metadata] | None = None,
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
        cpu_prediction_postprocessor: Callable[[Sequence[Sequence[T_Target]]], Sequence[Sequence[T_Target]]]
        | None = None,
        cpu_prediction_postprocessor_id: str | None = None,
        use_cache: bool = True,
        strict_cache_serialization: bool = False,
    ) -> tuple[
        Mapping[str, Any],
        Sequence[Sequence[T_Target]],
        Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]],
    ]:
        """Run :func:`checkmaite.core.cached_tasks.evaluate` for the bound task."""
        return evaluate(
            model=model,
            metric=metric,
            dataloader=dataloader,
            dataset=dataset,
            batch_size=batch_size,
            augmentation=augmentation,
            return_augmented_data=return_augmented_data,
            return_preds=return_preds,
            dataset_id=dataset_id,
            cpu_prediction_postprocessor=cpu_prediction_postprocessor,
            cpu_prediction_postprocessor_id=cpu_prediction_postprocessor_id,
            use_cache=use_cache,
            strict_cache_serialization=strict_cache_serialization,
            task=self.task,
        )

    def predict(
        self,
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
        strict_cache_serialization: bool = False,
    ) -> tuple[
        Sequence[Sequence[T_Target]],
        Sequence[tuple[Sequence[T_Input], Sequence[T_Target], Sequence[T_Metadata]]],
    ]:
        """Run :func:`checkmaite.core.cached_tasks.predict` for the bound task."""
        return predict(
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            batch_size=batch_size,
            augmentation=augmentation,
            return_augmented_data=return_augmented_data,
            dataset_id=dataset_id,
            use_cache=use_cache,
            strict_cache_serialization=strict_cache_serialization,
            task=self.task,
        )


def for_task(task: EvaluationTask[T_Target]) -> TaskCachedTasks[T_Target]:
    """Return :func:`evaluate` and :func:`predict` with ``task`` bound.

    A package that adds an evaluation task exports the bound functions so its
    users call them like ``maite.tasks.evaluate``, without naming the task::

        _cached = cached_tasks.for_task(TASK)
        evaluate = _cached.evaluate
        predict = _cached.predict
        evaluate_from_predictions = cached_tasks.evaluate_from_predictions

    The built-in tasks do the same in ``checkmaite.core.<task>.cached_tasks``.
    """
    return TaskCachedTasks(task)
