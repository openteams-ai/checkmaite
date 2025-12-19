import dataclasses
import enum
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

import maite.protocols.generic as gen
from maite.protocols import DatasetMetadata, MetricMetadata, ModelMetadata
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
)

from jatic_ri import cache_path
from jatic_ri.core._cache import PydanticCache, binary_de_serializer

TModel = TypeVar("TModel", bound=gen.Model[Any, Any])
TDataset = TypeVar("TDataset", bound=gen.Dataset[Any, Any, Any])
TMetric = TypeVar("TMetric", bound=gen.Metric[Any])

TConfig = TypeVar("TConfig", bound="CapabilityConfigBase")
TOutputs = TypeVar("TOutputs", bound=BaseModel)


class CapabilityConfigBase(BaseModel):
    pass


class CapabilityOutputsBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def _traverse(cls, obj: Any, fn: Callable[[Any], Any]) -> Any:
        if not isinstance(obj, type) and dataclasses.is_dataclass(obj):
            return type(obj)(**cls._traverse(dataclasses.asdict(obj), fn))  # pyright: ignore[reportArgumentType]
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            # named tuple
            return type(obj)(*cls._traverse(tuple(obj), fn))
        if isinstance(obj, (list, tuple)):
            return type(obj)([cls._traverse(i, fn) for i in obj])
        if isinstance(obj, dict):
            return type(obj)({k: cls._traverse(v, fn) for k, v in obj.items()})
        return fn(obj)

    @field_serializer("*", when_used="json-unless-none")
    def __serialize_binary(self, v: Any) -> Any:
        return self._traverse(v, binary_de_serializer.serialize)

    @field_validator("*", mode="before")
    @classmethod
    def __deserialize_binary(cls, v: Any) -> Any:
        return cls._traverse(v, binary_de_serializer.deserialize)


class CapabilityRunBase(BaseModel, Generic[TConfig, TOutputs]):
    """Abstract base class representing the results of an immutable capability run.

    This class encapsulates the configuration, inputs (datasets, models),
    metric used, and the specific outputs generated during a capability run.

    Parameters
    ----------
    capability_id
        Unique identifier of the capability that produced this run.
    config
        The configuration object used for this run.
    dataset_metadata
        Sequence of metadata associated with MAITE dataset objects.
    model_metadata
        Sequence of metadata associated with MAITE model objects.
    metric_metadata
        Sequence of metadata associated with MAITE metric objects.
    outputs
        The specific results produced by the run, with type defined
        by the generic parameter `TOutputs`.
    """

    model_config = ConfigDict(frozen=True)

    capability_id: str
    config: TConfig
    dataset_metadata: list[DatasetMetadata]
    model_metadata: list[ModelMetadata]
    metric_metadata: list[MetricMetadata]
    outputs: TOutputs

    @staticmethod
    def compute_uid(
        capability_id: str,
        config: TConfig,
        dataset_metadata: list[DatasetMetadata],
        model_metadata: list[ModelMetadata],
        metric_metadata: list[MetricMetadata],
    ) -> str:
        """Compute a unique identifier (SHA-256 hash) for a run configuration.

        The UID is deterministically generated based on the provided configuration
        object, dataset identifiers, model identifiers, and metric identifier.

        Parameters
        ----------
        capability_id
            Unique identifier of the capability.
        config
            The configuration object.
        dataset_metadata
            Sequence of metadata associated with MAITE dataset objects.
        model_metadata
            Sequence of metadata associated with MAITE model objects.
        metric_metadata
            Sequence of metadata associated with MAITE metric objects.

        Returns
        -------
        str
            A string representing the hexadecimal SHA-256 hash of the inputs.
        """

        # dataset/model/metric uniquely identified by ID,
        # hence not required to build UID from other metadata
        uid_content = {
            "capability_id": capability_id,
            "config": config.model_dump(mode="json"),
            "dataset_ids": [d["id"] for d in dataset_metadata],
            "model_ids": [d["id"] for d in model_metadata],
            "metric_id": [d["id"] for d in metric_metadata],
        }

        return hashlib.sha256(json.dumps(uid_content).encode("utf-8")).hexdigest()

    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:
        """Collect data for generating a report.

        Accesses in-depth data needed by Gradient to produce a report. This data
        is typically generated in the `_run` method or loaded from cached results.

        Parameters
        ----------
        threshold
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
            A list of dictionaries, where each dictionary represents a slide
            in the report. Each dictionary should contain the following keys:
        """
        raise NotImplementedError

    def collect_md_report(self, threshold: float) -> str:
        """Collect data for generating a markdown report.

        Accesses in-depth data needed to produce a markdown report. This data
        is typically generated in the `_run` method or loaded from cached results.

        Parameters
        ----------
        threshold
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.
        Returns
        -------
            A string containing the markdown report content.
        """
        raise NotImplementedError


class CapabilityRunCache(PydanticCache[CapabilityRunBase[Any, Any]]):
    def path(self, key: str) -> Path:
        d = cache_path() / "runs"
        d.mkdir(parents=True, exist_ok=True)
        return d / key


run_cache = CapabilityRunCache()


class Number(enum.Enum):
    """Enumeration for cardinality of models, datasets, or metrics.

    Specifies the number of supported items (e.g., models, datasets, metrics)
    for a capability.
    """

    ZERO = 0
    ONE = 1
    TWO = 2
    MANY = -1  # at least one


def _check_cardinality(owner_id: str, label: str, required: Number, n: int) -> None:
    "Helper function for checking cardinality of capability inputs is correct."

    # MANY just means at least 1
    if required == Number.MANY:
        if n < 1:
            raise TypeError(f"{owner_id} requires at least 1 {label}, but got 0.")
        return

    expected = int(required.value)
    if n != expected:
        s_expected = "" if expected == 1 else "s"
        s_got = "" if n == 1 else "s"
        raise TypeError(f"{owner_id} requires exactly {expected} {label}{s_expected}, but got {n} {label}{s_got}.")


class Capability(Generic[TOutputs, TDataset, TModel, TMetric, TConfig], ABC):
    """Base class for a capability.

    Attributes
    ----------
    _RUN_TYPE : type[CapabilityRunBase]
        The type of the run object associated with this capability.
    """

    # precise generic type is important, pyright doesn't like this
    # and so solution here is to just ignore pyright here and still
    # keep the generic type information. alternative is to drop the
    # ClassVar annotation, but not sure if that will impact our caching mechanism ...
    _RUN_TYPE: ClassVar[type[CapabilityRunBase[TConfig, TOutputs]]]  # type: ignore[reportGeneralTypeIssues]

    @classmethod
    def _create_config(cls) -> TConfig: ...

    @property
    @abstractmethod
    def supports_datasets(self) -> Number:
        """Number of datasets this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating dataset support.
        """

    @property
    @abstractmethod
    def supports_models(self) -> Number:
        """Number of models this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating model support.
        """

    @property
    @abstractmethod
    def supports_metrics(self) -> Number:
        """Number of metrics this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating metric support.
        """

    @cached_property
    def id(self) -> str:
        return f"{type(self).__module__}.{type(self).__name__}"

    def run(
        self,
        models: list[TModel] | None = None,
        datasets: list[TDataset] | None = None,
        metrics: list[TMetric] | None = None,
        config: TConfig | None = None,
        use_cache: bool = True,
    ) -> CapabilityRunBase[TConfig, TOutputs]:
        """Run the capability.

        Leverages cache if available and stores outputs of the evaluation
        in the capability.

        Parameters
        ----------
        models
            MAITE-compliant models.
        datasets
            MAITE-compliant datasets.
        metrics
            MAITE-compliant metrics.
        config
            The capability configuration object used for this run.

        use_cache
            Whether to use cached results if available, by default True.

        Returns
        -------
        CapabilityRunBase
            The results of the capability execution.
        """
        # replace None sentinel with iterable to reduce boilerplate checks in capability implementations
        datasets = datasets if datasets else []
        models = models if models else []
        metrics = metrics if metrics else []
        config = config if config else self._create_config()

        # TODO: insert pydantic config validation check here

        # validation to make sure correct number of models/datasets/metrics passed to capability
        _check_cardinality(owner_id=self.id, label="dataset", required=self.supports_datasets, n=len(datasets))
        _check_cardinality(owner_id=self.id, label="model", required=self.supports_models, n=len(models))
        _check_cardinality(owner_id=self.id, label="metric", required=self.supports_metrics, n=len(metrics))

        dataset_metadata = [d.metadata for d in datasets]
        model_metadata = [m.metadata for m in models]
        metric_metadata = [m.metadata for m in metrics]

        uid = self._RUN_TYPE.compute_uid(
            capability_id=self.id,
            config=config,
            dataset_metadata=dataset_metadata,
            model_metadata=model_metadata,
            metric_metadata=metric_metadata,
        )

        if use_cache:
            run = run_cache.get(uid)
            if run is not None:
                return run

        # TODO: investigate whether we wish to allow separate toggles
        # for prediction/evaluation and capability caches.
        run = self._RUN_TYPE(
            capability_id=self.id,
            config=config,
            dataset_metadata=dataset_metadata,
            model_metadata=model_metadata,
            metric_metadata=metric_metadata,
            outputs=self._run(
                models=models,
                datasets=datasets,
                metrics=metrics,
                config=config,
                use_prediction_and_evaluation_cache=use_cache,
            ),
        )

        if use_cache:
            run_cache.set(uid, run)

        return run

    @abstractmethod
    def _run(
        self,
        models: list[TModel],
        datasets: list[TDataset],
        metrics: list[TMetric],
        config: TConfig,
        use_prediction_and_evaluation_cache: bool,
    ) -> TOutputs: ...

    """Execute the core logic of the capability.

    Parameters
    ----------
    models
        MAITE-compliant models.
    datasets
        MAITE-compliant datasets.
    metrics
        MAITE-compliant metrics.
    config
        The capability configuration object used for this run.
    use_prediction_and_evaluation_cache
        Whether to use cached model prediction and evaluation results if available.

    Returns
    -------
    TOutputs
        The outputs generated by the capability.
    """
