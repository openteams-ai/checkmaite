"""Base Test Stage for all test implementations"""

import dataclasses
import enum
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
)

from jatic_ri import cache_path
from jatic_ri._common.test_stages.interfaces.plugins import (
    MetricPlugin,
    MultiModelPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    TwoDatasetPlugin,
)
from jatic_ri.util._cache import PydanticCache, binary_de_serializer


class ConfigBase(BaseModel):
    pass


TConfig = TypeVar("TConfig", bound=ConfigBase)


class OutputsBase(BaseModel):
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


TOutputs = TypeVar("TOutputs", bound=BaseModel)


class RunBase(BaseModel, Generic[TConfig, TOutputs]):
    """Abstract base class representing the results of an immutable execution run.

    This class encapsulates the configuration, inputs (datasets, models),
    metric used, and the specific outputs generated during a run, typically
    by a JATIC tool.

    Parameters
    ----------
    test_stage_id : str
        Unique identifier of the test stage that produced this run.
    config : TConfig
        The configuration object used for this run.
    dataset_ids : list[str]
        Sequence of unique identifiers for the datasets used.
    model_ids : list[str]
        Sequence of unique identifiers for the models used.
    metric_id : str
        Identifier for the primary metric evaluated or generated.
    outputs : TOutputs
        The specific results produced by the run, with type defined
        by the generic parameter `TOutputs`.

    """

    model_config = ConfigDict(frozen=True)

    test_stage_id: str
    config: TConfig
    dataset_ids: list[str]
    model_ids: list[str]
    metric_id: str
    outputs: TOutputs

    @staticmethod
    def compute_uid(
        test_stage_id: str,
        config: TConfig,
        dataset_ids: list[str],
        model_ids: list[str],
        metric_id: str,
    ) -> str:
        """Compute a unique identifier (SHA-256 hash) for a run configuration.

        The UID is deterministically generated based on the provided configuration
        object, dataset identifiers, model identifiers, and metric identifier.

        Parameters
        ----------
        test_stage_id : str
            Unique identifier of the test stage.
        config : TConfig
            The configuration object.
        dataset_ids : list[str]
            A sequence of dataset unique identifiers.
        model_ids : list[str]
            A sequence of model unique identifiers.
        metric_id : str
            The unique identifier for the metric used.

        Returns
        -------
        str
            A string representing the hexadecimal SHA-256 hash of the inputs.
        """
        uid_content = {
            "test_stage_id": test_stage_id,
            "config": config.model_dump(mode="json"),
            "dataset_ids": dataset_ids,
            "model_ids": model_ids,
            "metric_id": metric_id,
        }

        return hashlib.sha256(json.dumps(uid_content).encode("utf-8")).hexdigest()


class RunCache(PydanticCache[RunBase]):
    def path(self, key: str) -> Path:
        d = cache_path() / "runs"
        d.mkdir(parents=True, exist_ok=True)
        return d / key


run_cache = RunCache()


# TODO: this is only used validating the plugins and should be removed as soon as the plugins are gone
class RIValidationError(Exception):
    """Exception raised for validation errors.

    Parameters
    ----------
    message : str, optional
        Description of the error. Default is "Validation error occurred.".

    Attributes
    ----------
    message : str
        Description of the error.
    """

    def __init__(self, message: str = "Validation error occurred.") -> None:
        self.message = message
        super().__init__(self.message)


class Number(enum.Enum):
    """Enumeration for cardinality of models, datasets, or metrics.

    Specifies the number of supported items (e.g., models, datasets, metrics)
    for a tool or plugin.
    """

    ZERO = enum.auto()
    ONE = enum.auto()
    TWO = enum.auto()
    MANY = enum.auto()


class TestStage(Generic[TOutputs], ABC):
    """Base class for running a test and receiving report values.

    Attributes
    ----------
    _RUN_TYPE : type[RunBase]
        The type of the run object associated with this test stage.
    _deck : str
        Identifier for the report deck.
    _task : str
        Identifier for the task type.
    _batch_size : int
        Batch size for processing data. Default is 1.
        (Note: Not fully implemented yet - Ref Issue 270 "Expose batch size in test stages")
    """

    _RUN_TYPE: ClassVar[type[RunBase]]

    _deck: str
    _task: str
    _batch_size: int = 1  # Not fully implemented yet - Ref Issue 270 "Expose batch size in test stages"

    def __init__(self) -> None:
        # TODO: remove this as soon as collect_report_consumables has been moved to the respective Run object
        self._stored_run: RunBase | None = None

    @property
    def supports_datasets(self) -> Number:
        """Number of datasets this plugin supports.

        Returns
        -------
        Number
            An enumeration value (ZERO, ONE, TWO) indicating dataset support.
        """
        if isinstance(self, TwoDatasetPlugin):
            return Number.TWO
        if isinstance(self, SingleDatasetPlugin):
            return Number.ONE
        return Number.ZERO

    @property
    def supports_models(self) -> Number:
        """Number of models this plugin supports.

        Returns
        -------
        Number
            An enumeration value (ZERO, ONE, MANY) indicating model support.
        """
        if isinstance(self, MultiModelPlugin):
            return Number.MANY
        if isinstance(self, SingleModelPlugin):
            return Number.ONE
        return Number.ZERO

    @property
    def supports_metric(self) -> Number:
        """Whether this plugin supports a metric.

        Returns
        -------
        Number
            An enumeration value (ZERO, ONE) indicating metric support.
        """
        if isinstance(self, MetricPlugin):
            return Number.ONE
        return Number.ZERO

    # TODO: remove as soon as the plugins are removed
    def validate_plugins(self) -> None:
        plugin_requirements = {
            "SingleModelPlugin": (["model", "model_id"], "load_model"),
            "MultiModelPlugin": (["models"], "load_models"),
            "SingleDatasetPlugin": (["dataset", "dataset_id"], "load_dataset"),
            "TwoDatasetPlugin": (["dataset_1", "dataset_1_id", "dataset_2", "dataset_2_id"], "load_datasets"),
            "MetricPlugin": (["metric", "metric_id"], "load_metric"),
            "ThresholdPlugin": (["threshold"], "load_threshold"),
        }

        # Identify plugins in the inheritance hierarchy
        plugin_list = plugin_requirements.keys()
        inherited_classes = [cls for cls in self.__class__.mro() if cls.__name__ in plugin_list]

        # Iterate through each valid plugin inheritted
        for plugin in inherited_classes:
            plugin_name = plugin.__name__
            required_attrs, load_func = plugin_requirements[plugin_name]

            # Check for missing attributes and raise appropriate error
            missing_attr = next((attr for attr in required_attrs if not hasattr(self, attr)), None)
            if missing_attr:
                raise RIValidationError(
                    f"'{missing_attr}' not set! Please use `{load_func}()` function to set the '{missing_attr}'.",
                )

    @abstractmethod
    def _create_config(self) -> ConfigBase: ...

    # TODO: this is temporary compatibility code. Remove when datasets etc. can be passed to run()
    def _extract_run_inputs(self) -> tuple[list[str], list[str], str]:
        self.validate_plugins()

        dataset_ids: list[str]
        if isinstance(self, SingleDatasetPlugin):
            dataset_ids = [self.dataset_id]
        elif isinstance(self, TwoDatasetPlugin):
            dataset_ids = [self.dataset_1_id, self.dataset_2_id]
        else:
            dataset_ids = []

        model_ids: list[str]
        if isinstance(self, SingleModelPlugin):
            model_ids = [self.model_id]
        elif isinstance(self, MultiModelPlugin):
            model_ids = list(self.models.keys())
        else:
            model_ids = []

        metric_id = self.metric_id if isinstance(self, MetricPlugin) else ""

        return dataset_ids, model_ids, metric_id

    @cached_property
    def id(self) -> str:
        return f"{type(self).__module__}.{type(self).__name__}"

    def run(self, use_stage_cache: bool = True) -> RunBase:
        """Run the test stage.

        Leverages cache if available and stores outputs of the evaluation
        in the test stage.

        Parameters
        ----------
        use_stage_cache : bool, optional
            Whether to use cached results if available, by default True.

        Returns
        -------
        RunBase
            The results of the test stage execution.
        """
        config = self._create_config()
        dataset_ids, model_ids, metric_id = self._extract_run_inputs()

        uid = self._RUN_TYPE.compute_uid(
            test_stage_id=self.id, config=config, dataset_ids=dataset_ids, model_ids=model_ids, metric_id=metric_id
        )

        if use_stage_cache:
            run = self._stored_run = run_cache.get(uid)
            if run is not None:
                return run

        run = self._stored_run = self._RUN_TYPE(
            test_stage_id=self.id,
            config=config,
            dataset_ids=dataset_ids,
            model_ids=model_ids,
            metric_id=metric_id,
            outputs=self._run(),
        )

        if use_stage_cache:
            run_cache.set(uid, run)

        return run

    # TODO: update signature to accept config, models, datasets and metrics
    @abstractmethod
    def _run(self) -> TOutputs:
        """Execute the core logic of the test stage.

        This method should be overridden by subclasses to implement the
        specific test stage functionality.

        Returns
        -------
        TOutputs
            The outputs generated by the test stage.
        """

    @abstractmethod
    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect data for generating a report.

        Accesses in-depth data needed by Gradient to produce a report. This data
        is typically generated in the `_run` method or loaded from cached results.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, where each dictionary represents a slide
            in the report. Each dictionary should contain the following keys:

            - "deck" (str): Identifier for the report deck (e.g.,
              "image_classification_model_evaluation",
              "object_detection_model_evaluation",
              "object_detection_dataset_evaluation").
            - "layout_name" (str): Name of the layout for the slide. Layout names
              can be found in the jatic_increment_5_gradient_demo_repo:
              https://gitlab.jatic.net/jatic/morse/jatic-increment-5-gradient-demo-repo/-/tree/main/src/jatic_increment_5_gradient_demo_repo/cards?ref_type=heads
            - "layout_arguments" (dict): Arguments specific to the chosen layout.

        Examples
        --------
        Example for one slide in an object detection dataset evaluation report:

        >>> def collect_report_consumables(self):
        ...     return [
        ...         {
        ...             "deck": "object_detection_dataset_evaluation",
        ...             "layout_name": "OneImageText",
        ...             "layout_arguments": {
        ...                 "title": "This is my cool title",
        ...                 "text": "This is my cool text",
        ...                 "image_path": Path("path/to/my/image"),
        ...             },
        ...         }
        ...     ]
        """
