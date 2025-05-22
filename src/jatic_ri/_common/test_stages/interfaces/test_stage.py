"""Base Test Stage for all test implementations"""

import enum
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
)

import jatic_ri
from jatic_ri import cache_path
from jatic_ri._common.test_stages.interfaces.plugins import (
    MetricPlugin,
    MultiModelPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    TwoDatasetPlugin,
)
from jatic_ri.util._cachev2 import PydanticCache, binary_de_serializer


class ConfigBase(BaseModel):
    pass


TConfig = TypeVar("TConfig", bound=ConfigBase)


class OutputsBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def _traverse(cls, obj: Any, fn: Callable[[Any], Any]) -> Any:
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            # named tuple
            return type(obj)(*cls._traverse(tuple(obj), fn))
        if isinstance(obj, (list, tuple)):
            return type(obj)([cls._traverse(i, fn) for i in obj])
        if isinstance(obj, dict):
            return type(obj)({k: cls._traverse(v, fn) for k, v in obj.items()})
        return fn(obj)

    @field_serializer("*")
    def __serialize_binary(self, v: Any) -> Any:
        return self._traverse(v, binary_de_serializer.serialize)

    @field_validator("*", mode="before")
    @classmethod
    def __deserialize_binary(cls, v: Any) -> Any:
        return cls._traverse(v, binary_de_serializer.deserialize)


# FIXME: add bound=BaseModel as soon as all test stages are updated
TOutputs = TypeVar("TOutputs")


class RunBase(BaseModel, Generic[TConfig, TOutputs]):
    """
    Abstract base class representing the results of an immutable execution run.

    This class encapsulates the configuration, inputs (datasets, models),
    metric used, and the specific outputs generated during a run, typically
    by a JATIC tool.

    Attributes:
        test_stage_id: Unique identifier of the test stage that produced this run
        config: The configuration object used for this run.
        dataset_ids: Sequence of unique identifiers for the datasets used.
        model_ids: Sequence of unique identifiers for the models used.
        metric_id: Identifier for the primary metric evaluated or generated.
        outputs: The specific results produced by the run, with type defined
                 by the generic parameter `TOutputs`.
    """

    model_config = ConfigDict(frozen=True)

    test_stage_id: str
    config: TConfig
    dataset_ids: Sequence[str]
    model_ids: Sequence[str]
    metric_id: str
    outputs: TOutputs

    @staticmethod
    def compute_uid(
        test_stage_id: str,
        config: TConfig,
        dataset_ids: Sequence[str],
        model_ids: Sequence[str],
        metric_id: str,
    ) -> str:
        """
        Computes a unique identifier (SHA-256 hash) for a run configuration.

        The UID is deterministically generated based on the provided configuration
        object, dataset identifiers, model identifiers, and metric identifier.

        Args:
            test_stage_id: Unique identifier of the test stage
            config: The configuration object.
            dataset_ids: A sequence of dataset unique identifiers.
            model_ids: A sequence of model unique identifiers.
            metric_id: The unique identifier for the metric used.

        Returns:
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


# TODO: As the name implies, this class is only used for BC until all test stages are updated. Remove afterwards.
class CompatConfig(ConfigBase):
    pass


# TODO: As the name implies, this class is only used for BC until all test stages are updated. Remove afterwards.
class CompatRun(RunBase):
    @classmethod
    def _traverse(cls, obj: Any, fn: Callable[[Any], Any]) -> Any:
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return type(obj)(*cls._traverse(tuple(obj), fn))
        if isinstance(obj, (list, tuple)):
            return type(obj)([cls._traverse(i, fn) for i in obj])
        if isinstance(obj, dict):
            return type(obj)({k: cls._traverse(v, fn) for k, v in obj.items()})
        return fn(obj)

    @field_serializer("outputs", check_fields=False)
    def _serialize_outputs(self, v: Any) -> Any:
        return self._traverse(v, binary_de_serializer.serialize)

    @field_validator("outputs", mode="before", check_fields=False)
    @classmethod
    def _deserialize_outputs(cls, v: Any) -> Any:
        return cls._traverse(v, binary_de_serializer.deserialize)


# TODO: this class needs to be removed as soon as the
class Cache(Generic[TOutputs]):
    """Caching mechanism for test stages"""

    def read_cache(self, cache_path: str) -> TOutputs | None: ...
    def write_cache(self, cache_path: str, data: TOutputs) -> None: ...


# TODO: this is only used validating the plugins and should be removed as soon as the plugins are gone
class RIValidationError(Exception):
    """Exception raised for validation errors."""

    def __init__(self, message: str = "Validation error occurred.") -> None:
        self.message = message
        super().__init__(self.message)


class Number(enum.Enum):
    """
    Enumeration to specify the cardinality (number of supported items)
    for models, datasets, or metrics within a tool.
    """

    ZERO = enum.auto()
    ONE = enum.auto()
    TWO = enum.auto()
    MANY = enum.auto()


class TestStage(Generic[TOutputs], ABC):
    """Base class for running a test and recieving report values"""

    # TODO: remove the default value after &22
    _RUN_TYPE: ClassVar[type[RunBase]] = CompatRun

    _deck: str
    _task: str
    # TODO: Remove _outputs
    _outputs: TOutputs | None = None  # test results are expected to be stored within the test stage
    _batch_size: int = 1  # Not fully implemented yet - Ref Issue 270 "Expose batch size in test stages"

    # TODO: Remove this after all test stages have been updated
    cache: Cache[TOutputs] | None = None

    def __init__(self) -> None:
        # TODO: remove this as soon as collect_report_consumables has been moved to the respective Run object
        self._stored_run: RunBase | None = None

    @property
    def supports_datasets(self) -> Number:
        """Indicates the number of datasets this plugin supports."""
        if isinstance(self, TwoDatasetPlugin):
            return Number.TWO
        if isinstance(self, SingleDatasetPlugin):
            return Number.ONE
        return Number.ZERO

    @property
    def supports_models(self) -> Number:
        """Indicates the number of models this plugin supports."""
        if isinstance(self, MultiModelPlugin):
            return Number.MANY
        if isinstance(self, SingleModelPlugin):
            return Number.ONE
        return Number.ZERO

    @property
    def supports_metric(self) -> Number:
        """Indicates whether this plugin supports a metric."""
        if isinstance(self, MetricPlugin):
            return Number.ONE
        return Number.ZERO

    # TODO: Both getter and setter are only here for BC. Remove as soon as all test stages are updated
    @property
    def outputs(self) -> TOutputs:
        """Property getter for TestStage run outputs - raises RunTimeError if accessed before set"""
        if self._stored_run is None:
            raise RuntimeError("TestStage must be run before accessing outputs")
        return self._stored_run.outputs

    @outputs.setter
    def outputs(self, value: TOutputs) -> None:
        """Property setter for TestStage run outputs"""
        dataset_ids, model_ids, metric_id = self._extract_run_inputs()
        self._stored_run = CompatRun(
            test_stage_id=self.id,
            config=self._create_config(),
            dataset_ids=dataset_ids,
            model_ids=model_ids,
            metric_id=metric_id,
            outputs=value,
        )

    # TODO: remove after Run implemented in all test stages
    @property
    def cache_id(self) -> str:
        """Override this with a unique cache id to save outputs to cache"""
        return ""

    # TODO: remove after Run implemented in all test stages
    @property
    def cache_path(self) -> str:
        return str(jatic_ri.cache_path())

    # TODO: remove as soon as the plugins are removed
    def validate_plugins(self) -> None:
        plugin_requirements = {
            "SingleModelPlugin": (["model", "model_id"], "load_model"),
            "MultiModelPlugin": (["models"], "load_models"),
            "SingleDatasetPlugin": (["dataset", "dataset_id"], "load_dataset"),
            "TwoDatasetPlugin": (["dataset_1", "dataset_1_id", "dataset_2", "dataset_2_id"], "load_datasets"),
            "MetricPlugin": (["metric", "metric_id"], "load_metric"),
            "ThresholdPlugin": (["threshold"], "load_threshold"),
            "EvalToolPlugin": (["eval_tool"], "load_eval_tool"),
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

    # TODO: this should become a class variable like _RUN_TYPE after &22 is completed
    def _create_config(self) -> ConfigBase:
        return CompatConfig()

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
        """Run the test stage leveraging cache if available and store any outputs of the evaluation in test stage"""

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
    # TODO: use pydantic.BaseModel as return annotation after all test stages are upgraded
    @abstractmethod
    def _run(self) -> TOutputs:
        """Override this with logic to execute test stage and return outputs"""

    @abstractmethod
    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report generated in the run method or in the
        load_cached_results method

        Please return a list of dictionaries, one dictionary per slide

        For each dictionary, please include the following keys:
        - "deck": (str) image_classification_model_evaluation, object_detection_model_evaluation,
          object_detection_dataset_evaluation
        - "layout_name": (str) find the layout name in the jatic_increment_5_gradient_demo_repo, linked below
        https://gitlab.jatic.net/jatic/morse/jatic-increment-5-gradient-demo-repo/-/tree/main/src/jatic_increment_5_gradient_demo_repo/cards?ref_type=heads
        - "layout_arguments": (dict) arguments pertaining to the specific layout

        For example:
        # I have one slide, meant for the object detection dataset evaluation report
        [
            {"deck": "object_detection_dataset_evaluation",
            "layout_name": "OneImageText",
            "layout_arguments": {
                "title": "This is my cool title",
                "text": "This is my cool text",
                "image_path": Path("path/to/my/image")
                }
            }
        ]
        """
