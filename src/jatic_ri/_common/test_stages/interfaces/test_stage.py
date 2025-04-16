"""Base Test Stage for all test implementations"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, field_serializer, field_validator

from jatic_ri import cache_path

TData = TypeVar("TData")
TConfig = TypeVar("TConfig", bound=BaseModel)


class Run(BaseModel, Generic[TConfig, TData], ABC):
    """
    Abstract base class representing the results of an immutable execution run.

    This class encapsulates the configuration, inputs (datasets, models),
    metric used, and the specific outputs generated during a run, typically
    by a JATIC tool.

    Subclasses must implement the serialization and deserialization logic
    for their specific output types (`TData`).

    Attributes:
        config: The configuration object used for this run.
        dataset_ids: Sequence of unique identifiers for the datasets used.
        model_ids: Sequence of unique identifiers for the models used.
        metric_id: Identifier for the primary metric evaluated or generated.
        outputs: The specific results produced by the run, with type defined
                 by the generic parameter `TData`.
    """

    config: TConfig
    dataset_ids: Sequence[str]
    model_ids: Sequence[str]
    metric_id: str
    outputs: TData

    @field_serializer("outputs")
    @abstractmethod
    def serialize_outputs(self, outputs: TData) -> Any:
        """
        Abstract method to serialize the 'outputs' field for Pydantic.

        This method is automatically called by Pydantic during model serialization
        (e.g., via `.model_dump()`, `.model_dump_json()`). Subclasses MUST
        implement this to convert their specific `outputs` object (of type
        `TData`) into a representation suitable for inclusion in the
        serialized model data (typically JSON-compatible).

        Recommendation for Large/Binary Data:
        For data like tensors, dataframes, or large binary objects that are
        inefficient or impossible to represent directly in JSON, it is strongly
        recommended to serialize them to an external format (e.g., Parquet,
        Zarr, NPY, Pickle) on disk or cloud storage. This method should then
        return a JSON-serializable reference, such as a file path string,
        URI, or unique identifier.

        Args:
            outputs (TData): The outputs object instance from `self.outputs`.

        Returns:
            Any: A representation of the outputs suitable for Pydantic serialization.
                 This MUST be a standard JSON-serializable type (e.g., dict,
                 list, str, int, float, bool, None) or convertible to one by
                 Pydantic's encoders. If serializing externally, this would
                 typically be a string path or identifier.
        """

    @classmethod
    @field_validator("outputs", mode="before")
    @abstractmethod
    def deserialize_outputs(cls: type["Run"], data: Any) -> TData:
        """
        Abstract class method to deserialize the run's outputs.

        This method performs the inverse operation of `serialize_outputs`.
        Subclasses MUST implement this to convert the serialized representation
        (`data`) back into the original object structure defined by `TData`.

        It can be called manually, or by Pydantic before the `outputs` field is
        assigned during model parsing/initialization (e.g., via `.model_validate()`, `.model_validate_json()`).

        If `serialize_outputs` saved data externally and returned a path/identifier,
        this method will receive that path/identifier in the `data` argument
        and is responsible for loading the actual data from that location.

        Args:
            cls (Type[Self]): The specific subclass of `Run` being processed.
                              Can be used if deserialization logic depends on
                              class-level attributes or methods.
            data (Any): The serialized representation of the outputs, exactly
                        as produced by `serialize_outputs` (potentially after
                        JSON encoding/decoding if the model was serialized/
                        deserialized via JSON). The type depends on the output
                        of `serialize_outputs` (e.g., dict, list, str path).

        Returns:
            TData: The reconstructed outputs object.
        """


# TO DO: Temporary dummy implementation - remove after #339 epic is completed
class DummyRun(Run):
    def serialize_outputs(self, outputs: Any) -> None:  # noqa: ARG002
        raise RuntimeError("This is a dummy implementation and should not be used to serialize data.")

    @classmethod
    def deserialize_outputs(cls, data: Any) -> None:  # noqa: ARG003
        raise RuntimeError("This is a dummy implementation and should not be used to deserialize data.")


# TO DO: Temporary dummy implementation - remove after #339 epic is completed
class DummyConfig(BaseModel):
    pass


class Cache(Generic[TData]):
    """Caching mechanism for test stages"""

    def read_cache(self, cache_path: str) -> Optional[TData]: ...
    def write_cache(self, cache_path: str, data: TData) -> None: ...


class RIValidationError(Exception):
    """Exception raised for validation errors."""

    def __init__(self, message: str = "Validation error occurred.") -> None:
        self.message = message
        super().__init__(self.message)


class TestStage(Generic[TData], ABC):
    """Base class for running a test and recieving report values"""

    # TO DO: Use Generics instead of concrete implemetation
    __result__: DummyRun

    _deck: str
    _task: str
    # TO DO: Remove _outputs
    _outputs: Optional[TData] = None  # test results are expected to be stored within the test stage
    _batch_size: int = 1  # Not fully implemented yet - Ref Issue 270 "Expose batch size in test stages"
    cache: Optional[Cache[TData]] = None

    @property
    def outputs(self) -> TData:
        """Property getter for TestStage run outputs - raises RunTimeError if accessed before set"""
        if not hasattr(self, "__result__"):
            raise RuntimeError("TestStage must be run before accessing outputs")
        return self.__result__.outputs

    @outputs.setter
    def outputs(self, value: TData) -> None:
        """Property setter for TestStage run outputs"""
        # TO DO: move this logic inside of _run
        self.__result__ = DummyRun(config=DummyConfig(), model_ids=[], dataset_ids=[], metric_id="", outputs=value)

    @property
    def cache_id(self) -> str:
        """Override this with a unique cache id to save outputs to cache"""
        return ""

    @property
    def cache_path(self) -> str:
        return str(cache_path() / self.cache_id) if self.cache_id else ""

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

    # TO DO: update signature to accept config, models, datasets and metrics
    def run(self, use_stage_cache: bool = True) -> DummyRun:
        """Run the test stage leveraging cache if available and store any outputs of the evaluation in test stage"""
        if use_stage_cache and self.cache and self.cache_path:
            cached_outputs = self.cache.read_cache(self.cache_path)
            if cached_outputs:
                self.outputs = cached_outputs
                return self.__result__

        self.outputs = self._run()

        if use_stage_cache and self.cache and self.cache_path:
            self.cache.write_cache(self.cache_path, self.outputs)

        return self.__result__

    # TO DO: update signature to accept config, models, datasets and metrics
    @abstractmethod
    def _run(self) -> TData:
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
