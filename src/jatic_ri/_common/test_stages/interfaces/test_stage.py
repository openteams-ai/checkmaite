"""Base Test Stage for all test implementations"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from jatic_ri import cache_path

TData = TypeVar("TData")


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

    _deck: str
    _task: str
    _outputs: Optional[TData] = None  # test results are expected to be stored within the test stage
    _batch_size: int = 1  # Not fully implemented yet - Ref Issue 270 "Expose batch size in test stages"
    cache: Optional[Cache[TData]] = None

    @property
    def outputs(self) -> TData:
        """Property getter for TestStage run outputs - raises RunTimeError if accessed before set"""
        if self._outputs is None:
            raise RuntimeError("TestStage must be run before accessing outputs")
        return self._outputs

    @outputs.setter
    def outputs(self, value: TData) -> None:
        """Property setter for TestStage run outputs"""
        self._outputs = value

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

    def run(self, use_stage_cache: bool = True) -> None:
        """Run the test stage leveraging cache if available and store any outputs of the evaluation in test stage"""
        if use_stage_cache and self.cache and self.cache_path:
            cached_outputs = self.cache.read_cache(self.cache_path)
            if cached_outputs:
                self.outputs = cached_outputs
                return

        self.outputs = self._run()

        if use_stage_cache and self.cache and self.cache_path:
            self.cache.write_cache(self.cache_path, self.outputs)

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
