"""Base Test Stage for all test implementations"""

import os
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

TData = TypeVar("TData")


class Cache(Generic[TData]):
    """Caching mechanism for test stages"""

    def read_cache(self, cache_path: str) -> Optional[TData]: ...
    def write_cache(self, cache_path: str, data: TData) -> None: ...


class TestStage(Generic[TData], ABC):
    """Base class for running a test and recieving report values"""

    outputs: Optional[TData]  # test results are expected to be stored within the test stage
    cache: Optional[Cache[TData]] = None
    cache_base_path: str = ".tscache"

    @property
    def cache_id(self) -> str:
        """Override this with a unique cache id to save outputs to cache"""
        return ""

    @property
    def cache_path(self) -> str:
        return os.path.join(self.cache_base_path, self.cache_id) if self.cache_id else ""

    def run(self, use_cache: bool = True) -> None:
        """Run the test stage leveraging cache if available and store any outputs of the evaluation in test stage"""

        if use_cache and self.cache and self.cache_path:
            cached_outputs = self.cache.read_cache(self.cache_path)
            if cached_outputs:
                self.outputs = cached_outputs
                return

        self._run()

        if use_cache and self.cache and self.cache_path and self.outputs:
            self.cache.write_cache(self.cache_path, self.outputs)

    @abstractmethod
    def _run(self) -> None:
        """Override this with logic to execute test stage and store outputs"""

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
