"""Base Test Stage for all test implementations"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class TestStage(ABC):
    """Base class for running a test and recieving report values"""

    outputs: Optional[Any]  # test results are expected to be stored within the test stage

    @abstractmethod
    def run(self, use_cache: bool = True) -> None:
        """Run the test stage, and store any outputs of the evaluation in test stage"""

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
        return []
