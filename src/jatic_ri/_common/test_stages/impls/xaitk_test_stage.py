"""XAITKTestStage implementation"""

# Python generic imports
from __future__ import annotations

from typing import Any

# Local imports
from jatic_ri._common.test_stages.interfaces.plugins import (
    MetricPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    TDataset,
    ThresholdPlugin,
    TMetric,
    TModel,
)
from jatic_ri._common.test_stages.interfaces.test_stage import TestStage


class XAITKTestStageBase(
    TestStage[dict[str, Any]],
    SingleModelPlugin[TModel],
    SingleDatasetPlugin[TDataset],
    MetricPlugin[TMetric],
    ThresholdPlugin,
):
    """
    XAITK Test Stage that takes in the necessary arguements to demo saliency map generation.

    Attributes:
        config (dict[str, Any]):The configuration dictionary that will be used to create
                                saliency map generator object.
        stage_name (str): The name of the test stage.
        sal_generator_hash (str): A unique hash identifying the saliency map generator configuration.
        id2label (dict[int, Hashable]): A mapping of id to label for the dataset (will be moved in future)

    """

    config: dict[str, Any]
    stage_name: str
    sal_generator_hash: str
    img_batch_size: int

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.config = args
        self.stage_name = args["name"]
        self.img_batch_size = args["img_batch_size"]

    @property
    def cache_id(self) -> str:
        """Cache file for XAITK Test Stage"""
        return f"xaitk_{self._task}_{self.model_id}_{self.dataset_id}_{self.sal_generator_hash}.json"

    @property
    def name(self) -> str:
        """Returns classname as a string"""
        return self.__class__.__name__
