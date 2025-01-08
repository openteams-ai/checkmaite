"""
The Dataset Feasibility Test Stage implementation

Classes
-------
DatasetFeasibilityTestStage
    Generates a gradient report based on the Bayes Error Rate of a single dataset
"""

from typing import Any, Optional

import maite.protocols.image_classification as ic
import numpy as np
import pandas as pd
from dataeval.metrics.estimators import BEROutput, ber
from dataeval.utils.dataset import read_dataset
from gradient.templates_and_layouts.generic_layouts.text_data import TextData

from jatic_ri._common.test_stages.interfaces.plugins import SingleDatasetPlugin, ThresholdPlugin
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.util.cache import JSONCache, NumpyEncoder


class DatasetFeasibilityTestStage(
    TestStage[dict[str, dict[str, float]]], SingleDatasetPlugin[ic.Dataset], ThresholdPlugin
):
    """Image Classification feasibility test stage"""

    cache: Optional[Cache[dict[str, dict[str, float]]]] = JSONCache(encoder=NumpyEncoder)
    _precision = 3

    @property
    def cache_id(self) -> str:
        """Unique path identifier for feasibility tasks"""
        return f"feasibility_{self.dataset_id}_{self.threshold}.json"

    def _run(self) -> dict[str, dict[str, float]]:
        """Calculate the Bayes Error Rate from the images and labels"""

        images, targets, _ = read_dataset(self.dataset)  # type: ignore

        b: BEROutput = ber(images, targets)
        return {"feasibility": b.dict()}

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Create Gradient deck kwargs for Image Classification Feasibility"""
        results = self.outputs["feasibility"]
        is_feasible = results["ber"] > self.threshold
        feasibility_dict = {
            "Feasible": [str(is_feasible)],
            "Bayes Error Rate": [np.round(results["ber"], self._precision)],
            "Lower Bayes Error Rate": [np.round(results["ber_lower"], self._precision)],
            "Performance Goal": [self.threshold],
        }
        feasibility_df = pd.DataFrame.from_dict(feasibility_dict)

        title = f"Dataset: {self.dataset_id} | Category: Feasibility"
        heading = "Metric: Bayes Error Rate"
        content = [
            "**Result:**",
            f"Performance goal of {self.threshold} {'is' if is_feasible else 'is NOT'} feasible.",
            "Tests for:",
            " * Achievability of performance goal",
            "Risk(s):",
            " * Performance goal cannot be achieved by any model (problem too hard)",
            " * Models that report performance above the goal are overfit and \
                will not generalize to real-world problems",
            "Action:",
            f"* {'No action required' if is_feasible else 'Reduce difficulty of the problem statement'}",
        ]

        return [
            {
                "deck": "image_classification_dataset_evaluation",
                "layout_name": "TextData",
                "layout_arguments": {
                    TextData.ArgKeys.TITLE.value: title,
                    TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                    TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                    TextData.ArgKeys.DATA_COLUMN_TABLE.value: feasibility_df,
                },
            },
        ]
