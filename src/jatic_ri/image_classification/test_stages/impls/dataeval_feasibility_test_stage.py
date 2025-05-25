"""
The Dataset Feasibility Test Stage implementation

Classes
-------
DatasetFeasibilityTestStage
    Generates a gradient report based on the Bayes Error Rate of a single dataset
"""

from typing import Any

import maite.protocols.image_classification as ic
import numpy as np
import pandas as pd
from dataeval.data import Embeddings, Metadata
from dataeval.metrics.estimators import BEROutput, ber
from gradient import SubText
from gradient.slide_deck.shapes import Text
from gradient.templates_and_layouts.generic_layouts.section_by_item import SectionByItem

from jatic_ri._common.test_stages.impls._dataeval_utils import get_resnet18
from jatic_ri._common.test_stages.interfaces.plugins import SingleDatasetPlugin, ThresholdPlugin
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.util.cache import JSONCache, NumpyEncoder


class DatasetFeasibilityTestStage(
    TestStage[dict[str, dict[str, float]]], SingleDatasetPlugin[ic.Dataset], ThresholdPlugin
):
    """Image Classification feasibility test stage"""

    cache: Cache[dict[str, dict[str, float]]] | None = JSONCache(encoder=NumpyEncoder)
    _precision = 3

    # TODO: move to config
    _device = "cpu"

    @property
    def cache_id(self) -> str:
        """Unique path identifier for feasibility tasks"""
        return f"feasibility_{self.dataset_id}_{self.threshold}.json"

    def _run(self) -> dict[str, dict[str, float]]:
        """Calculate the Bayes Error Rate from the images and labels"""

        model, transform = get_resnet18()
        embeddings = Embeddings(self.dataset, self._batch_size, transform, model, self._device)
        metadata = Metadata(self.dataset)

        b: BEROutput = ber(embeddings.to_numpy(), metadata.class_labels)
        return {"feasibility": b.data()}

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
            Text(t)
            for t in (
                [SubText("Result:", bold=True)],
                f"Performance goal of {self.threshold} {'is' if is_feasible else 'is NOT'} feasible.",
                [SubText("Tests for:", bold=True)],
                " * Achievability of performance goal",
                [SubText("Risk(s):", bold=True)],
                " * Performance goal cannot be achieved by any model (problem too hard)",
                " * Models that report performance above the goal are overfit and \
                will not generalize to real-world problems",
                [SubText("Action:", bold=True)],
                f"* {'No action required' if is_feasible else 'Reduce difficulty of the problem statement'}",
            )
        ]

        return [
            {
                "deck": "image_classification_dataset_evaluation",
                "layout_name": "SectionByItem",
                "layout_arguments": {
                    SectionByItem.ArgKeys.TITLE.value: title,
                    SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
                    SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
                    SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: feasibility_df,
                },
            },
        ]
