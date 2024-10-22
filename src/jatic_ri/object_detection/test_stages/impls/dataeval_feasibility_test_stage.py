"""DataEval Feasibility Test Stage"""

from collections.abc import Sequence
from typing import Any, Optional

import maite.protocols.object_detection as od
import numpy as np
import pandas as pd
from dataeval.metrics.estimators import uap
from gradient.slide_deck.shapes import Text
from gradient.templates_and_layouts.generic_layouts.text_data import TextData
from maite.workflows import predict

from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import (
    MetricPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    ThresholdPlugin,
)
from jatic_ri.util.cache import JSONCache, NumpyEncoder


class DatasetFeasibilityTestStage(
    TestStage[dict[str, float]],
    SingleModelPlugin,
    SingleDatasetPlugin,
    MetricPlugin,
    ThresholdPlugin,
):
    """Docstring"""

    cache: Optional[Cache[dict[str, float]]] = JSONCache(encoder=NumpyEncoder)

    @property
    def cache_id(self) -> str:
        """Unique path identifier for feasibility tasks"""
        return f"feasibility_{self.dataset_id}_{self.model_id}_{self.threshold}.json"

    def _run(self) -> dict[str, float]:
        # Returns a tuple of the predictions (as a sequence of batches) and
        # a sequence of tuples containing the information associated with each batch.
        # Associated information unneeded
        predictions, _ = predict(model=self.model, dataset=self.dataset, batch_size=32)

        # Reassign to enforce type-hinting checks
        batches: Sequence[Sequence[od.TargetType]] = predictions

        # Flattens batches into continuous list of all target objects
        targets: Sequence[od.TargetType] = []
        for batch in batches:
            targets.extend(batch)

        # Collect all labels as list of target labels, then stack
        labels = np.array([target.labels for target in targets]).flatten()

        unique_labels = set(labels)
        # Collect scores and reshape into (N, C)
        scores = [target.scores for target in targets]
        scores = np.array(scores).reshape(-1, len(unique_labels))

        result = uap(labels=labels, scores=scores)

        return {"uap": result.uap}

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Create kwargs for Gradient create deck function"""

        # Data to be reported on the slide
        uap = self.outputs["uap"]
        threshold = self.threshold
        is_feasible = uap >= threshold

        # Set up a dataframe with formatting
        feasibility_df = pd.DataFrame(
            {
                "Feasible": [str(is_feasible)],
                "Upperbound Average Precision": [round(uap, 3)],
                "Performance Goal": [round(threshold, 3)],
            },
        )

        # Gradient slide kwargs
        title = f"Dataset: {self.dataset_id} | Category: Feasibility"
        heading = "Metric: UAP"
        text = [
            "**Result:**",
            f"* Performance goal of {threshold} {'is' if is_feasible else 'is NOT'} feasible.",
            "**Tests for:**",
            "* Achievability of performance goal",
            "**Risks:**",
            "* Performance goal cannot be achieved by any model",
            "* Models that report performance above the bound are overfit and will not generalize to real-world problems",  # noqa: E501
            "**Action:**",
            f"* {'No action required' if is_feasible else 'Reduce difficulty of the problem statement'}",
        ]
        content = [Text(t, fontsize=16) for t in text]

        # Set up Gradient slide
        feasibility_slide_args = {
            "deck": "object_detection_dataset_evaluation",
            "layout_name": "TextData",
            "layout_arguments": {
                TextData.ArgKeys.TITLE.value: title,
                TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                TextData.ArgKeys.DATA_COLUMN_TABLE.value: feasibility_df,
            },
        }

        return [feasibility_slide_args]
