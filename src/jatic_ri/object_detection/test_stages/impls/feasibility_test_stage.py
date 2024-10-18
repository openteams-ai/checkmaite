"""DataEval Feasibility Test Stage"""

from collections.abc import Sequence
from typing import Any, Optional

import maite.protocols.object_detection as od
import numpy as np
from dataeval.metrics.estimators import uap
from maite.workflows import evaluate

from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import (
    MetricPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    ThresholdPlugin,
)
from jatic_ri.util.cache import JSONCache, NumpyEncoder


class FeasibilityTestStage(
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
        return f"feasibility-{self.dataset}-{self.model}-{self.threshold}.json"

    def _run(self) -> dict[str, float]:
        results = evaluate(model=self.model, dataset=self.dataset)

        batches: Sequence[Sequence[od.TargetType]] = results[1]

        # Flattens batches into continuous list of all targets
        targets: Sequence[od.TargetType] = []
        for batch in batches:
            targets.extend(batch)

        # Collect all labels into 1D array of shape (N,)
        labels = np.array([target.labels for target in targets]).flatten()
        unique_labels = set(labels)
        # Collect scores and reshape into (N, C)
        scores = [target.scores for target in targets]
        scores = np.array(scores).reshape(-1, len(unique_labels))

        result = uap(labels=labels, scores=scores)

        return {"uap": result.uap}

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect"""
        return [
            {
                "feasible": self.outputs["uap"] >= self.threshold,
            },
        ]
