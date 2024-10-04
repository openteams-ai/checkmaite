"""Drift Test Stage Implementation"""

from functools import partial
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from dataeval.detectors.drift import (
    DriftCVM,
    DriftKS,
    DriftMMD,
)
from dataeval.utils import read_dataset

from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import TwoDatasetPlugin
from jatic_ri.util.cache import JSONCache, NumpyEncoder


class DatasetDriftTestStage(TestStage[dict[str, Any]], TwoDatasetPlugin):
    """
    Drift detection TestStage implementation.

    Performs 3 methods of drift detection (Maximum mean discrepency, Cramer-von Mises, and Kolmogorov-Smirnov)
    against the operational dataset using the development dataset as the reference. Takes target model for
    preprocessing the images before performing drift detection against the embeddings.
    """

    outputs: Optional[dict[str, Any]] = None
    cache: Optional[Cache[dict[str, Any]]] = JSONCache(encoder=NumpyEncoder)
    device = "cpu"

    @property
    def cache_id(self) -> str:
        """Unique identifier for cached results"""
        return f"drift-{self.dataset_1_id}-{self.dataset_2_id}.json"

    def _run(self) -> None:
        """Run MMD, CVM and KS drift detectors"""
        images_1 = torch.stack(read_dataset(self.dataset_1)[0])  # type: ignore
        images_2 = torch.stack(read_dataset(self.dataset_2)[0])  # type: ignore

        drift_kwargs = {"x_ref": images_1}
        drift_cls = {
            "Maximum Mean Discrepency": partial(DriftMMD, device=self.device),
            "Cramér-von Mises": DriftCVM,
            "Kolmogorov-Smirnov": DriftKS,
        }

        self.outputs = {k: cls(**drift_kwargs).predict(images_2).dict() for k, cls in drift_cls.items()}

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect drift results"""

        if not self.outputs:
            return []

        def get_mean(d: dict, array_key: str, value_key: str) -> float:
            return np.mean(d[array_key]) if array_key in d else d[value_key]

        drift_df = pd.DataFrame(
            {
                "Method": list(self.outputs),
                "Has drifted?": ["Yes" if d["is_drift"] else "No" for d in self.outputs.values()],
                "Test statistic": [str(get_mean(d, "distances", "distance")) for d in self.outputs.values()],
                "P-value": [str(get_mean(d, "p_vals", "p_val")) for d in self.outputs.values()],
            },
        )

        drift_slide_args = {
            "deck": "object_detection_dataset_evaluation",
            "layout_name": "TableText",
            "layout_arguments": {
                "title": "Drift Results",
                "text": "Drift Body Text",
                "table": drift_df,
            },
        }

        return [drift_slide_args]
