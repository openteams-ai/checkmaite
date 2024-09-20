"""Drift Test Stage Implementation"""

from functools import partial
from statistics import mean
from typing import Any

import torch
from dataeval.detectors.drift import (
    DriftCVM,
    DriftKS,
    DriftMMD,
)

from jatic_ri._common.test_stages.interfaces.test_stage import TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import TwoDatasetPlugin


class DatasetDriftTestStage(TestStage, TwoDatasetPlugin):
    """
    Drift detection TestStage implementation.

    Performs 3 methods of drift detection (Maximum mean discrepency, Cramer-von Mises, and Kolmogorov-Smirnov)
    against the operational dataset using the development dataset as the reference. Takes target model for
    preprocessing the images before performing drift detection against the embeddings.
    """

    outputs = None
    device = "cpu"

    def run(self, use_cache: bool = False) -> None:
        """Run MMD, CVM and KS drift detectors"""

        if use_cache:
            return

        images_1 = torch.stack([data[0] for data in self.dataset_1])  # type: ignore
        images_2 = torch.stack([data[0] for data in self.dataset_2])  # type: ignore

        # model expects embedding outputs rather than the prediction outputs
        # preprocess_fn = partial(preprocess_drift, model=self.model, batch_size=64, device=self.device)
        drift_kwargs = {"x_ref": images_1}
        drift_cls = {
            "Maximum Mean Discrepency": partial(DriftMMD, device=self.device),
            "Cramér-von Mises": DriftCVM,
            "Kolmogorov-Smirnov": DriftKS,
        }

        self.outputs = {k: cls(**drift_kwargs).predict(images_2) for k, cls in drift_cls.items()}

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect drift results"""

        if not isinstance(self.outputs, dict):
            return []

        def get_mean(output, attr_mean, attr) -> float:  # noqa: ANN001
            return mean(getattr(output, attr_mean)) if hasattr(output, attr_mean) else getattr(output, attr)

        return [
            {
                "Method": list(self.outputs),
                "Has drifted?": ["Yes" if d.is_drift else "No" for d in self.outputs.values()],
                "Test statistic": [str(get_mean(d, "distances", "distance")) for d in self.outputs.values()],
                "P-value": [str(get_mean(d, "p_vals", "p_val")) for d in self.outputs.values()],
            },
        ]
