"""NRTKTestStage implementation"""

# Python generic imports
from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

# 3rd party imports
import pandas as pd

# MAITE imports
from maite.workflows import evaluate

# NRTK imports
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

# SMQTK imports
from smqtk_core.configuration import from_config_dict

# Import TestStage
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.object_detection.augmentation import JATICDetectionAugmentation
from jatic_ri.object_detection.test_stages.interfaces.plugins import (
    MetricPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    ThresholdPlugin,
)
from jatic_ri.util.cache import JSONCache


class NRTKTestStage(
    TestStage[list[dict[str, Any]]],
    SingleDatasetPlugin,
    SingleModelPlugin,
    MetricPlugin,
    ThresholdPlugin,
):
    """
    NRTK Test Stage to perform augmentation on images in a dataset based
    on a given factory configuration.
    """

    config: dict[str, Any]
    stage_name: str
    factory: PerturbImageFactory
    factory_hash: str
    cache: Cache[list[dict[str, Any]]] | None = JSONCache()

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.config = args
        self.stage_name = args["name"]
        self.factory = from_config_dict(args["perturber_factory"], PerturbImageFactory.get_impls())
        self.factory_hash = sha256(json.dumps(args["perturber_factory"]).encode("utf-8")).hexdigest()

    @property
    def cache_id(self) -> str:
        """Cache file for NRTK Test Stage"""
        return f"nrtk_{self.model_id}_{self.dataset_id}_{self.factory_hash}.json"

    def _run(self) -> list[dict[str, Any]]:
        """Run the test stage, and store any outputs of the evaluation in test stage"""

        outputs = list()  # noqa: C408

        for perturber in self.factory:
            augmentation = JATICDetectionAugmentation(perturber)

            perturbed_metrics, _, _ = evaluate(
                model=self.model,
                dataset=self.dataset,
                metric=self.metric,
                batch_size=1,
                augmentation=augmentation,
                return_augmented_data=False,
                return_preds=False,
            )
            outputs.append(perturbed_metrics)

        return outputs

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report generated in the run method or in the
        load_cached_results method"""

        lowest_perturb_score = [metric[self.metric_id] for metric in self.outputs]

        final_dict = {
            "dataset": self.dataset_id,
            "model": self.model_id,
            self.factory.theta_key: self.factory.thetas,
            self.metric_id: lowest_perturb_score,
        }
        df_perturbation = pd.DataFrame.from_dict(final_dict)

        return [
            {
                "deck": "object_detection_dataset_evaluation",
                "layout_name": "NRTKEvaluation",
                "layout_arguments": {
                    "title": self.name,
                    "data": df_perturbation,
                    "line_col": "item_response_curve",
                    "x_data_col": self.factory.theta_key,
                    "y_data_col": self.metric_id,
                    "perturbation_type": self.factory.get_config()["perturber"],
                    "lower_bound": 3.4,
                    "upper_bound": 5.3,
                    "model": self.model_id,
                    "plot_kwargs": {
                        "y_threshold_value": self.threshold,
                        "title": "NRTK Item Response Curve",
                    },
                },
            },
        ]

    @property
    def name(self) -> str:
        """Returns classname as a string"""
        return self.__class__.__name__
