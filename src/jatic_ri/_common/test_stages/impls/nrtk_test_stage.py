"""NRTKTestStage implementation"""

# Python generic imports
from __future__ import annotations

from abc import abstractmethod
from typing import Any

import pandas as pd
from maite.workflows import evaluate
from nrtk.interfaces.perturb_image import PerturbImage

# NRTK imports
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

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
from jatic_ri.image_classification.augmentation import JATICClassificationAugmentation
from jatic_ri.object_detection.augmentation import JATICDetectionAugmentation


class NRTKTestStageBase(
    TestStage["dict[str, Any]"],
    SingleDatasetPlugin[TDataset],
    SingleModelPlugin[TModel],
    MetricPlugin[TMetric],
    ThresholdPlugin,
):
    """
    NRTK Test Stage to perform augmentation on images in a dataset based
    on a given factory configuration.`

    Attributes:
        config (dict[str, Any]):The configuration dictionary that will be used to create
                                PerturbImageFactory object.
        stage_name (str): The name of the test stage.
        factory_hash (str): A unique hash identifying the perturbation factory configuration.
        factory (PerturbImageFactory):The perturbation factory used to generate PerturbImage
                                      augmentations.

    """

    config: dict[str, Any]
    stage_name: str
    factory_hash: str
    factory: PerturbImageFactory

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.config = args
        self.stage_name = args["name"]

    @property
    def cache_id(self) -> str:
        """Cache file for NRTK Test Stage"""
        return f"nrtk_{self._task}_{self.model_id}_{self.dataset_id}_{self.factory_hash}.json"

    @property
    def name(self) -> str:
        """Returns classname as a string"""
        return self.__class__.__name__

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report generated in the run method or in the
        load_cached_results method"""

        lowest_perturb_score = [self.outputs[perturber_output][self.metric_id] for perturber_output in self.outputs]

        final_dict = {
            "dataset": self.dataset_id,
            "model": self.model_id,
            self.factory.theta_key: self.factory.thetas,
            self.metric_id: lowest_perturb_score,
        }
        df_perturbation = pd.DataFrame.from_dict(final_dict)
        df_perturbation["line_id"] = "item_response_curve"

        return [
            {
                "deck": self._deck,
                "layout_name": "NRTKEvaluation",
                "layout_arguments": {
                    "title": self.name,
                    "data": df_perturbation,
                    "line_col": "line_id",
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

    def _run(self) -> dict[str, dict[str, Any]]:
        """Run the test stage, and store any outputs of the evaluation in test stage"""

        outputs = dict()  # noqa: C408
        for count, perturber in enumerate(self.factory):
            augmentation = self._augmentation_wrapper(perturber)

            perturbed_metrics, _, _ = evaluate(
                model=self.model,
                dataset=self.dataset,
                metric=self.metric,
                batch_size=1,
                augmentation=augmentation,
                return_augmented_data=False,
                return_preds=False,
            )
            outputs[f"perturbation_{count}"] = perturbed_metrics

        return outputs

    @abstractmethod
    def _augmentation_wrapper(
        self, perturber: PerturbImage
    ) -> JATICDetectionAugmentation | JATICClassificationAugmentation:
        """Takes in a perturber and returns a maite Augmentation"""
