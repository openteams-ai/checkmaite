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
from jatic_ri.util.cache import JSONCache, TensorEncoder


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
        config: The configuration dictionary that will be used to create
                                PerturbImageFactory object.
        stage_name: The name of the test stage.
        factory_hash: A unique hash identifying the perturbation factory configuration.
        factory: The perturbation factory used to generate PerturbImage
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
        self.cache = JSONCache(encoder=TensorEncoder)

    @property
    def cache_id(self) -> str:
        """Cache file for NRTK Test Stage"""
        return f"nrtk_{self._task}_{self.model_id}_{self.dataset_id}_{self.factory_hash}.json"

    @property
    def name(self) -> str:
        """Returns classname as a string"""
        return self.__class__.__name__

    def __labelx_gen(self, perturber: str, theta_key: Any) -> str:
        """Returns the labelx name for the report consumables stage"""
        perturber = perturber.rpartition(".")[-1].replace("Perturber", "")
        theta_key_map = {
            "factor": "Factor",
            "s_x": "Root Mean Squared",
            "s_y": "Root Mean Squared",
            "p_x": "Pitch in X Direction",
            "w_x": "Detector Width",
            "w_y": "Detector Height",
            "read_noise": "Read Noise",
            "max_n": "Maximum ADC Level",
            "bit_depth": "Bit Depth",
            "da_x": "Drifts (radians/s)",
            "da_y": "Drifts (radians/s)",
            "ihaze": "Weather Model",
            "altitude": "Sensor Altitude",
            "ground_range": "Ground Range",
            "aircraft_speed": "Aircraft Speed",
            "amount": "Amount of Noise",
            "salt_vs_pepper": "Percentage of Salt vs Pepper",
            "mean": "Mean",
            "var": "Variance",
            "rng": "Pseudo Random Number Generator",
            "D": "Effective Aperture Diameter",
            "f": "Focal Length",
            "ifov": "Instantaneous Field of View",
            "eta": "Relative Linear Obscuration",
            "slant_range": "Line-of-Sight Distance",
            "interp": "Interpolation Method",
        }
        return f"{perturber} {theta_key_map.get(theta_key, theta_key)}"

    def __labely_gen(self) -> str:
        """Returns the labely name for the report consumables stage"""
        y_label_map = {
            "accuracy": "Accuracy",
            "f1_score": "F1 Score",
            "precision": "Precision",
            "recall": "Recall",
            "mAP": "Mean Average Precision",
            "map_50": "Mean Average Precision",
        }
        return y_label_map.get(self.metric_id, self.metric_id)

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
                        "x_label": self.__labelx_gen(self.factory.get_config()["perturber"], self.factory.theta_key),
                        "y_label": self.__labely_gen(),
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
