"""NRTKTestStage implementation"""

import re
from typing import Any

import pandas as pd
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interop.maite.interop.image_classification.augmentation import JATICClassificationAugmentation
from nrtk.interop.maite.interop.object_detection.augmentation import JATICDetectionAugmentation

from jatic_ri._common.test_stages.interfaces.plugins import (
    MetricPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    TDataset,
    ThresholdPlugin,
    TMetric,
    TModel,
)
from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, OutputsBase, RunBase, TestStage
from jatic_ri.cached_tasks import evaluate
from jatic_ri.util._types import DeSerializablePlugfigurable


class NRTKTestStageConfig(ConfigBase):
    name: str
    perturber_factory: DeSerializablePlugfigurable[PerturbImageFactory]


class NRTKTestStageOutputs(OutputsBase):
    perturbations: list[dict[str, Any]]


class NRTKTestStageRun(RunBase):
    config: NRTKTestStageConfig
    outputs: NRTKTestStageOutputs


class NRTKTestStageBase(
    TestStage[NRTKTestStageOutputs],
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

    """

    stage_name: str

    _RUN_TYPE = NRTKTestStageRun

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.config = NRTKTestStageConfig.model_validate(args)

    def _create_config(self) -> NRTKTestStageConfig:
        return self.config

    @property
    def name(self) -> str:
        """Returns classname as a string"""
        return self.__class__.__name__

    def __labelx_gen(self, perturber: str, theta_key: Any) -> str:
        """Returns the labelx name for the report consumables stage"""
        perturber = perturber.rpartition(".")[-1].replace("Perturber", "")
        theta_key_map = {
            "factor": "Factor",
            "ksize": "Kernel Size",
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

        if self._stored_run is None:
            raise RuntimeError("TestStage must be run before accessing outputs")
        outputs = self._stored_run.outputs

        lowest_perturb_score = [perturber_output[self.metric_id] for perturber_output in outputs.perturbations]

        final_dict = {
            "dataset": self.dataset_id,
            "model": self.model_id,
            self.config.perturber_factory.theta_key: self.config.perturber_factory.thetas,
            self.metric_id: lowest_perturb_score,
        }
        df_perturbation = pd.DataFrame.from_dict(final_dict)
        df_perturbation["line_id"] = "item_response_curve"

        # convert pert classname into semantic label
        # (e.g. nrtk.impls.perturb_image.generic.PIL.enhance.BrightnessPerturber into Brightness Perturber)
        perturbation_classname = self.config.perturber_factory.get_config()["perturber"].split(".")[-1]
        perturbation_label = " ".join(re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", perturbation_classname))

        return [
            {
                "deck": self._deck,
                "layout_name": "NRTKEvaluation",  # specialized template in gradient codebase
                "layout_arguments": {
                    "title": self.name,
                    "data": df_perturbation,
                    "line_col": "line_id",
                    "x_data_col": self.config.perturber_factory.theta_key,
                    "y_data_col": self.metric_id,
                    "perturbation_type": perturbation_label,
                    "lower_bound": 3.4,
                    "upper_bound": 5.3,
                    "model": self.model_id,
                    "plot_kwargs": {
                        "y_threshold_value": self.threshold,
                        "title": "NRTK Robustness Curve",
                        "x_label": self.__labelx_gen(
                            self.config.perturber_factory.get_config()["perturber"],
                            self.config.perturber_factory.theta_key,
                        ),
                        "y_label": self.__labely_gen(),
                    },
                },
            },
        ]

    def _run(self) -> NRTKTestStageOutputs:
        """Run the test stage, and store any outputs of the evaluation in test stage"""

        perturbations = list()  # noqa: C408
        for perturber in self.config.perturber_factory:
            if self._task == "object_detection":
                augmentation = JATICDetectionAugmentation(augment=perturber, augment_id="JATICDetection")
                perturbed_metrics, _, _ = evaluate(
                    model=self.model,
                    dataset=self.dataset,
                    metric=self.metric,
                    batch_size=1,
                    augmentation=augmentation,
                    return_augmented_data=False,
                    return_preds=False,
                )
            elif self._task == "image_classification":
                augmentation = JATICClassificationAugmentation(augment=perturber, augment_id="JATICClassification")
                perturbed_metrics, _, _ = evaluate(
                    model=self.model,
                    dataset=self.dataset,
                    metric=self.metric,
                    batch_size=1,
                    augmentation=augmentation,
                    return_augmented_data=False,
                    return_preds=False,
                )
            else:
                raise ValueError(f"Invalid value for _task provided, _task:{self._task}")

            perturbations.append(perturbed_metrics)

        return NRTKTestStageOutputs(perturbations=perturbations)
