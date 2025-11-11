"""NRTKTestStage implementation"""

import re
from typing import Any

import pandas as pd
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interop.maite.interop.image_classification.augmentation import JATICClassificationAugmentation
from nrtk.interop.maite.interop.object_detection.augmentation import JATICDetectionAugmentation

from jatic_ri._common.test_stages.interfaces.test_stage import (
    ConfigBase,
    Number,
    OutputsBase,
    RunBase,
    TDataset,
    TestStage,
    TMetric,
    TModel,
)
from jatic_ri.cached_tasks import evaluate
from jatic_ri.util._types import DeSerializablePlugfigurable

PERTURBER_LABELS = {
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

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1_score": "F1 Score",
    "precision": "Precision",
    "recall": "Recall",
    "mAP": "Mean Average Precision",
    "map_50": "Mean Average Precision",
}


class NRTKTestStageConfig(ConfigBase):
    name: str
    perturber_factory: DeSerializablePlugfigurable[PerturbImageFactory]


class NRTKTestStageOutputs(OutputsBase):
    perturbations: list[dict[str, Any]]


class NRTKTestStageRun(RunBase):
    config: NRTKTestStageConfig
    outputs: NRTKTestStageOutputs

    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report generated in the run method or in the
        load_cached_results method

        Parameters
        ----------
        threshold : float
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
        list[dict[str, Any]]
            A list of slide definitions for the full report.
        """

        outputs = self.outputs

        dataset_id = self.dataset_metadata[0]["id"]
        metric_id = self.metric_metadata[0]["id"]
        model_id = self.model_metadata[0]["id"]

        lowest_perturb_score = [perturber_output[metric_id] for perturber_output in outputs.perturbations]

        final_dict = {
            "dataset": dataset_id,
            "model": model_id,
            self.config.perturber_factory.theta_key: self.config.perturber_factory.thetas,
            metric_id: lowest_perturb_score,
        }
        df_perturbation = pd.DataFrame.from_dict(final_dict)
        df_perturbation["line_id"] = "item_response_curve"

        # convert pert classname into semantic label
        # (e.g. nrtk.impls.perturb_image.generic.PIL.enhance.BrightnessPerturber into Brightness Perturber)
        perturbation_classname = self.config.perturber_factory.get_config()["perturber"].split(".")[-1]
        perturbation_label = " ".join(re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", perturbation_classname))

        perturber = self.config.perturber_factory.get_config()["perturber"].rpartition(".")[-1].replace("Perturber", "")
        theta_key = self.config.perturber_factory.theta_key

        return [
            {
                "deck": self.test_stage_id,
                "layout_name": "NRTKEvaluation",  # specialized template in gradient codebase
                "layout_arguments": {
                    "title": self.test_stage_id,
                    "data": df_perturbation,
                    "line_col": "line_id",
                    "x_data_col": self.config.perturber_factory.theta_key,
                    "y_data_col": metric_id,
                    "perturbation_type": perturbation_label,
                    "lower_bound": 3.4,
                    "upper_bound": 5.3,
                    "model": model_id,
                    "plot_kwargs": {
                        "y_threshold_value": threshold,
                        "title": "NRTK Robustness Curve",
                        "x_label": f"{perturber} {PERTURBER_LABELS.get(theta_key, theta_key)}",
                        "y_label": METRIC_LABELS.get(metric_id, metric_id),
                    },
                },
            },
        ]


class NRTKTestStageBase(TestStage[NRTKTestStageOutputs, TDataset, TModel, TMetric]):
    """
    NRTK Test Stage to perform augmentation on images in a dataset based
    on a given factory configuration.`

    Attributes:
        config: The configuration dictionary that will be used to create
                                PerturbImageFactory object.

    """

    _RUN_TYPE = NRTKTestStageRun

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.config = NRTKTestStageConfig.model_validate(args)

    def _create_config(self) -> NRTKTestStageConfig:
        return self.config

    @property
    def supports_datasets(self) -> Number:
        """Number of datasets this test stage supports.

        Returns
        -------
        Number
            An enumeration value indicating dataset support.
        """
        return Number.ONE

    @property
    def supports_models(self) -> Number:
        """Number of models this test stage supports.

        Returns
        -------
        Number
            An enumeration value indicating model support.
        """
        return Number.ONE

    @property
    def supports_metrics(self) -> Number:
        """Number of metrics this test stage supports.

        Returns
        -------
        Number
            An enumeration value indicating metric support.
        """
        return Number.ONE

    def _run(
        self,
        models: list[TModel],
        datasets: list[TDataset],
        metrics: list[TMetric],
    ) -> NRTKTestStageOutputs:
        """Run the test stage, and store any outputs of the evaluation in test stage"""
        model = models[0]
        dataset = datasets[0]
        metric = metrics[0]

        perturbations = []
        for perturber in self.config.perturber_factory:
            if self._task == "object_detection":
                augmentation = JATICDetectionAugmentation(augment=perturber, augment_id="JATICDetection")
                perturbed_metrics, _, _ = evaluate(
                    model=model,
                    dataset=dataset,
                    metric=metric,
                    batch_size=1,
                    augmentation=augmentation,
                    return_augmented_data=False,
                    return_preds=False,
                )
            elif self._task == "image_classification":
                augmentation = JATICClassificationAugmentation(augment=perturber, augment_id="JATICClassification")
                perturbed_metrics, _, _ = evaluate(
                    model=model,
                    dataset=dataset,
                    metric=metric,
                    batch_size=1,
                    augmentation=augmentation,
                    return_augmented_data=False,
                    return_preds=False,
                )
            else:
                raise ValueError(f"Invalid value for _task provided, _task:{self._task}")

            perturbations.append(perturbed_metrics)

        return NRTKTestStageOutputs(perturbations=perturbations)
