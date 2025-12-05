import re
from typing import Any

import pandas as pd
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from pydantic import Field
from smqtk_core.configuration import from_config_dict

from jatic_ri.core._types import DeSerializablePlugfigurable
from jatic_ri.core.capability_core import (
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    CapabilityRunner,
    Number,
    TDataset,
    TMetric,
    TModel,
)

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


def _default_perturber_factory() -> PerturbImageFactory:
    return from_config_dict(
        {
            "type": "nrtk.impls.perturb_image_factory.generic.one_step.OneStepPerturbImageFactory",
            "nrtk.impls.perturb_image_factory.generic.one_step.OneStepPerturbImageFactory": {
                "perturber": "nrtk.impls.perturb_image.generic.PIL.enhance.BrightnessPerturber",
                "theta_key": "factor",
                "theta_value": 10.0,
            },
        },
        PerturbImageFactory.get_impls(),
    )


class NrtkAugmentationConfig(CapabilityConfigBase):
    name: str = "natural_robustness_test_factory"
    perturber_factory: DeSerializablePlugfigurable[PerturbImageFactory] = Field(
        default_factory=_default_perturber_factory
    )


class NrtkAugmentationOutputs(CapabilityOutputsBase):
    perturbations: list[dict[str, Any]]
    return_key: str


class NrtkAugmentationRun(CapabilityRunBase[NrtkAugmentationConfig, NrtkAugmentationOutputs]):
    config: NrtkAugmentationConfig
    outputs: NrtkAugmentationOutputs

    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report generated in the run method or in the
        load_cached_results method

        Parameters
        ----------
        threshold
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
            A list of slide definitions for the full report.
        """

        outputs = self.outputs

        dataset_id = self.dataset_metadata[0]["id"]

        model_id = self.model_metadata[0]["id"]

        return_key = self.outputs.return_key

        lowest_perturb_score = [perturber_output[return_key] for perturber_output in outputs.perturbations]

        final_dict = {
            "dataset": dataset_id,
            "model": model_id,
            self.config.perturber_factory.theta_key: self.config.perturber_factory.thetas,
            return_key: lowest_perturb_score,
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
                "deck": self.capability_id,
                "layout_name": "NRTKEvaluation",  # specialized template in gradient codebase
                "layout_arguments": {
                    "title": self.capability_id,
                    "data": df_perturbation,
                    "line_col": "line_id",
                    "x_data_col": self.config.perturber_factory.theta_key,
                    "y_data_col": return_key,
                    "perturbation_type": perturbation_label,
                    "lower_bound": 3.4,
                    "upper_bound": 5.3,
                    "model": model_id,
                    "plot_kwargs": {
                        "y_threshold_value": threshold,
                        "title": "NRTK Robustness Curve",
                        "x_label": f"{perturber} {PERTURBER_LABELS.get(theta_key, theta_key)}",
                        "y_label": METRIC_LABELS.get(return_key, return_key),
                    },
                },
            },
        ]


class NrtkAugmentationBase(
    CapabilityRunner[NrtkAugmentationOutputs, TDataset, TModel, TMetric, NrtkAugmentationConfig]
):
    """Perform augmentation on images in a dataset based on a given configuration."""

    _RUN_TYPE = NrtkAugmentationRun

    @classmethod
    def _create_config(cls) -> NrtkAugmentationConfig:
        return NrtkAugmentationConfig()

    @property
    def supports_datasets(self) -> Number:
        """Number of datasets this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating dataset support.
        """
        return Number.ONE

    @property
    def supports_models(self) -> Number:
        """Number of models this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating model support.
        """
        return Number.ONE

    @property
    def supports_metrics(self) -> Number:
        """Number of metrics this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating metric support.
        """
        return Number.ONE
