import contextlib
import re
from typing import Any

import pandas as pd
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from pydantic import Field
from smqtk_core.configuration import from_config_dict

from checkmaite.core._utils import deprecated, requires_optional_dependency
from checkmaite.core.analytics_store._schema import BaseRecord
from checkmaite.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
    TDataset,
    TMetric,
    TModel,
)
from checkmaite.core.report._markdown import MarkdownOutput
from checkmaite.core.report._plotting_utils import save_figure_to_tempfile

PERTURBER_LABELS = {
    "factor": "Factor",
    "ksize": "Kernel Size",
    "s_x": "Jitter Amplitude - X (radians)",
    "s_y": "Jitter Amplitude - Y (radians)",
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
            "type": "nrtk.impls.perturb_image_factory.PerturberOneStepFactory",
            "nrtk.impls.perturb_image_factory.PerturberOneStepFactory": {
                "perturber": "nrtk.impls.perturb_image.photometric.enhance.BrightnessPerturber",
                "theta_key": "factor",
                "theta_value": 10.0,
            },
        },
        PerturbImageFactory.get_impls(),
    )


class NrtkRobustnessConfig(CapabilityConfigBase):
    name: str = "natural_robustness_test_factory"
    perturber_factory: PerturbImageFactory = Field(default_factory=_default_perturber_factory)


class NrtkRobustnessOutputs(CapabilityOutputsBase):
    perturbations: list[dict[str, Any]]
    return_key: str


class NrtkRobustnessRecord(BaseRecord, table_name="nrtk_robustness"):
    """Record for NrtkRobustness capability results.

    One record is emitted per (theta_value, metric_key) pair, enabling
    direct SQL reconstruction of robustness curves. The ``is_primary``
    flag marks rows corresponding to the capability's ``return_key``.

    Attributes
    ----------
    dataset_id : str
        Dataset identifier (cross-capability JOIN key).
    model_id : str
        Model identifier.
    metric_id : str
        Metric identifier.
    perturber_class : str
        Short class name of the perturber (e.g. ``"BrightnessPerturber"``).
    perturber_type : str
        Human-readable label (e.g. ``"Brightness Perturber"``).
    theta_key : str
        Perturbation parameter name (e.g. ``"factor"``, ``"ksize"``).
    theta_index : int
        Ordinal position in the sweep (0-based).
    theta_value : float
        Parameter value at this perturbation level.
    metric_key : str
        Metric output key (e.g. ``"accuracy"``, ``"f1_score"``).
    metric_value : float
        Score at this perturbation level.
    is_primary : bool
        True when ``metric_key`` matches the capability's ``return_key``.
    """

    # Cross-capability JOIN key (single-dataset convention)
    dataset_id: str

    # Entity identifiers
    model_id: str
    metric_id: str

    # Perturber identification
    perturber_class: str  # e.g. "BrightnessPerturber"
    perturber_type: str  # e.g. "Brightness Perturber" (human-readable)

    # Perturbation point
    theta_key: str  # e.g. "factor", "ksize"
    theta_index: int  # ordinal position in the sweep (0-based)
    theta_value: float  # parameter value at this point

    # Metric result
    metric_key: str  # e.g. "accuracy", "f1_score"
    metric_value: float  # score at this perturbation level
    is_primary: bool  # True when metric_key == return_key


class NrtkRobustnessRun(CapabilityRunBase[NrtkRobustnessConfig, NrtkRobustnessOutputs]):
    config: NrtkRobustnessConfig
    outputs: NrtkRobustnessOutputs

    def extract(self) -> list[NrtkRobustnessRecord]:
        """Extract per-perturbation-point metrics from this NrtkRobustness run.

        Iterates all theta points and metric keys from the perturbation
        outputs, converting each to a flat scalar record. Tensor values
        are coerced to Python floats via ``.item()`` with a fallback
        through ``.detach().cpu().numpy()``.
        """
        # Single dataset/model/metric (Number.ONE)
        dataset_id = self.dataset_metadata[0]["id"]
        model_id = self.model_metadata[0]["id"]
        metric_id = self.metric_metadata[0]["id"]
        return_key = self.outputs.return_key
        thetas = list(self.config.perturber_factory.thetas)
        theta_key = self.config.perturber_factory.theta_key

        # Extract short class name and split CamelCase into human-readable words
        # e.g. "nrtk...BrightnessPerturber" -> "BrightnessPerturber" -> "Brightness Perturber"
        perturbation_classname = self.config.perturber_factory.get_config()["perturber"].split(".")[-1]
        perturbation_label = " ".join(re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", perturbation_classname))

        import torch

        records: list[NrtkRobustnessRecord] = []
        for theta_index, (theta, perturber_output) in enumerate(zip(thetas, self.outputs.perturbations, strict=True)):
            for metric_key, raw_value in perturber_output.items():
                # Defensive tensor/numpy conversion (mirrors collect_md_report)
                if isinstance(raw_value, torch.Tensor):
                    try:
                        val = raw_value.item()
                    except (RuntimeError, TypeError):
                        val = float(raw_value.detach().cpu().numpy())
                else:
                    with contextlib.suppress(TypeError, ValueError, OverflowError):
                        raw_value = float(raw_value)
                    val = raw_value

                records.append(
                    NrtkRobustnessRecord(
                        run_uid=self.run_uid,
                        dataset_id=dataset_id,
                        model_id=model_id,
                        metric_id=metric_id,
                        perturber_class=perturbation_classname,
                        perturber_type=perturbation_label,
                        theta_key=theta_key,
                        theta_index=theta_index,
                        theta_value=float(theta),
                        metric_key=metric_key,
                        metric_value=val,
                        is_primary=(metric_key == return_key),
                    )
                )
        return records

    # The order is important
    @requires_optional_dependency("gradient", install_hint="pip install '.[unsupported]'")
    @deprecated(replacement="collect_md_report")
    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:  # pragma: no cover
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
        df_perturbation["line_id"] = "Item-Response Curve"

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
                    "title": "Natural Robustness Toolkit (NRTK)",
                    "data": df_perturbation,
                    "line_col": "line_id",
                    "x_data_col": self.config.perturber_factory.theta_key,
                    "y_data_col": return_key,
                    "perturbation_type": perturbation_label,
                    "lower_bound": min(self.config.perturber_factory.thetas),
                    "upper_bound": max(self.config.perturber_factory.thetas),
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

    def collect_md_report(self, threshold: float) -> str:
        """Generate Markdown-formatted report for NRTK perturbation analysis.

        This mirrors the semantics of collect_report_consumables:
        it uses the same return_key, theta_key, and perturbation configuration
        that feed the Gradient robustness curve.
        """
        outputs = self.outputs

        dataset_id = self.dataset_metadata[0]["id"]
        model_id = self.model_metadata[0]["id"]

        # Use the same key that collect_report_consumables uses for y-values
        return_key = outputs.return_key
        theta_key = self.config.perturber_factory.theta_key

        # Human-readable labels for x/y axes
        theta_label = PERTURBER_LABELS.get(theta_key, theta_key)
        metric_label = METRIC_LABELS.get(return_key, return_key)

        # Same perturbation label logic as collect_report_consumables
        perturbation_classname = self.config.perturber_factory.get_config()["perturber"].split(".")[-1]
        perturbation_label = " ".join(re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", perturbation_classname))

        # Extract the robustness curve data
        values: list[float] = []
        for perturber_output in outputs.perturbations:
            val = perturber_output[return_key]
            # Convert tensors / numpy scalars to Python float - avoid linting issues
            import torch

            if isinstance(val, torch.Tensor):
                try:
                    val = val.item()
                except (AttributeError, RuntimeError, TypeError):
                    val = float(val.detach().cpu().numpy())
            else:
                with contextlib.suppress(TypeError, ValueError, OverflowError):
                    val = float(val)
            values.append(val)

        md = MarkdownOutput("Natural Robustness Toolkit (NRTK)")

        # High-level summary
        md.add_text(f"**Model**: {model_id}")
        md.add_text(f"**Dataset**: {dataset_id}")
        md.add_text(f"**Perturbation Type**: {perturbation_label}")
        md.add_text(f"**Metric**: {metric_label}")
        md.add_text(f"**Threshold**: {threshold}")
        md.add_blank_line()

        md.add_section("Configuration")
        md.add_metrics_list(
            {
                "Theta key": theta_key,
                "Lower bound": min(self.config.perturber_factory.thetas),
                "Upper bound": max(self.config.perturber_factory.thetas),
                "Metric key": return_key,
            }
        )

        # Curve data: same x/y pairing as the DataFrame in collect_report_consumables
        md.add_section("Perturbation Results")
        thetas = list(self.config.perturber_factory.thetas)
        rows: list[list[str]] = []
        for theta, val in zip(thetas, values, strict=True):
            rows.append([str(theta), f"{val:.4f}"])

        perturber = self.config.perturber_factory.get_config()["perturber"].rpartition(".")[-1].replace("Perturber", "")

        # Generate the plot inline ()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # Plot the threshold line
        ax.axhline(
            y=threshold,
            linestyle="solid",
            color="red",
            linewidth=3,
        )

        # Plot the data line
        ax.plot(thetas, values, label="Item-Response Curve")

        # Set labels and title
        ax.set_xlabel(f"{perturber} {theta_label}")
        ax.set_ylabel(metric_label)
        ax.set_title("NRTK Robustness Curve")
        ax.legend()

        md.add_text(
            f"This test seeks to evaluate the performance of Model ID {model_id} on dataset ID {dataset_id} as a "
            f"{perturbation_label} perturber is applied.  The {theta_label} varies from {min(thetas)} to {max(thetas)}."
            f" Model Performance below the red line indicates when a model has failed and should not be used."
        )
        md.add_image(save_figure_to_tempfile(fig), alt_text="NRTK Robustness Curve")
        plt.close(fig)

        md.add_table(headers=[theta_label, metric_label], rows=rows)

        return md.render()


class NrtkRobustnessBase(Capability[NrtkRobustnessOutputs, TDataset, TModel, TMetric, NrtkRobustnessConfig]):
    """Perform augmentation on images in a dataset based on a given configuration."""

    _RUN_TYPE = NrtkRobustnessRun

    @classmethod
    def _create_config(cls) -> NrtkRobustnessConfig:
        return NrtkRobustnessConfig()

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
