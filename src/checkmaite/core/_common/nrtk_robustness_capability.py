import contextlib
import copy
import hashlib
import importlib.metadata
import json
import math
import re
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from nrtk.interfaces import PerturbFactory
from pydantic import Field
from smqtk_core.configuration import from_config_dict

from checkmaite.core._utils import (
    CHECKMAITE_PLUGINS_UNSUPPORTED_INSTALL_HINT,
    deprecated,
    requires_optional_dependency,
)
from checkmaite.core.analytics_store._schema import BaseRecord
from checkmaite.core.cached_tasks import evaluate
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
from checkmaite.core.metric_keys import numeric_value, resolve_metric_key, undefined_to_none
from checkmaite.core.report import InlineTextReport
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


def _default_perturber_factory() -> PerturbFactory:
    return from_config_dict(
        {
            "type": "nrtk.impls.perturb_factory.PerturberOneStepFactory",
            "nrtk.impls.perturb_factory.PerturberOneStepFactory": {
                "perturber": "nrtk.impls.perturb_image.photometric.enhance.BrightnessPerturber",
                "theta_key": "factor",
                "theta_value": 10.0,
            },
        },
        PerturbFactory.get_impls(),
    )


class NrtkRobustnessConfig(CapabilityConfigBase):
    name: str = "natural_robustness_test_factory"
    perturber_factory: PerturbFactory = Field(default_factory=_default_perturber_factory)
    metric_key: str | None = Field(
        default=None,
        description=(
            "Key of the metric result to plot, for example 'map_50'. Needed only when the metric returns "
            "several numbers; if it returns one, that one is used."
        ),
    )


def _theta_key(factory: PerturbFactory) -> str:
    """Name of the perturber parameter the factory sweeps.

    NRTK 1.1 deprecates the ``theta_key`` attribute in favour of ``get_config()``.
    """
    return factory.get_config()["theta_key"]


def _augment_id(perturber: object) -> str:
    """Cache ID for the augmentation that applies one NRTK perturber.

    The ID hashes the perturber's full class path, its ``get_config()`` and the
    exact installed NRTK and pyBSM versions. CheckMAITE builds this ID, not the
    user, so it must change whenever the perturbed images can change, including
    when NRTK is upgraded. The full version is used because even a patch release
    can change a perturbation.
    """
    cls = type(perturber)
    properties = {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "config": perturber.get_config(),  # type: ignore[attr-defined]
        "versions": {name: _installed_version(name) for name in ("nrtk", "pybsm")},
    }
    return hashlib.sha256(json.dumps(properties, sort_keys=True, default=_exact_json).encode()).hexdigest()


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _exact_json(value: Any) -> Any:
    """JSON form of a config value that ``json`` cannot encode, without losing information."""
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    raise TypeError(f"Cannot build a cache ID from a perturber config value of type {type(value).__name__}.")


def _is_per_class(metric_key: str) -> bool:
    return metric_key == "classes" or metric_key.endswith("_per_class")


def _perturber_class(factory: PerturbFactory) -> str:
    """Short class name of the factory's perturber, e.g. ``"BrightnessPerturber"``."""
    return factory.get_config()["perturber"].rpartition(".")[-1]


def _words(class_name: str) -> str:
    """Split a CamelCase class name into words, e.g. ``"BrightnessPerturber"`` -> ``"Brightness Perturber"``."""
    return " ".join(re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", class_name))


def _to_float(value: Any) -> Any:
    """Convert a metric value (a tensor, NumPy scalar or number) to a Python float where possible."""
    import torch

    if isinstance(value, torch.Tensor):
        try:
            return value.item()
        except (RuntimeError, TypeError):
            return float(value.detach().cpu().numpy())
    with contextlib.suppress(TypeError, ValueError, OverflowError):
        return float(value)
    return value


def _snapshot(value: Any) -> Any:
    """Copy a metric value so later evaluations cannot change it, with undefined values as ``None``.

    A metric may return arrays or tensors that share its internal state, which
    the next ``reset`` and ``update`` overwrite in place.
    """
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().clone()
    else:
        # An uncopyable value is kept as returned.
        with contextlib.suppress(Exception):
            value = copy.deepcopy(value)
    return undefined_to_none(value)


class NrtkRobustnessPoint(CapabilityOutputsBase):
    """Metric results at one perturbation level.

    Attributes
    ----------
    theta
        Value of the perturber parameter at this level, for example a
        brightness factor of ``0.5``.
    metrics
        The metric's results on the dataset perturbed at this level, as
        returned by the metric's ``compute()``.
    """

    theta: float
    metrics: dict[str, Any]


class NrtkRobustnessOutputs(CapabilityOutputsBase):
    """Results of an NRTK robustness sweep.

    Attributes
    ----------
    perturber
        Short class name of the perturber, for example ``"BrightnessPerturber"``.
    theta_key
        Name of the perturber parameter the sweep varies, for example
        ``"factor"``.
    baseline
        The metric's results on the original, unperturbed dataset. Compare
        each point with it to see how much performance drops.
    points
        The metric's results at each perturbation level, in sweep order.
    metric_key
        Key of the plotted result in ``baseline`` and in each point's
        ``metrics``, for example ``"map_50"``: the config's ``metric_key``,
        or the metric's only numeric result.
    """

    perturber: str
    theta_key: str
    baseline: dict[str, Any]
    points: list[NrtkRobustnessPoint]
    metric_key: str


class NrtkRobustnessRecord(BaseRecord, table_name="nrtk_robustness"):
    """Record for NrtkRobustness capability results.

    One record is emitted per (theta_value, metric_key) pair for every
    result that is a single number at that level and on the baseline;
    results such as per-class arrays are skipped. This enables
    direct SQL reconstruction of robustness curves. The ``is_primary``
    flag marks rows for the plotted result, the outputs' ``metric_key``.

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
    baseline_value : float
        Score for the same ``metric_key`` on the unperturbed dataset, so
        ``baseline_value - metric_value`` is the drop at this level.
    is_primary : bool
        True when ``metric_key`` is the plotted result.
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
    baseline_value: float  # score for metric_key on the unperturbed dataset
    is_primary: bool  # True when metric_key is the plotted result


class NrtkRobustnessRun(CapabilityRunBase[NrtkRobustnessConfig, NrtkRobustnessOutputs]):
    config: NrtkRobustnessConfig
    outputs: NrtkRobustnessOutputs

    def extract(self) -> list[NrtkRobustnessRecord]:
        """Extract per-perturbation-point metrics from this NrtkRobustness run.

        Iterates all perturbation points and metric keys in the outputs,
        converting each value and its baseline to a flat scalar record.
        Results that are not single numbers are skipped, and so are
        ``classes`` and ``*_per_class`` results, which torchmetrics returns as
        single numbers for one class (or ``-1`` when per-class metrics are off).
        """
        # Single dataset/model/metric (Number.ONE)
        dataset_id = self.dataset_metadata[0]["id"]
        model_id = self.model_metadata[0]["id"]
        metric_id = self.metric_metadata[0]["id"]
        outputs = self.outputs

        return [
            NrtkRobustnessRecord(
                run_uid=self.run_uid,
                dataset_id=dataset_id,
                model_id=model_id,
                metric_id=metric_id,
                perturber_class=outputs.perturber,
                perturber_type=_words(outputs.perturber),
                theta_key=outputs.theta_key,
                theta_index=theta_index,
                theta_value=point.theta,
                metric_key=metric_key,
                metric_value=value,
                baseline_value=baseline_value,
                is_primary=(metric_key == outputs.metric_key),
            )
            for theta_index, point in enumerate(outputs.points)
            for metric_key, raw_value in point.metrics.items()
            if not _is_per_class(metric_key)
            and (value := numeric_value(raw_value)) is not None
            and (baseline_value := numeric_value(outputs.baseline.get(metric_key))) is not None
        ]

    # The order is important
    @requires_optional_dependency("gradient", install_hint=CHECKMAITE_PLUGINS_UNSUPPORTED_INSTALL_HINT)
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
        metric_key = outputs.metric_key
        theta_key = outputs.theta_key
        thetas = [point.theta for point in outputs.points]

        df_perturbation = pd.DataFrame.from_dict(
            {
                "dataset": dataset_id,
                "model": model_id,
                theta_key: thetas,
                metric_key: [numeric_value(point.metrics[metric_key]) for point in outputs.points],
            }
        )
        df_perturbation["line_id"] = "Item-Response Curve"

        return [
            {
                "deck": self.capability_id,
                "layout_name": "NRTKEvaluation",  # specialized template in gradient codebase
                "layout_arguments": {
                    "title": "Natural Robustness Toolkit (NRTK)",
                    "data": df_perturbation,
                    "line_col": "line_id",
                    "x_data_col": theta_key,
                    "y_data_col": metric_key,
                    "perturbation_type": _words(outputs.perturber),
                    "lower_bound": min(thetas),
                    "upper_bound": max(thetas),
                    "model": model_id,
                    "plot_kwargs": {
                        "y_threshold_value": threshold,
                        "title": "NRTK Robustness Curve",
                        "x_label": f"{outputs.perturber.replace('Perturber', '')} "
                        f"{PERTURBER_LABELS.get(theta_key, theta_key)}",
                        "y_label": METRIC_LABELS.get(metric_key, metric_key),
                    },
                },
            },
        ]

    def collect_md_report(self, threshold: float) -> InlineTextReport:
        """Generate a Markdown report of the robustness curve and its baseline."""
        outputs = self.outputs
        dataset_id = self.dataset_metadata[0]["id"]
        model_id = self.model_metadata[0]["id"]
        metric_key = outputs.metric_key
        theta_key = outputs.theta_key

        # Human-readable labels for x/y axes
        theta_label = PERTURBER_LABELS.get(theta_key, theta_key)
        metric_label = METRIC_LABELS.get(metric_key, metric_key)
        perturbation_label = _words(outputs.perturber)

        thetas = [point.theta for point in outputs.points]
        # None marks a level where the metric is undefined.
        values = [numeric_value(point.metrics[metric_key]) for point in outputs.points]
        baseline = _to_float(outputs.baseline[metric_key])

        md = MarkdownOutput("Natural Robustness Toolkit (NRTK)")

        # High-level summary
        md.add_text(f"**Model**: {model_id}")
        md.add_text(f"**Dataset**: {dataset_id}")
        md.add_text(f"**Perturbation Type**: {perturbation_label}")
        md.add_text(f"**Metric**: {metric_label}")
        md.add_text(f"**Baseline (unperturbed)**: {baseline:.4f}")
        md.add_text(f"**Threshold**: {threshold}")
        md.add_blank_line()

        md.add_section("Configuration")
        md.add_metrics_list(
            {
                "Theta key": theta_key,
                "Lower bound": min(thetas),
                "Upper bound": max(thetas),
                "Metric key": metric_key,
            }
        )

        md.add_section("Perturbation Results")
        rows = [
            [str(theta), "undefined", "undefined"]
            if value is None
            else [str(theta), f"{value:.4f}", f"{value - baseline:+.4f}"]
            for theta, value in zip(thetas, values, strict=True)
        ]

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.axhline(y=threshold, linestyle="solid", color="red", linewidth=3, label="Threshold")
        ax.axhline(y=baseline, linestyle="dashed", color="gray", linewidth=2, label="Baseline (unperturbed)")
        ax.plot(thetas, [math.nan if value is None else value for value in values], label="Item-Response Curve")
        ax.set_xlabel(f"{outputs.perturber.replace('Perturber', '')} {theta_label}")
        ax.set_ylabel(metric_label)
        ax.set_title("NRTK Robustness Curve")
        ax.legend()

        md.add_text(
            f"This test seeks to evaluate the performance of Model ID {model_id} on dataset ID {dataset_id} as a "
            f"{perturbation_label} perturber is applied.  The {theta_label} varies from {min(thetas)} to {max(thetas)}."
            f" The dashed line is the model's {metric_label} on the unperturbed dataset ({baseline:.4f})."
            f" Model Performance below the red line indicates when a model has failed and should not be used."
        )
        md.add_embedded_image(save_figure_to_tempfile(fig), alt_text="NRTK Robustness Curve", remove_source=True)
        plt.close(fig)

        md.add_table(headers=[theta_label, metric_label, "Change from baseline"], rows=rows)

        md.add_provenance(self)

        return InlineTextReport(
            media_type="text/markdown",
            content=md.render(),
            filename=f"{self.capability_id}.md",
        )


class NrtkRobustnessBase(Capability[NrtkRobustnessOutputs, TDataset, TModel, TMetric, NrtkRobustnessConfig]):
    """Evaluate a model on the unperturbed dataset and at each level of an NRTK perturbation sweep."""

    _RUN_TYPE = NrtkRobustnessRun
    cache_dependencies = ("nrtk",)
    # NRTK's MAITE augmentation class for the task, which applies one perturber.
    _augmentation_type: ClassVar[type]

    def _run(
        self,
        models: list[TModel],
        datasets: list[TDataset],
        metrics: list[TMetric],
        config: NrtkRobustnessConfig,
        use_prediction_and_evaluation_cache: bool,
    ) -> NrtkRobustnessOutputs:
        """Evaluate the model on the unperturbed dataset, then at each perturbation level."""
        model, dataset, metric = models[0], datasets[0], metrics[0]
        factory = config.perturber_factory
        # Read before evaluating anything, so an unsupported factory fails at once.
        perturber_class, theta_key = _perturber_class(factory), _theta_key(factory)

        def evaluate_with(augmentation: Any) -> dict[str, Any]:
            results, _, _ = evaluate(
                model=model,
                dataset=dataset,
                metric=metric,
                batch_size=1,
                augmentation=augmentation,
                return_augmented_data=False,
                return_preds=False,
                use_cache=use_prediction_and_evaluation_cache,
                task=self.task,
            )
            # Copied, because the metric may return values that share state the next evaluation overwrites.
            return {key: _snapshot(value) for key, value in results.items()}

        baseline = evaluate_with(None)
        # Choose the plotted result from the baseline, so a missing or ambiguous key fails before the sweep.
        metric_key = resolve_metric_key(baseline, config.metric_key)
        points = [
            NrtkRobustnessPoint(
                theta=theta,
                metrics=evaluate_with(self._augmentation_type(augment=perturber, augment_id=_augment_id(perturber))),
            )
            for theta, perturber in zip(factory.thetas, factory, strict=True)
        ]
        return NrtkRobustnessOutputs(
            perturber=perturber_class,
            theta_key=theta_key,
            baseline=baseline,
            points=points,
            metric_key=metric_key,
        )

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
