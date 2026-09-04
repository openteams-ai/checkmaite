from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pydantic import Field, model_validator
from typing_extensions import TypeVar

from checkmaite.core._common.metric_fanout import (
    MaiteEvaluationMetricError,
    _canonicalize_metrics,
    _MetricFanout,
)
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
from checkmaite.core.report import InlineTextReport
from checkmaite.core.report._markdown import MarkdownOutput
from checkmaite.core.report._plotting_utils import create_metrics_bar_plot, save_figure_to_tempfile


class MaiteEvaluationRecord(BaseRecord, table_name="maite_evaluation"):
    """Record for one numeric result produced by a metric.

    One record is emitted per numeric output of ``Metric.compute()``. Per-class
    breakdowns are stored as separate records with ``scope="class"`` and the
    class name in ``class_name``. Multi-metric runs emit records for every
    member under the same run UID, distinguished by ``metric_id``.
    """

    dataset_id: str
    model_id: str
    metric_id: str
    output_key: str
    output_value: float
    scope: str = "overall"
    class_name: str | None = None


class MaiteEvaluationConfig(CapabilityConfigBase):
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Number of dataset items to pass to the MAITE model per inference call.",
    )
    run_cache_schema_version: Literal[1] = 1


TMaiteEvaluationConfig = TypeVar(
    "TMaiteEvaluationConfig",
    bound=MaiteEvaluationConfig,
    default=MaiteEvaluationConfig,
)


class MaiteMetricResult(CapabilityOutputsBase):
    """Normalized result for one member of a MAITE evaluation.

    ``overall_metric_name`` is the metric's optional Checkmaite ``return_key``.
    It identifies the member's headline value within ``scalar_values``.
    """

    metric_id: str
    overall_metric_name: str | None
    result: dict[str, Any]
    scalar_values: dict[str, float]
    class_metrics: dict[str, float | None] | None

    @model_validator(mode="after")
    def _validate_overall_metric_name(self) -> "MaiteMetricResult":
        if self.overall_metric_name is not None and self.overall_metric_name not in self.scalar_values:
            raise ValueError("overall_metric_name must identify a value in scalar_values.")
        return self

    @property
    def overall_metric_value(self) -> float | None:
        if self.overall_metric_name is None:
            return None
        return self.scalar_values[self.overall_metric_name]


class MaiteEvaluationOutputs(CapabilityOutputsBase):
    """Results from one or more metrics, keyed by canonical metric ID."""

    metrics: dict[str, MaiteMetricResult]


def _numeric_scalar(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _extract_class_metrics(
    result: dict[str, Any],
    return_key: str | None,
    model_metadata: Mapping[str, Any],
) -> dict[str, float | None] | None:
    if "per_class_flag" not in result:
        return None

    result.pop("per_class_flag")
    index2label = model_metadata.get("index2label")
    if not isinstance(index2label, Mapping):
        raise ValueError("Per-class metric results require model metadata 'index2label'.")

    class_metrics: dict[str, float | None] = {}
    for index, label in index2label.items():
        value = result.pop(str(index), None)
        scalar = None if value is None else _numeric_scalar(value)
        if value is not None and scalar is None:
            raise TypeError(f"Per-class result for index {index!r} must be numeric or None.")
        # Preserve the established display-label behavior, including accepting
        # repeated labels without rejecting the evaluation.
        class_metrics[str(label)] = scalar

    if return_key is None:
        raise ValueError("Per-class metric results require the metric to define return_key.")
    if result.keys() != {return_key}:
        raise ValueError(
            "When 'per_class_flag' is included, the metric must return one overall value plus class values, "
            f"but got {', '.join(sorted(result))}."
        )
    return class_metrics


def _normalize_metric_result(
    metric: Any,
    raw_result: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
) -> MaiteMetricResult:
    metric_id = metric.metadata["id"]
    try:
        result = dict(raw_result)
        return_key = getattr(metric, "return_key", None)
        if return_key is not None and (type(return_key) is not str or not return_key):
            raise TypeError("Metric return_key must be a non-empty string when defined.")

        class_metrics = _extract_class_metrics(result, return_key, model_metadata)
        scalar_values = {key: scalar for key, value in result.items() if (scalar := _numeric_scalar(value)) is not None}
        if return_key is not None and return_key not in scalar_values:
            raise ValueError(f"Metric return_key {return_key!r} does not identify a numeric top-level result.")

        return MaiteMetricResult(
            metric_id=metric_id,
            overall_metric_name=return_key,
            result=result,
            scalar_values=scalar_values,
            class_metrics=class_metrics,
        )
    except MaiteEvaluationMetricError:
        raise
    except Exception as error:
        raise MaiteEvaluationMetricError(
            metric_id=metric_id,
            stage="normalize",
            error_type=type(error).__name__,
            message=str(error),
        ) from error


class MaiteEvaluationRun(CapabilityRunBase[TMaiteEvaluationConfig, MaiteEvaluationOutputs]):
    config: TMaiteEvaluationConfig
    outputs: MaiteEvaluationOutputs

    @staticmethod
    def compute_uid(
        capability_id: str,
        config: TMaiteEvaluationConfig,
        dataset_metadata: Sequence[Mapping[str, Any]],
        model_metadata: Sequence[Mapping[str, Any]],
        metric_metadata: Sequence[Mapping[str, Any]],
    ) -> str:
        """Compute an order-independent UID for the metric collection."""
        return CapabilityRunBase.compute_uid(
            capability_id=capability_id,
            config=config,
            dataset_metadata=dataset_metadata,
            model_metadata=model_metadata,
            metric_metadata=sorted(metric_metadata, key=lambda metadata: metadata["id"]),
        )

    @requires_optional_dependency("gradient", install_hint=CHECKMAITE_PLUGINS_UNSUPPORTED_INSTALL_HINT)
    @deprecated(replacement="collect_md_report")
    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:  # pragma: no cover
        """Return one legacy Gradient report slide per metric."""
        slides: list[dict[str, Any]] = []
        for metric_id, metric_result in self.outputs.metrics.items():
            text = f"*Model*: {self.model_metadata[0]['id']} \n\n"
            text += f"*Dataset*: {self.dataset_metadata[0]['id']} \n\n"
            text += f"*Metric*: {metric_id}"
            figure: Figure | None = None

            if metric_result.overall_metric_name is not None:
                text += f"\n\n*{metric_result.overall_metric_name}*: {metric_result.overall_metric_value:.2f}"

            if metric_result.class_metrics is not None and metric_result.overall_metric_name is not None:
                class_metrics = {key: value for key, value in metric_result.class_metrics.items() if value is not None}
                missing_classes = [key for key, value in metric_result.class_metrics.items() if value is None]
                figure = create_per_class_bar_plot(
                    overall_metric_name=metric_result.overall_metric_name,
                    overall_metric_value=cast(float, metric_result.overall_metric_value),
                    class_metrics=class_metrics,
                    threshold=threshold,
                )
                if missing_classes:
                    text += "\n\n\nClasses present in the model index but not the test dataset:\n"
                    text += "".join(f"\\* {class_name}\n" for class_name in missing_classes)
            elif metric_result.scalar_values:
                highlighted_key = metric_result.overall_metric_name or next(iter(metric_result.scalar_values))
                figure = create_metrics_bar_plot(
                    metric_result.scalar_values,
                    metric_key=highlighted_key,
                    threshold=threshold,
                    width=0.4,
                )

            if figure is None:
                placeholder_figure, axes = plt.subplots()
                axes.axis("off")
                axes.text(0.5, 0.5, "No numeric top-level values", ha="center", va="center")
                placeholder_figure.tight_layout()
                plt.close(placeholder_figure)
                figure = placeholder_figure

            layout_arguments: dict[str, Any] = {
                "title": f"Basic Evaluation with MAITE: {metric_id}",
                "text": text,
                "item": Path(save_figure_to_tempfile(figure)),
            }
            slides.append(
                {
                    "deck": self.capability_id,
                    "layout_name": "ItemByNarrowText",
                    "layout_arguments": layout_arguments,
                }
            )
        return slides

    def collect_md_report(self, threshold: float) -> InlineTextReport:
        md = MarkdownOutput(title="Basic Evaluation with MAITE")
        md.add_section(heading="Model Evaluation Summary")
        md.add_text(f"**Model**: {self.model_metadata[0]['id']}")
        md.add_text(f"**Dataset**: {self.dataset_metadata[0]['id']}")

        for metric_id, metric_result in self.outputs.metrics.items():
            md.add_subsection(metric_id)
            if metric_result.overall_metric_name is not None:
                md.add_text(f"**{metric_result.overall_metric_name}**: {metric_result.overall_metric_value:.2f}")

            if metric_result.scalar_values:
                md.add_table(
                    headers=["Output", "Value"],
                    rows=[[key, f"{value:.4f}"] for key, value in metric_result.scalar_values.items()],
                )
            else:
                md.add_text("This metric returned no numeric top-level values.")

            if metric_result.class_metrics is not None and metric_result.overall_metric_name is not None:
                class_metrics = {key: value for key, value in metric_result.class_metrics.items() if value is not None}
                missing_classes = [key for key, value in metric_result.class_metrics.items() if value is None]
                figure = create_per_class_bar_plot(
                    overall_metric_name=metric_result.overall_metric_name,
                    overall_metric_value=cast(float, metric_result.overall_metric_value),
                    class_metrics=class_metrics,
                    threshold=threshold,
                )
                md.add_image(save_figure_to_tempfile(figure), alt_text=f"Per-Class Metrics: {metric_id}")
                if missing_classes:
                    md.add_subsection(f"Classes Missing for {metric_id}")
                    md.add_bulleted_list(missing_classes)
            elif metric_result.scalar_values:
                highlighted_key = metric_result.overall_metric_name or next(iter(metric_result.scalar_values))
                figure = create_metrics_bar_plot(
                    metric_result.scalar_values,
                    metric_key=highlighted_key,
                    threshold=threshold,
                    width=0.4,
                )
                md.add_image(save_figure_to_tempfile(figure), alt_text=f"Overall Metrics: {metric_id}")

        return InlineTextReport(
            media_type="text/markdown",
            content=md.render(),
            filename=f"{self.capability_id}.md",
        )

    def extract(self) -> list[MaiteEvaluationRecord]:
        """Extract numeric overall and per-class values for every metric."""
        dataset_id = self.dataset_metadata[0]["id"]
        model_id = self.model_metadata[0]["id"]
        records: list[MaiteEvaluationRecord] = []

        for metric_id, metric_result in self.outputs.metrics.items():
            for key, value in metric_result.scalar_values.items():
                records.append(
                    MaiteEvaluationRecord(
                        run_uid=self.run_uid,
                        dataset_id=dataset_id,
                        model_id=model_id,
                        metric_id=metric_id,
                        output_key=key,
                        output_value=value,
                        scope="overall",
                    )
                )

            if metric_result.class_metrics and metric_result.overall_metric_name is not None:
                for class_name, value in metric_result.class_metrics.items():
                    if value is not None:
                        records.append(
                            MaiteEvaluationRecord(
                                run_uid=self.run_uid,
                                dataset_id=dataset_id,
                                model_id=model_id,
                                metric_id=metric_id,
                                output_key=metric_result.overall_metric_name,
                                output_value=value,
                                scope="class",
                                class_name=class_name,
                            )
                        )

        return records


class MaiteEvaluationBase(
    Capability[MaiteEvaluationOutputs, TDataset, TModel, TMetric, TMaiteEvaluationConfig],
):
    """Evaluate one model and dataset with one or more metrics."""

    _RUN_TYPE = MaiteEvaluationRun

    @classmethod
    @abstractmethod
    def _create_config(cls) -> TMaiteEvaluationConfig:
        """Create the task-specific evaluation configuration."""

    @property
    def supports_datasets(self) -> Number:
        return Number.ONE

    @property
    def supports_models(self) -> Number:
        return Number.ONE

    @property
    def supports_metrics(self) -> Number:
        return Number.MANY

    def run(
        self,
        models: list[TModel] | None = None,
        datasets: list[TDataset] | None = None,
        metrics: list[TMetric] | None = None,
        config: TMaiteEvaluationConfig | None = None,
        use_cache: bool = True,
    ) -> CapabilityRunBase[TMaiteEvaluationConfig, MaiteEvaluationOutputs]:
        """Canonicalize metric IDs before capability UID and cache handling."""
        canonical_metrics = _canonicalize_metrics(metrics or [])
        return super().run(
            models=models,
            datasets=datasets,
            metrics=canonical_metrics,
            config=config,
            use_cache=use_cache,
        )

    def _run(
        self,
        models: list[TModel],
        datasets: list[TDataset],
        metrics: list[TMetric],
        config: TMaiteEvaluationConfig,
        use_prediction_and_evaluation_cache: bool,
    ) -> MaiteEvaluationOutputs:
        model = models[0]
        dataset = datasets[0]
        metric_fanout = _MetricFanout(metrics)
        cpu_prediction_postprocessor, cpu_prediction_postprocessor_id = self._cpu_prediction_postprocessor(config)
        result, _, _ = evaluate(
            model=model,
            metric=metric_fanout,
            dataset=dataset,
            batch_size=config.batch_size,
            cpu_prediction_postprocessor=cpu_prediction_postprocessor,
            cpu_prediction_postprocessor_id=cpu_prediction_postprocessor_id,
            return_augmented_data=False,
            return_preds=False,
            use_cache=use_prediction_and_evaluation_cache,
        )
        if result is None:
            raise RuntimeError(
                f'Evaluate returned no results for model ID {model.metadata["id"]!r} and '
                f'dataset ID {dataset.metadata["id"]!r}.'
            )

        normalized_results: dict[str, MaiteMetricResult] = {}
        for metric in metrics:
            metric_id = metric.metadata["id"]
            raw_metric_result = result.get(metric_id)
            if not isinstance(raw_metric_result, Mapping):
                error = TypeError("The fanout evaluation did not return a mapping for this metric.")
                raise MaiteEvaluationMetricError(
                    metric_id=metric_id,
                    stage="normalize",
                    error_type=type(error).__name__,
                    message=str(error),
                ) from error
            normalized_results[metric_id] = _normalize_metric_result(metric, raw_metric_result, model.metadata)

        return MaiteEvaluationOutputs(metrics=normalized_results)

    def _cpu_prediction_postprocessor(
        self,
        _config: TMaiteEvaluationConfig,
    ) -> tuple[Callable[[Any], Any] | None, str | None]:
        """Return optional CPU prediction postprocessing and its stable cache identity."""
        return None, None


def create_per_class_bar_plot(
    overall_metric_name: str, overall_metric_value: float, class_metrics: dict[str, float], threshold: float
) -> Figure:
    """Create a bar plot for per-class metrics alongside overall metric and threshold."""
    bar_color = "blue"
    threshold_line_color = "red"
    overall_line_color = "orange"
    fig, ax = plt.subplots(figsize=(9, 5))
    index = np.arange(len(class_metrics))

    for idx, (_, value) in enumerate(class_metrics.items()):
        ax.bar(index[idx], value, color=bar_color)

    ax.axhline(threshold, color=threshold_line_color)
    plt.text(len(index), threshold, "Threshold", va="center", color=threshold_line_color)
    ax.axhline(overall_metric_value, color=overall_line_color)
    plt.text(len(index), overall_metric_value, f"Overall {overall_metric_name}", color=overall_line_color)
    ax.set_title(f"Class-wise Metric: {overall_metric_name}")
    ax.set_xticks(index, class_metrics.keys(), rotation=45, ha="right")
    fig.tight_layout()
    plt.close(fig)
    return fig
