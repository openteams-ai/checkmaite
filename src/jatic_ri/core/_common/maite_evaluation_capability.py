from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pydantic
from matplotlib.figure import Figure

from jatic_ri.core.cached_tasks import evaluate
from jatic_ri.core.capability_core import (
    CapabilityConfigBase,
    CapabilityRunBase,
    CapabilityRunner,
    Number,
    TDataset,
    TMetric,
    TModel,
)
from jatic_ri.core.report._plotting_utils import create_metrics_bar_plot, save_figure_to_tempfile


class MaiteEvaluationConfig(CapabilityConfigBase):
    pass


class MaiteEvaluationOutputs(pydantic.BaseModel):
    overall_metric_name: str
    result: dict[str, float]
    class_metrics: dict[str, float | None] | None

    @property
    def overall_metric_value(self) -> float:
        return self.result[self.overall_metric_name]


class MaiteEvaluationRun(CapabilityRunBase[MaiteEvaluationConfig, MaiteEvaluationOutputs]):
    config: MaiteEvaluationConfig
    outputs: MaiteEvaluationOutputs

    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:
        """Access data for Gradient report generation.

        Retrieves in-depth data produced during the `run` method or loaded
        from cache, formatted for Gradient slide creation.

        Parameters
        ----------
        threshold
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
            A list of dictionaries, where each dictionary represents a slide.
            Each dictionary must contain the following keys:
        """

        text = ""
        text += f"*Model*: {self.model_metadata[0]['id']} \n\n"
        text += f"*Dataset*: {self.dataset_metadata[0]['id']} \n\n"
        text += f"*{self.outputs.overall_metric_name}*: {self.outputs.overall_metric_value:.2f}"

        if self.outputs.class_metrics is not None:
            class_metrics: dict[str, float] = {}
            missing_classes: list[str] = []
            for k, v in self.outputs.class_metrics.items():
                if v is not None:
                    class_metrics[k] = v
                else:
                    missing_classes.append(k)

            fig = create_per_class_bar_plot(
                overall_metric_name=self.outputs.overall_metric_name,
                overall_metric_value=self.outputs.overall_metric_value,
                class_metrics=class_metrics,
                threshold=threshold,
            )
            text += "\n\n\nClasses present in the model index but not the test dataset:\n"
            for missing_class in missing_classes:
                text += f"\\* {missing_class}\n"

        else:
            fig = create_metrics_bar_plot(
                self.outputs.result, metric_key=self.outputs.overall_metric_name, threshold=threshold, width=0.4
            )

        return [
            {
                "deck": self.capability_id,
                "layout_name": "ItemByNarrowText",
                "layout_arguments": {
                    "title": "Basic Evaluation with MAITE",
                    "text": text,
                    "item": Path(save_figure_to_tempfile(fig)),
                },
            },
        ]


class MaiteEvaluationBase(CapabilityRunner[MaiteEvaluationOutputs, TDataset, TModel, TMetric, MaiteEvaluationConfig]):
    """Evaluation implementation of CapabilityRunner interface."""

    _RUN_TYPE = MaiteEvaluationRun

    @classmethod
    def _create_config(cls) -> MaiteEvaluationConfig:
        return MaiteEvaluationConfig()

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

    def _run(
        self,
        models: list[TModel],
        datasets: list[TDataset],
        metrics: list[TMetric],
        config: MaiteEvaluationConfig,  # noqa: ARG002
        use_prediction_and_evaluation_cache: bool,
    ) -> MaiteEvaluationOutputs:
        """Run the capability and store evaluation outputs.

        Returns
        -------
        The outputs of the evaluation.

        Raises
        ------
        RuntimeError
            If the evaluation returns no results or if per-class metrics
            are malformed.
        """

        model = models[0]
        model_id = model.metadata["id"]
        metric = metrics[0]
        metric_id = metric.metadata["id"]
        dataset = datasets[0]
        dataset_id = dataset.metadata["id"]

        result, _, _ = evaluate(
            model=model,
            metric=metric,
            dataset=dataset,
            dataset_id=dataset_id,
            return_augmented_data=True,
            use_cache=use_prediction_and_evaluation_cache,
        )
        if result is None:
            raise RuntimeError(
                f'Evaluate method returned no results for model ID "{model_id}", \
                dataset ID "{dataset_id}", and metric ID "{metric_id}"',
            )

        # MAITE dictates dict[str, Any] here so enforcing a conversion to float might not be possible
        result = {k: float(v) for k, v in result.items()}

        overall_metric_name = str(getattr(metric, "return_key", metric_id))

        # TODO: can we formalize this per_class logic as part of MAITE itself?
        if "per_class_flag" in result:
            del result["per_class_flag"]
            class_metrics = {
                label: result.pop(str(index), None)
                for index, label in model.metadata["index2label"].items()  # pyright: ignore[reportTypedDictNotRequiredAccess]
            }
            if result.keys() != {overall_metric_name}:
                raise RuntimeError(
                    f"When 'per_class_flag' is included in the results, the metric should be a single value, "
                    f"but got {', '.join(sorted(result.keys()))}."
                )
        else:
            class_metrics = None

        return MaiteEvaluationOutputs(
            overall_metric_name=overall_metric_name, result=result, class_metrics=class_metrics
        )


def create_per_class_bar_plot(
    overall_metric_name: str, overall_metric_value: float, class_metrics: dict[str, float], threshold: float
) -> Figure:
    """Create a bar plot for per-class metrics alongside overall metric and threshold.

    Parameters
    ----------
    overall_metric_name
        Name of the overall metric (e.g., "Accuracy").
    overall_metric_value
        Value of the overall metric.
    class_metrics
        Dictionary mapping class names to their metric values.
    threshold
        Threshold value to be plotted as a horizontal line.

    Returns
    -------
    The matplotlib Figure object containing the plot.
    """
    bar_color = "blue"  # TODO: make these configurable with AppStyling
    threshold_line_color = "red"
    overall_line_color = "orange"

    # create initial canvas
    fig, ax = plt.subplots(figsize=(9, 5))

    # set up bar spacing
    index = np.arange(len(class_metrics))  # the x locations for the bars

    for idx, (_, value) in enumerate(class_metrics.items()):
        ax.bar(index[idx], value, color=bar_color)

    # plot and label the threshold line
    ax.axhline(threshold, color=threshold_line_color)  # TODO: make these configurable with AppStyling
    plt.text(len(index), threshold, "Threshold", va="center", color=threshold_line_color)
    # plot and label the overall metric line
    ax.axhline(overall_metric_value, color=overall_line_color)  # TODO: make these configurable with AppStyling
    plt.text(len(index), overall_metric_value, f"Overall {overall_metric_name}", color=overall_line_color)

    ax.set_title(f"Class-wise Metric: {overall_metric_name}")
    ax.set_xticks(index, class_metrics.keys(), rotation=45, ha="right")
    fig.tight_layout()

    return fig
