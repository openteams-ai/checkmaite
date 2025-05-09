"""Baseline Evaluation base class"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pydantic
from matplotlib.figure import Figure

from jatic_ri._common.test_stages.interfaces.plugins import (
    EvalToolPlugin,
    MetricPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    TDataset,
    ThresholdPlugin,
    TMetric,
    TModel,
)
from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, RunBase, TestStage
from jatic_ri.util.utils import create_metrics_bar_plot, save_figure_to_tempfile


class BaselineEvaluationConfig(ConfigBase):
    pass


class BaselineEvaluationOutputs(pydantic.BaseModel):
    overall_metric_name: str
    result: dict[str, float]
    class_metrics: Optional[dict[str, Optional[float]]]

    @property
    def overall_metric_value(self) -> float:
        return self.result[self.overall_metric_name]


class BaselineEvaluationRun(RunBase):
    config: BaselineEvaluationConfig
    outputs: BaselineEvaluationOutputs


class BaselineEvaluationBase(
    TestStage[BaselineEvaluationOutputs],
    SingleModelPlugin[TModel],
    SingleDatasetPlugin[TDataset],
    MetricPlugin[TMetric],
    ThresholdPlugin,
    EvalToolPlugin,
):
    """Baseline evaluation implementation of TestStage interface with single model, dataset and metric plugins

    Parameters
    ----------

    Inherited attributes:
        eval_tool: EvaluationTool
        model: gen.Model
        model_id: str
        dataset: gen.Dataset
        dataset_id: str
        metric: gen.Metric
        metric_id: str
        threshold: float
    """

    _RUN_TYPE = BaselineEvaluationRun

    def _create_config(self) -> ConfigBase:
        return BaselineEvaluationConfig()

    def _run(self) -> BaselineEvaluationOutputs:
        """Run the test stage, and store any outputs of the evaluation in test stage"""
        result, _, _ = self.eval_tool.evaluate(
            model=self.model,
            model_id=self.model_id,
            metric=self.metric,
            metric_id=self.metric_id,
            dataset=self.dataset,
            dataset_id=self.dataset_id,
        )
        if result is None:
            raise RuntimeError(
                f'Evaluate method returned no results for model ID "{self.model_id}", \
                dataset ID "{self.dataset_id}", and metric ID "{self.metric_id}"',
            )

        # MAITE dictates dict[str, Any] here so enforcing a conversion to float might not be possible
        result = {k: float(v) for k, v in result.items()}

        overall_metric_name = str(getattr(self.metric, "return_key", self.metric.metadata["id"]))

        if "per_class_flag" in result:
            del result["per_class_flag"]
            class_metrics = {
                label: result.pop(str(index), None)
                for index, label in self.model.metadata["index2label"].items()  # type: ignore[reportTypedDictNotRequiredAccess]
            }
            if result.keys() != {overall_metric_name}:
                raise RuntimeError(
                    f"When 'per_class_flag' is included in the results, the metric should be a single value, "
                    f"but got {', '.join(sorted(result.keys()))}."
                )
        else:
            class_metrics = None

        return BaselineEvaluationOutputs(
            overall_metric_name=overall_metric_name, result=result, class_metrics=class_metrics
        )

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report generated in the run method or in the
        load_cached_results method
        Please return a list of dictionaries, one dictionary per slide

        For each dictionary, please include the following keys:
        - "deck": (str) image_classification_model_evaluation, object_detection_model_evaluation,
                        image_classification_dataset_evaluation
        - "layout_name": (str) find the layout name in the jatic_increment_5_gradient_demo_repo, linked below
        https://gitlab.jatic.net/jatic/morse/jatic-increment-5-gradient-demo-repo/-/tree/main/src/jatic_increment_5_gradient_demo_repo/cards?ref_type=heads
        - "layout_arguments": (dict) arguments pertaining to the specific layout
        """
        run = self._stored_run
        if run is None:
            raise Exception("No clean result computed or loaded before call to `collect_report_consumables`")

        text = ""
        text += f"*Model*: {run.model_ids[0]} \n\n"
        text += f"*Dataset*: {run.dataset_ids[0]} \n\n"
        text += f"*{run.outputs.overall_metric_name}*: {run.outputs.overall_metric_value:.2f}"

        if run.outputs.class_metrics is not None:
            class_metrics: dict[str, float] = {}
            missing_classes: list[str] = []
            for k, v in run.outputs.class_metrics.items():
                if v is not None:
                    class_metrics[k] = v
                else:
                    missing_classes.append(k)

            fig = create_per_class_bar_plot(
                overall_metric_name=run.outputs.overall_metric_name,
                overall_metric_value=run.outputs.overall_metric_value,
                class_metrics=class_metrics,
                threshold=self.threshold,
            )
            text += "\n\n\nClasses present in the model index but not the test dataset:\n"
            for missing_class in missing_classes:
                text += f"\\* {missing_class}\n"

        else:
            fig = create_metrics_bar_plot(
                run.outputs.result, metric_key=run.outputs.overall_metric_name, threshold=self.threshold, width=0.4
            )

        return [
            {
                "deck": self._deck,
                "layout_name": "ItemByNarrowText",
                "layout_arguments": {
                    "title": "Basic Evaluation with MAITE",
                    "text": text,
                    "item": Path(save_figure_to_tempfile(fig)),
                },
            },
        ]


def create_per_class_bar_plot(
    overall_metric_name: str, overall_metric_value: float, class_metrics: dict[str, float], threshold: float
) -> Figure:
    bar_color = "blue"
    threshold_line_color = "red"
    overall_line_color = "orange"

    # create initial canvas
    fig, ax = plt.subplots(figsize=(9, 5))

    # set up bar spacing
    index = np.arange(len(class_metrics))  # the x locations for the bars

    for idx, (_, value) in enumerate(class_metrics.items()):
        ax.bar(index[idx], value, color=bar_color)

    # plot and label the threshold line
    ax.axhline(threshold, color=threshold_line_color)
    plt.text(len(index), threshold, "Threshold", va="center", color=threshold_line_color)
    # plot and label the overall metric line
    ax.axhline(overall_metric_value, color=overall_line_color)
    plt.text(len(index), overall_metric_value, f"Overall {overall_metric_name}", color=overall_line_color)

    ax.set_title(f"Class-wise Metric: {overall_metric_name}")
    ax.set_xticks(index, class_metrics.keys(), rotation=45, ha="right")
    fig.tight_layout()

    return fig
