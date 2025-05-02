"""Baseline Evaluation base class"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.util.cache import JSONCache, NumpyEncoder
from jatic_ri.util.utils import create_metrics_bar_plot, save_figure_to_tempfile


def as_float(value: float | str) -> float:
    """Ensure value is a float"""
    if not isinstance(value, float):
        value = float(value)
    return value


class BaselineEvaluationBase(
    TestStage[dict[str, float]],
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
        outputs: Optional[TData]
        cache: Optional[Cache[TData]] = None
        use_stage_cache: bool = False
        eval_tool: EvaluationTool
        model: gen.Model
        model_id: str
        dataset: gen.Dataset
        dataset_id: str
        metric: gen.Metric
        metric_id: str
        threshold: float
    """

    cache: Cache[dict[str, Any]] | None = JSONCache(encoder=NumpyEncoder, compress=True)

    def __init__(self) -> None:
        super().__init__()

    def _validate(self) -> None:
        """Ensure that all the attributes are properly set"""
        if self.model is None:
            raise Exception("self.model is None")

        if self.metric is None:
            raise Exception("self.metric is None")

        if self.dataset is None:
            raise Exception("self.dataset is None")

        if self.eval_tool is None:
            raise Exception("self.eval_tool is None")

    @property
    def _metric_key(self) -> str:
        if self.model is None:
            raise Exception("self.model is None")
        # Get the human readable return_key from a wrapped Metric if available, otherwise fallback to the metric_id
        return getattr(self.metric, "return_key", self.metric_id)

    @property
    def cache_id(self) -> str:
        """Cache file for Baseline Evaluation Test Stage"""
        return f"baseline-{self._task}-{self.model_id}-{self.dataset_id}-{self.metric_id}.json"

    def _run(self) -> dict[str, float]:
        """Run the test stage, and store any outputs of the evaluation in test stage"""
        self._validate()
        # Type checker does not assume "index2label" is present without this check... or if the check is in _validate()
        if "index2label" not in self.model.metadata:
            raise (KeyError("'index2label' not found in model metadata but is required by BaselineEvaluationTestStage"))
        result, _, _ = self.eval_tool.evaluate(
            model=self.model,
            model_id=self.model_id,
            metric=self.metric,
            metric_id=self.metric_id,
            dataset=self.dataset,
            dataset_id=self.dataset_id,
            return_preds=True,
        )

        if result is None:
            raise RuntimeError(
                f'Evaluate method returned no results for model ID "{self.model_id}", \
                dataset ID "{self.dataset_id}", and metric ID "{self.metric_id}"',
            )
        # convert tensors to floats, this return is set into self.outputs.  Append index2label reverse mapping for
        # displaying class names in Gradient slides
        return {k: as_float(v) for k, v in result.items()} | {
            f"class_{v}": k for k, v in self.model.metadata["index2label"].items()
        }

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
        metric_key = self._metric_key

        if self.outputs is None:
            raise Exception("No clean result computed or loaded before call to `collect_report_consumables`")

        text = ""
        text += f"*Model*: {self.model_id} \n\n"
        text += f"*Dataset*: {self.dataset_id} \n\n"
        text += f"*{metric_key}*: {self.outputs[metric_key]:.2f}"

        # Code path for classwise_map50_metric case
        if "per_class_flag" in self.outputs:
            class_metrics: dict[str, float] = {}
            missing_classes: list[str] = []
            # For every 'class_*' key, add metric for class_name: metric if it exists. Otherwise, add to missing_classes
            for k, v in self.outputs.items():
                if k.startswith("class_"):
                    # Extract the class label from the key
                    class_label = k.split("_")[1]
                    class_metric: float | None = self.outputs.get(str(int(v)))
                    if class_metric:
                        class_metrics.update({class_label: class_metric})
                    else:
                        missing_classes.append(class_label)

            fig = create_per_class_bar_plot(
                overall_metric_name=metric_key,
                overall_metric_value=self.outputs[metric_key],
                class_metrics=class_metrics,
                threshold=self.threshold,
            )
            text += "\n\n\nClasses present in the model index but not the test dataset:\n"
            for missing_class in missing_classes:
                text += f"\\* {missing_class}\n"

        # Code path for all other metrics
        else:
            # Remove the class labels from outputs and create a bar plot of the other results
            metrics = {k: v for k, v in self.outputs.items() if "class_" not in k}
            fig = create_metrics_bar_plot(metrics, metric_key=metric_key, threshold=self.threshold, width=0.4)
        # save to tempfile
        image_path = save_figure_to_tempfile(fig)

        # non-styling related underscores are not permitted by gradient
        # this doesn't account for underscores that have already been escaped!
        # and removes even the styling related underscores :/
        text = text.replace("_", r"\_")

        return [
            {
                "deck": self._deck,
                "layout_name": "OneImageText",
                "layout_arguments": {
                    "title": "Basic Evaluation with MAITE",
                    "text": text,
                    "image_path": Path(image_path),
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
