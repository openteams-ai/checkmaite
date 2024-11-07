"""Baseline Evaluation base class"""

from typing import Any, Union

from maite.workflows import evaluate

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
from jatic_ri.util.utils import create_metrics_bar_plot, save_figure_to_tempfile


def as_float(value: Union[float, str]) -> float:
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
):
    """Baseline evaluation implementation of TestStage interface with single model, dataset and metric plugins

    Parameters
    ----------

    Inherited attributes:
        outputs: Optional[TData]
        cache: Optional[Cache[TData]] = None
        cache_base_path: str = ".tscache"
        use_cache: bool = False
        model: od.Model
        model_id: str
        dataset: od.Dataset
        dataset_id: str
        metric: od.Metric
        metric_id: str
        threshold: float
    """

    metric_key: str
    _deck: str

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

    @property
    def cache_id(self) -> str:
        """Cache file for Baseline Evaluation Test Stage"""
        return f"baseline-{self.model_id}-{self.dataset_id}.json"

    def _run(self) -> None:
        """Run the test stage, and store any outputs of the evaluation in test stage"""
        self._validate()

        self.metric_key = self.metric._return_key  # noqa: SLF001

        # run evaluation
        result, _, _ = evaluate(
            model=self.model,
            metric=self.metric,
            dataset=self.dataset,
            return_preds=True,
        )

        if result is None:
            raise RuntimeError(
                f'Maite evaluate method returned no results for model ID "{self.model_id}", \
                dataset ID "{self.dataset_id}", and metric ID "{self.metric_id}"',
            )

        # convert tensors to floats, this return is set into self.outputs
        return {k: as_float(v) for k, v in result.items()}

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

        if self.outputs is None:
            raise Exception("No clean result computed or loaded before call to `collect_report_consumables`")

        # create bar plot of metric results
        fig = create_metrics_bar_plot(self.outputs, metric_key=self.metric_key, threshold=self.threshold)
        # save to tempfile
        image_path = save_figure_to_tempfile(fig)

        text = ""
        text += f"**Model**: {self.model_id} \n\n"
        text += f"**Dataset**: {self.dataset_id} \n\n"
        text += f"**{self.metric_key}**: {self.outputs[self.metric_key]}"

        return [
            {
                "deck": self._deck,
                "layout_name": "OneImageText",
                "layout_arguments": {
                    "title": "Basic Evaluation with MAITE",
                    "text": text,
                    "image_path": image_path,
                },
            },
        ]
