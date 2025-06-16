"""Plugins to be used for image classification and object detection implementations of
TestStage classes. Implementations may use one or more plugins depending on the application.
"""

from typing import Generic, TypeVar

from maite._internals.protocols import generic as gen

from jatic_ri.util.evaluation import EvaluationTool

TDataset = TypeVar("TDataset", bound=gen.Dataset)
TMetric = TypeVar("TMetric", bound=gen.Metric)
TModel = TypeVar("TModel", bound=gen.Model)


# MODELS --------------------------------------------
class SingleModelPlugin(Generic[TModel]):
    """TestStage Plugin for loading a single, MAITE-compliant model.

    Attributes
    ----------
    model : TModel
        The MAITE-compliant model.
    model_id : str
        Identifier for the model.
    """

    model: TModel
    model_id: str

    def load_model(self, model: TModel, model_id: str) -> None:
        """Injest a pre-loaded MAITE-compliant model.

        Parameters
        ----------
        model : TModel
            The MAITE-compliant model to load.
        model_id : str
            Identifier for the model.
        """
        self.model = model
        self.model_id = model_id


class MultiModelPlugin(Generic[TModel]):
    """TestStage Plugin for loading multiple, MAITE-compliant models.

    Attributes
    ----------
    models : dict[str, TModel]
        A dictionary mapping model IDs to MAITE-compliant models.
    """

    models: dict[str, TModel]  # {model_id: model}

    def load_models(self, models: dict[str, TModel]) -> None:
        """Injest a pre-loaded MAITE-compliant set of models.

        Parameters
        ----------
        models : dict[str, TModel]
            A dictionary mapping model IDs to MAITE-compliant models.
        """
        self.models = models


# DATASETS -------------------------------------------
class SingleDatasetPlugin(Generic[TDataset]):
    """TestStage Plugin for loading a single, MAITE-compliant dataset.

    Attributes
    ----------
    dataset : TDataset
        The MAITE-compliant dataset.
    dataset_id : str
        Identifier for the dataset.
    """

    dataset: TDataset
    dataset_id: str

    def load_dataset(self, dataset: TDataset, dataset_id: str) -> None:
        """Injest a pre-loaded MAITE-compliant dataset.

        Parameters
        ----------
        dataset : TDataset
            The MAITE-compliant dataset to load.
        dataset_id : str
            Identifier for the dataset.
        """
        self.dataset = dataset
        self.dataset_id = dataset_id


class TwoDatasetPlugin(Generic[TDataset]):
    """TestStage Plugin for loading two, MAITE-compliant datasets.

    Attributes
    ----------
    dataset_1 : TDataset
        The first MAITE-compliant dataset.
    dataset_1_id : str
        Identifier for the first dataset.
    dataset_2 : TDataset
        The second MAITE-compliant dataset.
    dataset_2_id : str
        Identifier for the second dataset.
    """

    dataset_1: TDataset
    dataset_1_id: str
    dataset_2: TDataset
    dataset_2_id: str

    def load_datasets(
        self,
        dataset_1: TDataset,
        dataset_1_id: str,
        dataset_2: TDataset,
        dataset_2_id: str,
    ) -> None:
        """Injest a pre-loaded MAITE-compliant set of datasets.

        Parameters
        ----------
        dataset_1 : TDataset
            The first MAITE-compliant dataset.
        dataset_1_id : str
            Identifier for the first dataset.
        dataset_2 : TDataset
            The second MAITE-compliant dataset.
        dataset_2_id : str
            Identifier for the second dataset.
        """
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.dataset_1_id = dataset_1_id
        self.dataset_2_id = dataset_2_id


# METRICS AND THRESHOLD -------------------------------
class MetricPlugin(Generic[TMetric]):
    """TestStage Plugin for loading a metric.

    Attributes
    ----------
    metric : TMetric
        The MAITE-compliant metric.
    metric_id : str
        Identifier for the metric.
    """

    metric: TMetric
    metric_id: str

    def load_metric(self, metric: TMetric, metric_id: str) -> None:
        """Injest a pre-loaded, MAITE-compliant metric.

        Parameters
        ----------
        metric : TMetric
            The MAITE-compliant metric to load.
        metric_id : str
            Identifier for the metric.
        """
        self.metric = metric
        self.metric_id = metric_id


class ThresholdPlugin:
    """TestStage Plugin for loading a threshold.

    Attributes
    ----------
    threshold : float
        The threshold value.
    """

    threshold: float

    def load_threshold(self, threshold: float) -> None:
        """Set threshold for the test.

        Parameters
        ----------
        threshold : float
            The threshold value to set.
        """
        self.threshold = threshold


# WORKFLOWS -------------------------------------------
class EvalToolPlugin:
    """TestStage Plugin for loading an evaluation tool with a cache.

    Attributes
    ----------
    eval_tool : EvaluationTool
        The evaluation tool.
    """

    eval_tool: EvaluationTool

    def load_eval_tool(self, eval_tool: EvaluationTool) -> None:
        """Load tool for MAITE evaluate workflow with a shareable cache.

        Parameters
        ----------
        eval_tool : EvaluationTool
            The evaluation tool to load.
        """
        self.eval_tool = eval_tool
