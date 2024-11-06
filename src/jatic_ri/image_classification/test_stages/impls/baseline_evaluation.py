"""Baseline Evaluation implementation"""

from jatic_ri._common.test_stages.impls import BaselineEvaluationBase


class BaselineEvaluation(BaselineEvaluationBase):
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

    _deck: str = "image_classification_model_evaluation"
