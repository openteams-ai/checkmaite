"""Baseline Evaluation implementation"""

import maite.protocols.image_classification as ic

from jatic_ri._common.test_stages.impls import BaselineEvaluationBase


class BaselineEvaluation(BaselineEvaluationBase[ic.Model, ic.Dataset, ic.Metric]):
    """Baseline evaluation implementation of TestStage interface with single model, dataset and metric plugins

    Parameters
    ----------

    Inherited attributes:
        outputs: TData | None
        cache: Cache[TData] | None = None
        use_stage_cache: bool = False
        eval_tool: EvaluationTool
        model: ic.Model
        model_id: str
        dataset: ic.Dataset
        dataset_id: str
        metric: ic.Metric
        metric_id: str
        threshold: float
    """

    _deck: str = "image_classification_model_evaluation"
    _task: str = "ic"
