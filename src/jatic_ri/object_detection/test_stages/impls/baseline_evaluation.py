"""Baseline Evaluation implementation"""

import maite.protocols.object_detection as od

from jatic_ri._common.test_stages.impls import BaselineEvaluationBase


class BaselineEvaluation(BaselineEvaluationBase[od.Model, od.Dataset, od.Metric]):
    """Baseline evaluation implementation of TestStage interface with single model, dataset and metric plugins

    Parameters
    ----------

    Inherited attributes:
        outputs: TData | None
        cache: Cache[TData] | None = None
        use_stage_cache: bool = False
        eval_tool: EvaluationTool
        model: od.Model
        model_id: str
        dataset: od.Dataset
        dataset_id: str
        metric: od.Metric
        metric_id: str
        threshold: float
    """

    _deck: str = "object_detection_model_evaluation"
    _task: str = "od"
