"""Baseline Evaluation implementation"""

import maite.protocols.object_detection as od

from jatic_ri._common.test_stages.impls import BaselineEvaluationBase


class BaselineEvaluation(BaselineEvaluationBase[od.Dataset, od.Model, od.Metric]):
    """Baseline evaluation implementation of TestStage interface with single model, dataset and metric plugins

    Parameters
    ----------

    Inherited attributes:
        threshold: float
    """

    _deck: str = "object_detection_model_evaluation"
    _task: str = "od"
