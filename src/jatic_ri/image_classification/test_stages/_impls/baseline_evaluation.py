"""Baseline Evaluation implementation"""

import maite.protocols.image_classification as ic

from jatic_ri._common.test_stages.impls import BaselineEvaluationBase


class BaselineEvaluation(BaselineEvaluationBase[ic.Dataset, ic.Model, ic.Metric]):
    """Baseline evaluation implementation of TestStage interface with single model, dataset and metric plugins

    Parameters
    ----------

    Inherited attributes:
        threshold: float
    """

    _deck: str = "image_classification_model_evaluation"
    _task: str = "ic"
