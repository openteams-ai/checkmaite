import maite.protocols.object_detection as od

from checkmaite.core._common.maite_evaluation_capability import MaiteEvaluationBase


class MaiteEvaluation(MaiteEvaluationBase[od.Dataset, od.Model, od.Metric]):
    """Evaluation of a single model, dataset and metric plugins"""
