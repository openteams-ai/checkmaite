import maite.protocols.object_detection as od

from jatic_ri.core._common.maite_evaluation_capability import MaiteEvaluationBase


class MaiteEvaluation(MaiteEvaluationBase[od.Dataset, od.Model, od.Metric]):
    """Evaluation of a single model, dataset and metric plugins"""
