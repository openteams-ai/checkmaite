import maite.protocols.image_classification as ic

from jatic_ri.core._common.maite_evaluation_capability import MaiteEvaluationBase


class MaiteEvaluation(MaiteEvaluationBase[ic.Dataset, ic.Model, ic.Metric]):
    """Baseline evaluation implementation with single model, dataset and metric plugins"""
