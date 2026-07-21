import maite.protocols.image_classification as ic

from checkmaite.core._common.maite_evaluation_capability import MaiteEvaluationBase, MaiteEvaluationConfig


class MaiteEvaluation(MaiteEvaluationBase[ic.Dataset, ic.Model, ic.Metric]):
    """Baseline evaluation implementation with single model, dataset and metric plugins"""

    @classmethod
    def _create_config(cls) -> MaiteEvaluationConfig:
        return MaiteEvaluationConfig()
