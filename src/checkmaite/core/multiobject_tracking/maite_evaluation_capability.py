import maite.protocols.multiobject_tracking as mot

from checkmaite.core._common.maite_evaluation_capability import MaiteEvaluationBase, MaiteEvaluationConfig


class MaiteEvaluation(MaiteEvaluationBase[mot.Dataset, mot.Model, mot.Metric]):
    """Evaluate one multi-object-tracking model and dataset with one or more metrics."""

    @classmethod
    def _create_config(cls) -> MaiteEvaluationConfig:
        return MaiteEvaluationConfig()
