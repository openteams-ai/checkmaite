import maite.protocols.image_classification as ic

from checkmaite.core._common.maite_evaluation_capability import MaiteEvaluationBase, MaiteEvaluationConfig


class MaiteEvaluation(MaiteEvaluationBase[ic.Dataset, ic.Model, ic.Metric]):
    """Evaluate one image-classification model and dataset with one or more metrics."""

    @classmethod
    def _create_config(cls) -> MaiteEvaluationConfig:
        return MaiteEvaluationConfig()
