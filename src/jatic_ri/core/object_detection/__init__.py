import logging

from jatic_ri.core._common.dataeval_bias_capability import DataevalBiasConfig, DataevalBiasOutputs
from jatic_ri.core._common.dataeval_cleaning_capability import DataevalCleaningConfig, DataevalCleaningOutputs
from jatic_ri.core._common.dataeval_shift_capability import DataevalShiftConfig, DataevalShiftOutputs
from jatic_ri.core._common.maite_evaluation_capability import MaiteEvaluationConfig, MaiteEvaluationOutputs
from jatic_ri.core._common.nrtk_robustness_capability import NrtkRobustnessConfig, NrtkRobustnessOutputs
from jatic_ri.core.object_detection.dataeval_bias_capability import DataevalBias
from jatic_ri.core.object_detection.dataeval_cleaning_capability import DataevalCleaning
from jatic_ri.core.object_detection.dataeval_feasability_capability import (
    DataevalFeasibility,
    DataevalFeasibilityConfig,
    DataevalFeasibilityOutputs,
)
from jatic_ri.core.object_detection.dataeval_shift_capability import DataevalShift
from jatic_ri.core.object_detection.maite_evaluation_capability import MaiteEvaluation
from jatic_ri.core.object_detection.nrtk_robustness_capability import NrtkRobustness
from jatic_ri.core.object_detection.xaitk_explainable_capability import (
    XaitkExplainable,
    XaitkExplainableConfig,
    XaitkExplainableOutputs,
)

try:
    from jatic_ri.core.object_detection.reallabel_labelling_capability import (
        ReallabelLabelling,
        ReallabelLabellingConfig,
        ReallabelLabellingOutputs,
    )
except ImportError:
    logging.debug(
        "reallabel or its dependencies are not installed. Labelling will not be available. "
        "To use Labelling, please install reallabel and its dependencies."
    )

try:
    from jatic_ri.core._common.survivor_capability import SurvivorConfig, SurvivorOutputs
    from jatic_ri.core.object_detection.survivor_capability import Survivor
except ImportError:
    logging.debug(
        "survivor or its dependencies are not installed. Survivor will not be available. "
        "To use Survivor, please install survivor and its dependencies."
    )

try:
    from jatic_ri.core.object_detection.heart_adversarial_capability import (
        HeartAdversarial,
        HeartAdversarialAttackConfig,
        HeartAdversarialConfig,
        HeartAdversarialOutputs,
    )
except ImportError:
    logging.debug(
        "heart_library or its dependencies are not installed. HeartTestStage will not be available. "
        "To use HeartTestStage, please install heart_library and its dependencies."
    )

__all__ = [
    "MaiteEvaluation",
    "MaiteEvaluationConfig",
    "MaiteEvaluationOutputs",
    "DataevalBias",
    "DataevalBiasConfig",
    "DataevalBiasOutputs",
    "DataevalCleaning",
    "DataevalCleaningConfig",
    "DataevalCleaningOutputs",
    "DataevalFeasibility",
    "DataevalFeasibilityConfig",
    "DataevalFeasibilityOutputs",
    "DataevalShift",
    "DataevalShiftConfig",
    "DataevalShiftOutputs",
    "HeartAdversarialAttackConfig",
    "HeartAdversarialConfig",
    "HeartAdversarialOutputs",
    "HeartAdversarial",
    "NrtkRobustness",
    "NrtkRobustnessConfig",
    "NrtkRobustnessOutputs",
    "ReallabelLabelling",
    "ReallabelLabellingConfig",
    "ReallabelLabellingOutputs",
    "Survivor",
    "SurvivorConfig",
    "SurvivorOutputs",
    "XaitkExplainable",
    "XaitkExplainableConfig",
    "XaitkExplainableOutputs",
]
