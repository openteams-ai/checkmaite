import logging

from jatic_ri.core._common.dataeval_bias_capability import DataevalBiasConfig, DataevalBiasOutputs
from jatic_ri.core._common.dataeval_cleaning_capability import DataevalCleaningConfig, DataevalCleaningOutputs
from jatic_ri.core._common.dataeval_shift_capability import DataevalShiftConfig, DataevalShiftOutputs
from jatic_ri.core._common.maite_evaluation_capability import MaiteEvaluationConfig, MaiteEvaluationOutputs
from jatic_ri.core._common.nrtk_augmentation_capability import NrtkAugmentationConfig, NrtkAugmentationOutputs
from jatic_ri.core.image_classification.dataeval_bias_capability import DataevalBias
from jatic_ri.core.image_classification.dataeval_cleaning_capability import DataevalCleaning
from jatic_ri.core.image_classification.dataeval_feasability_capability import (
    DataevalFeasibility,
    DataevalFeasibilityConfig,
    DataevalFeasibilityOutputs,
)
from jatic_ri.core.image_classification.dataeval_shift_capability import DataevalShift
from jatic_ri.core.image_classification.maite_evaluation_capability import MaiteEvaluation
from jatic_ri.core.image_classification.nrtk_augmentation_capability import NrtkAugmentation
from jatic_ri.core.image_classification.xaitk_explainable_capability import (
    XaitkExplainable,
    XaitkExplainableConfig,
    XaitkExplainableOutputs,
)

try:
    from jatic_ri.core._common.survivor_capability import (
        SurvivorConfig,
        SurvivorOutputs,
    )
    from jatic_ri.core.image_classification.survivor_capability import Survivor
except ImportError:
    logging.debug(
        "survivor or its dependencies are not installed. Survivor will not be available. "
        "To use Survivor, please install survivor and its dependencies."
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
    "NrtkAugmentation",
    "NrtkAugmentationConfig",
    "NrtkAugmentationOutputs",
    "Survivor",
    "SurvivorConfig",
    "SurvivorOutputs",
    "XaitkExplainable",
    "XaitkExplainableConfig",
    "XaitkExplainableOutputs",
]
