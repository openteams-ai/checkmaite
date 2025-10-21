"""Object detection test stages"""

import logging

# Import Config and Outputs classes from common test stages
from jatic_ri._common.test_stages.impls.baseline_evaluation_test_stage import (
    BaselineEvaluationOutputs,
)
from jatic_ri._common.test_stages.impls.dataeval_bias_test_stage import (
    DataevalBiasConfig,
    DataevalBiasOutputs,
)
from jatic_ri._common.test_stages.impls.dataeval_cleaning_test_stage import (
    DataevalCleaningOutputs,
)
from jatic_ri._common.test_stages.impls.dataeval_shift_test_stage import (
    DataevalShiftOutputs,
)
from jatic_ri._common.test_stages.impls.nrtk_test_stage import (
    NRTKTestStageConfig,
    NRTKTestStageOutputs,
)
from jatic_ri._common.test_stages.impls.survivor_test_stage import (
    SurvivorConfig,
    SurvivorOutputs,
)

# Import all test stages from internal _impls module
from jatic_ri.object_detection.test_stages._impls.baseline_evaluation import BaselineEvaluation
from jatic_ri.object_detection.test_stages._impls.dataeval_bias_test_stage import DatasetBiasTestStage
from jatic_ri.object_detection.test_stages._impls.dataeval_cleaning_test_stage import DatasetCleaningTestStage
from jatic_ri.object_detection.test_stages._impls.dataeval_feasibility_test_stage import (
    DatasetObjectDetectionFeasibilityConfig,
    DatasetObjectDetectionFeasibilityOutputs,
    DatasetObjectDetectionFeasibilityTestStage,
)
from jatic_ri.object_detection.test_stages._impls.dataeval_shift_test_stage import DatasetShiftTestStage
from jatic_ri.object_detection.test_stages._impls.nrtk_test_stage import NRTKTestStage
from jatic_ri.object_detection.test_stages._impls.reallabel_test_stage import (
    RealLabelConfig,
    RealLabelOutputs,
    RealLabelTestStage,
)
from jatic_ri.object_detection.test_stages._impls.survivor_test_stage import SurvivorTestStage
from jatic_ri.object_detection.test_stages._impls.xaitk_test_stage import (
    XAITKConfigOD,
    XAITKOutputsOD,
    XAITKTestStage,
)

# Check if optional heart_library dependencies are available at runtime
try:
    from jatic_ri.object_detection.test_stages._impls.heart_test_stage import (
        HeartAttackConfig,
        HeartOutputs,
        HeartTestStage,
    )
except ImportError:
    logging.debug(
        "heart_library or its dependencies are not installed. HeartTestStage will not be available. "
        "To use HeartTestStage, please install heart_library and its dependencies."
    )

__all__ = [
    "BaselineEvaluation",
    "BaselineEvaluationOutputs",
    "DataevalBiasConfig",
    "DataevalBiasOutputs",
    "DataevalCleaningOutputs",
    "DataevalShiftOutputs",
    "DatasetBiasTestStage",
    "DatasetCleaningTestStage",
    "DatasetObjectDetectionFeasibilityConfig",
    "DatasetObjectDetectionFeasibilityOutputs",
    "DatasetObjectDetectionFeasibilityTestStage",
    "DatasetShiftTestStage",
    "HeartAttackConfig",
    "HeartOutputs",
    "HeartTestStage",
    "NRTKTestStage",
    "NRTKTestStageConfig",
    "NRTKTestStageOutputs",
    "RealLabelConfig",
    "RealLabelOutputs",
    "RealLabelTestStage",
    "SurvivorConfig",
    "SurvivorOutputs",
    "SurvivorTestStage",
    "XAITKConfigOD",
    "XAITKOutputsOD",
    "XAITKTestStage",
]
