"""Image Classification test stages

This module provides a unified public API for all image classification test stages.

All test stages, their configuration classes, and output classes can be imported directly
from this module. Internal implementation details are in the _impls subpackage.

Example:
    >>> from jatic_ri.image_classification.test_stages import (
    ...     BaselineEvaluation,
    ...     NRTKTestStage,
    ...     NRTKTestStageConfig,
    ...     NRTKTestStageOutputs,
    ... )
"""

import logging

# Import Config and Outputs classes from common test stages
from jatic_ri._common.test_stages.impls.baseline_evaluation_test_stage import (
    BaselineEvaluationConfig,
    BaselineEvaluationOutputs,
)
from jatic_ri._common.test_stages.impls.dataeval_bias_test_stage import (
    DataevalBiasConfig,
    DataevalBiasOutputs,
)
from jatic_ri._common.test_stages.impls.dataeval_cleaning_test_stage import (
    DataevalCleaningConfig,
    DataevalCleaningOutputs,
)
from jatic_ri._common.test_stages.impls.dataeval_shift_test_stage import DataevalShiftConfig, DataevalShiftOutputs
from jatic_ri._common.test_stages.impls.nrtk_test_stage import NRTKTestStageConfig, NRTKTestStageOutputs

# Import all test stages from internal _impls module
from jatic_ri.image_classification.test_stages._impls.baseline_evaluation import BaselineEvaluation
from jatic_ri.image_classification.test_stages._impls.dataeval_bias_test_stage import DatasetBiasTestStage
from jatic_ri.image_classification.test_stages._impls.dataeval_cleaning_test_stage import (
    DatasetCleaningTestStage,
)
from jatic_ri.image_classification.test_stages._impls.dataeval_feasibility_test_stage import (
    DatasetImageClassificationFeasibilityConfig,
    DatasetImageClassificationFeasibilityOutputs,
    DatasetImageClassificationFeasibilityTestStage,
)
from jatic_ri.image_classification.test_stages._impls.dataeval_shift_test_stage import DatasetShiftTestStage
from jatic_ri.image_classification.test_stages._impls.nrtk_test_stage import NRTKTestStage
from jatic_ri.image_classification.test_stages._impls.xaitk_test_stage import (
    XAITKConfigIC,
    XAITKOutputsIC,
    XAITKTestStage,
)

# Check if optional survivor dependencies are available at runtime
try:
    from jatic_ri._common.test_stages.impls.survivor_test_stage import (
        SurvivorConfig,
        SurvivorOutputs,
    )
    from jatic_ri.image_classification.test_stages._impls.survivor_test_stage import SurvivorTestStage
except ImportError:
    logging.debug(
        "survivor or its dependencies are not installed. SurvivorTestStage will not be available. "
        "To use SurvivorTestStage, please install survivor and its dependencies."
    )

__all__ = [
    # BaselineEvaluation
    "BaselineEvaluation",
    "BaselineEvaluationConfig",
    "BaselineEvaluationOutputs",
    # DatasetBiasTestStage
    "DatasetBiasTestStage",
    "DataevalBiasConfig",
    "DataevalBiasOutputs",
    # DatasetCleaningTestStage
    "DatasetCleaningTestStage",
    "DataevalCleaningConfig",
    "DataevalCleaningOutputs",
    # DatasetImageClassificationFeasibilityTestStage
    "DatasetImageClassificationFeasibilityTestStage",
    "DatasetImageClassificationFeasibilityConfig",
    "DatasetImageClassificationFeasibilityOutputs",
    # DatasetShiftTestStage
    "DatasetShiftTestStage",
    "DataevalShiftConfig",
    "DataevalShiftOutputs",
    # NRTKTestStage
    "NRTKTestStage",
    "NRTKTestStageConfig",
    "NRTKTestStageOutputs",
    # SurvivorTestStage
    "SurvivorTestStage",
    "SurvivorConfig",
    "SurvivorOutputs",
    # XAITKTestStage
    "XAITKTestStage",
    "XAITKConfigIC",
    "XAITKOutputsIC",
]
