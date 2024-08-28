"""Implementation of TestStage for one model, one dataset, a metric and threshold"""

from jatic_ri._common.test_stages.interfaces.test_stage import TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import (
    MetricThresholdPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
)


class ModelDatasetMetricThreshold(
    TestStage,
    SingleModelPlugin,
    SingleDatasetPlugin,
    MetricThresholdPlugin,
):
    """High level implementation of TestStage requiring one model, one dataset, a metric, and a threshold"""

    pass
