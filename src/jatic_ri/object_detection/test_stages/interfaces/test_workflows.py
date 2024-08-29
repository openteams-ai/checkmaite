"""High level interfaces to be used as base classes for implementation"""

from jatic_ri._common.test_stages.interfaces.test_stage import TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import (
    MetricThresholdPlugin,
    MultiModelPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    TwoDatasetPlugin,
)


class ModelDatasetMetricThreshold(
    TestStage,
    SingleModelPlugin,
    SingleDatasetPlugin,
    MetricThresholdPlugin,
):
    """High level implementation of TestStage requiring one model, one dataset, a metric, and a threshold"""

    pass


class SingleDataset(
    TestStage,
    SingleDatasetPlugin,
):
    """High level implementation of TestStage requiring one dataset"""

    pass


class TwoDataset(
    TestStage,
    TwoDatasetPlugin,
):
    """High level implementation of TestStage requiring two dataset"""

    pass


class MultiModelSingleDataset(
    TestStage,
    MultiModelPlugin,
    SingleDatasetPlugin,
):
    """High level implementation of TestStage requiring multiple models and one dataset"""

    pass


class DatasetMetricThreshold(
    TestStage,
    SingleDatasetPlugin,
    MetricThresholdPlugin,
):
    """High level implementation of TestStage requiring one dataset, metric, and threshold"""

    pass
