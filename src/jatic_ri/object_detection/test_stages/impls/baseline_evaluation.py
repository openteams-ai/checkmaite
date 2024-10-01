"""Baseline Evaluation implementation"""

from typing import Any

from jatic_ri._common.test_stages.interfaces.test_stage import TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import (
    MetricPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    ThresholdPlugin,
)


class BaselineEvaluation(TestStage[bool], SingleModelPlugin, SingleDatasetPlugin, MetricPlugin, ThresholdPlugin):
    """Baseline evaluation implementation of TestStage interface with single model, dataset and metric plugins"""

    def _run(self) -> None:
        """TODO: actually define run here"""
        pass

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """TODO: actually define here"""
        return super().collect_report_consumables()
