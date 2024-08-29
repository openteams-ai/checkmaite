"""Baseline Evaluation implementation"""

from typing import Any

from jatic_ri.object_detection.test_stages.interfaces.test_workflows import ModelDatasetMetricThreshold


class BaselineEvaluation(ModelDatasetMetricThreshold):
    """Baseline evaluation implementation of ModelDatasetMetricThreshold interface"""

    def run(self, use_cache: bool = True) -> None:
        """TODO: actually define run here"""
        pass

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """TODO: actually define here"""
        return super().collect_report_consumables()
