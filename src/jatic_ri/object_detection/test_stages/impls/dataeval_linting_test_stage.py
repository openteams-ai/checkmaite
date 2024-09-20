from typing import Any  # noqa: D100

from dataeval.detectors.linters import Duplicates, Outliers

from jatic_ri.object_detection.test_stages.interfaces.test_workflows import SingleDatasetPlugin, TestStage


class DatasetLintingTest(TestStage, SingleDatasetPlugin):
    """
    Dataset Linting TestStage implementation.

    Performs dataset linting by identifying duplicates (exact and near) as well as statistical outliers
    using various pixel and image statistics on the image data.
    """

    outputs = None

    def run(self, use_cache: bool = False) -> None:
        """Run linting"""

        if use_cache:
            return

        images = [data[0] for data in self.dataset]

        dupes = Duplicates().evaluate(images)
        outliers = Outliers().evaluate(images)

        if self.outputs is None:
            self.outputs = {}
        self.outputs["duplicates"] = dupes.dict()
        self.outputs["outliers"] = outliers.dict()

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect duplicates and outliers"""
        if self.outputs is None:
            return []

        return [self.outputs]
