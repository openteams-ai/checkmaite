from typing import Any, Optional  # noqa: D100

from dataeval.detectors.linters import Duplicates, Outliers

from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import SingleDatasetPlugin
from jatic_ri.util.cache import JSONCache


class DatasetLintingTest(TestStage[dict[str, Any]], SingleDatasetPlugin):
    """
    Dataset Linting TestStage implementation.

    Performs dataset linting by identifying duplicates (exact and near) as well as statistical outliers
    using various pixel and image statistics on the image data.
    """

    cache: Optional[Cache[dict[str, Any]]] = JSONCache()

    @property
    def cache_id(self) -> str:
        """Unique cache id for output"""
        return f"linting-{self.dataset_id}.json"

    def _run(self) -> dict[str, Any]:
        """Run linting"""
        images = [data[0] for data in self.dataset]

        dupes = Duplicates().evaluate(images)
        outliers = Outliers().evaluate(images)

        return {"duplicates": dupes.dict(), "outliers": outliers.dict()}

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect duplicates and outliers"""

        return [self.outputs]
