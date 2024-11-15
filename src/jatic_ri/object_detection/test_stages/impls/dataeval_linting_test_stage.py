"""DataEval Linting Image Classification Test Stage"""

from __future__ import annotations

from typing import Any

import maite.protocols.object_detection as od
import numpy as np
from dataeval.interop import as_numpy
from dataeval.metrics.stats import DatasetStatsOutput, HashStatsOutput, datasetstats, hashstats
from numpy.typing import NDArray

from jatic_ri._common.test_stages.impls.dataeval_linting_test_stage import DatasetLintingTestStageBase


class DatasetLintingTestStage(DatasetLintingTestStageBase[od.Dataset]):
    """
    Dataset Linting TestStage implementation.

    Performs dataset linting by identifying duplicates (exact and near) as well as statistical outliers
    using various pixel and image statistics on the image data on the datset images.
    """

    _deck: str = "object_detection_linting_evaluation"
    _task: str = "od"

    def _run_stats(self) -> tuple[HashStatsOutput, DatasetStatsOutput]:
        """Run stats for specific dataset type"""
        hashes = hashstats((d[0] for d in self.dataset), (d[1].boxes for d in self.dataset))
        stats = datasetstats((d[0] for d in self.dataset), (d[1].boxes for d in self.dataset))
        return hashes, stats

    def _get_image_label_box(self, index: int, target: int | None) -> tuple[NDArray[Any], str, NDArray[np.int_]]:
        """Get image, label and box from dataset at specified index and target"""
        datum = self.dataset[index]
        image = as_numpy(datum[0])
        label = int(as_numpy(datum[1].labels)[target or 0])
        box = as_numpy(datum[1].boxes).astype(np.int_)[target or 0]
        mapping: dict[int, str] | None = getattr(self.dataset, "index2label", None)
        label = "NO_LABEL" if label is None else str(label) if mapping is None else mapping[label]
        return image, label, box
