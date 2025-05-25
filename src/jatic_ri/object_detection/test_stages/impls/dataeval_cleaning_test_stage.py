"""DataEval Cleaning Image Classification Test Stage"""

from typing import Any

import maite.protocols.object_detection as od
import numpy as np
from dataeval.metrics.stats import boxratiostats, dimensionstats, hashstats, imagestats, labelstats
from numpy.typing import NDArray

from jatic_ri._common.test_stages.impls.dataeval_cleaning_test_stage import DatasetCleaningTestStageBase, DatasetStats


class DatasetCleaningTestStage(DatasetCleaningTestStageBase[od.Dataset]):
    """
    Dataset Cleaning TestStage implementation.

    Performs dataset cleaning by identifying duplicates (exact and near) as well as statistical outliers
    using various pixel and image statistics on the image data on the datset images.
    """

    _deck: str = "object_detection_cleaning_evaluation"
    _task: str = "od"

    def _run_stats(self) -> DatasetStats:
        """Run stats for specific dataset type"""
        hashes = hashstats(self.dataset)
        labstats = labelstats(self.dataset)
        imgstats = imagestats(self.dataset)
        boxstats = imagestats(self.dataset, per_box=True)
        dimstats = dimensionstats(self.dataset)
        ratiostats = boxratiostats(dimensionstats(self.dataset, per_box=True), dimstats)
        return DatasetStats(hashes, labstats, imgstats, boxstats, ratiostats)

    def _get_image_label_box(self, index: int, target: int | None) -> tuple[NDArray[Any], str, NDArray[np.int_] | None]:
        """Get image, label and box from dataset at specified index and target"""
        datum = self.dataset[index]
        image = np.asarray(datum[0])
        if np.asarray(datum[1].labels).size > 0:
            label = int(np.asarray(datum[1].labels)[target or 0])
            box = np.asarray(datum[1].boxes).astype(np.int_)[target or 0]
        else:
            label = None
            box = None
        mapping: dict[int, str] | None = getattr(self.dataset, "index2label", None)
        label = "NO_LABEL" if label is None else str(label) if mapping is None else mapping[label]
        return image, label, box
