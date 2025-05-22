"""DataEval Cleaning Image Classification Test Stage"""

from typing import Any

import maite.protocols.object_detection as od
import numpy as np
from dataeval.interop import as_numpy
from dataeval.metrics.stats import (
    DatasetStatsOutput,
    DimensionStatsOutput,
    HashStatsOutput,
    LabelStatsOutput,
    boxratiostats,
    datasetstats,
    hashstats,
    labelstats,
)
from maite.protocols import ArrayLike
from numpy.typing import NDArray

from jatic_ri._common.test_stages.impls.dataeval_cleaning_test_stage import DatasetCleaningTestStageBase


class DatasetCleaningTestStage(DatasetCleaningTestStageBase[od.Dataset]):
    """
    Dataset Cleaning TestStage implementation.

    Performs dataset cleaning by identifying duplicates (exact and near) as well as statistical outliers
    using various pixel and image statistics on the image data on the datset images.
    """

    _deck: str = "object_detection_cleaning_evaluation"
    _task: str = "od"

    def _increment_invalid_boxes(self, boxes: ArrayLike) -> NDArray:
        boxes_ = np.array(boxes, dtype=np.int_)

        for box in boxes_:
            if box[3] == box[1]:  # Height is zero
                box[3] += 1
            if box[2] == box[0]:  # Width is zero
                box[2] += 1

        return boxes_

    def _run_stats(
        self,
    ) -> tuple[HashStatsOutput, DatasetStatsOutput, LabelStatsOutput, DatasetStatsOutput, DimensionStatsOutput]:
        """Run stats for specific dataset type"""
        hashes = hashstats(d[0] for d in self.dataset)
        labstats = labelstats(d[1].labels for d in self.dataset)
        imgstats = datasetstats(d[0] for d in self.dataset)
        imgs = (d[0] for d in self.dataset)
        boxes = (self._increment_invalid_boxes(d[1].boxes) for d in self.dataset)
        boxstats = datasetstats(imgs, bboxes=boxes)
        ratiostats = boxratiostats(boxstats.dimensionstats, imgstats.dimensionstats)
        return hashes, imgstats, labstats, boxstats, ratiostats

    def _get_image_label_box(self, index: int, target: int | None) -> tuple[NDArray[Any], str, NDArray[np.int_] | None]:
        """Get image, label and box from dataset at specified index and target"""
        datum = self.dataset[index]
        image = as_numpy(datum[0])
        if as_numpy(datum[1].labels).size > 0:
            label = int(as_numpy(datum[1].labels)[target or 0])
            box = as_numpy(datum[1].boxes).astype(np.int_)[target or 0]
        else:
            label = None
            box = None
        mapping: dict[int, str] | None = getattr(self.dataset, "index2label", None)
        label = "NO_LABEL" if label is None else str(label) if mapping is None else mapping[label]
        return image, label, box
