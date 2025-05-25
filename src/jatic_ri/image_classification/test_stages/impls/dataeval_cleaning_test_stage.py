"""DataEval Cleaning Image Classification Test Stage"""

from typing import Any

import maite.protocols.image_classification as ic
import numpy as np
from dataeval.metrics.stats import hashstats, imagestats, labelstats
from numpy.typing import NDArray

from jatic_ri._common.test_stages.impls.dataeval_cleaning_test_stage import DatasetCleaningTestStageBase, DatasetStats


class DatasetCleaningTestStage(DatasetCleaningTestStageBase[ic.Dataset]):
    """
    Dataset Cleaning TestStage implementation.

    Performs dataset cleaning by identifying duplicates (exact and near) as well as statistical outliers
    using various pixel and image statistics on the image data on the datset images.
    """

    _deck: str = "image_classification_cleaning_evaluation"
    _task: str = "ic"

    def _run_stats(self) -> DatasetStats:
        """Run stats for specific dataset type"""
        hashes = hashstats(self.dataset)
        labstats = labelstats(self.dataset)
        imgstats = imagestats(self.dataset)
        return DatasetStats(hashes, labstats, imgstats, None, None)

    def _get_image_label_box(self, index: int, target: int | None) -> tuple[NDArray[Any], str, NDArray[np.int_] | None]:  # noqa: ARG002
        """Get image, label and box from dataset at specified index and target"""
        data = self.dataset[index]
        image = np.asarray(data[0])
        label = int(np.argmax(np.asarray(data[1])))
        mapping: dict[int, str] | None = getattr(self.dataset, "index2label", None)
        label = "NO_LABEL" if label is None else str(label) if mapping is None else mapping[label]
        return image, label, None
