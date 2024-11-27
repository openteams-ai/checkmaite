"""DataEval Object Detection Bias Test Stage"""

from typing import Any

import maite.protocols.object_detection as od
import numpy as np
from dataeval.utils.metadata import _flatten_dict, merge_metadata
from numpy.typing import NDArray

from jatic_ri._common.test_stages.impls.dataeval_bias_test_stage import DatasetBiasTestStageBase


class DatasetBiasTestStage(DatasetBiasTestStageBase[od.Dataset]):
    """
    Measures four aspects of bias in a single dataset and programmatically generates a Gradient report
    with the measurements of bias, potential risks, and any actions required to reduce bias if found

    Bias is measured using four metrics: balance, coverage, diversity, parity.

    Balance, diversity, and parity calculate different aspects of correlation
    between metadata factors and class labels, while coverage is calculated using only the images
    """

    _deck: str = "object_detection_bias_evaluation"
    _task: str = "od"

    def _get_images_labels_factors(self) -> tuple[list[NDArray[Any]], NDArray[np.int_], dict[str, NDArray[Any]]]:
        """Aggregate dataset into images, labels and metadata_factors"""

        images: list[NDArray[Any]] = []
        labels: list[NDArray[np.int_]] = []
        metadatas: list[dict[str, Any]] = []

        for d in self.dataset:
            images.append(np.asarray(d[0]))
            np_labels = np.asarray(d[1].labels, dtype=np.int_)
            labels.append(np_labels)
            # we need to flatten/homogenize the metadata into a format for bias functionality
            # where dictionary has no nested dictionaries, and the value for each key is a list
            # of length N, where N is the number of labels/targets
            flattened = _flatten_dict(d[2], sep="_", ignore_lists=False, fully_qualified=False)
            # if the dataset does not have metadata in the proper form, then we need to expand
            # the metadata dictionary to be of the afformentioned shape
            if not all(isinstance(x, list) and len(x) for x in flattened.values()):
                flattened = {k: [v] * len(np_labels) for k, v in flattened.items()}
            metadatas.append(flattened)

        metadata = {k: np.asarray(v) for k, v in merge_metadata(metadatas).items()}
        return images, np.asarray(labels).flatten(), metadata
