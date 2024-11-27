"""DataEval Image Classification Bias Test Stage"""

from typing import Any

import maite.protocols.image_classification as ic
import numpy as np
from dataeval.utils import merge_metadata
from numpy.typing import NDArray

from jatic_ri._common.test_stages.impls.dataeval_bias_test_stage import DatasetBiasTestStageBase


class DatasetBiasTestStage(DatasetBiasTestStageBase[ic.Dataset]):
    """
    Measures four aspects of bias in a single dataset and programmatically generates a Gradient report
    with the measurements of bias, potential risks, and any actions required to reduce bias if found

    Bias is measured using four metrics: balance, coverage, diversity, parity.

    Balance, diversity, and parity calculate different aspects of correlation
    between metadata factors and class labels, while coverage is calculated using only the images
    """

    _deck: str = "image_classification_bias_evaluation"
    _task: str = "ic"

    def _get_images_labels_factors(self) -> tuple[list[NDArray[Any]], NDArray[np.int_], dict[str, NDArray[Any]]]:
        """Aggregate dataset into images, labels and metadata_factors"""

        images: list[NDArray[Any]] = []
        labels: list[np.intp] = []
        metadatas: list[dict[str, Any]] = []

        for d in self.dataset:
            images.append(np.asarray(d[0]))
            labels.append(np.argmax(d[1]))  # labels are one-hot encoded
            metadatas.append(d[2])

        metadata = {k: np.asarray(v) for k, v in merge_metadata(metadatas, ignore_lists=True).items()}
        return images, np.asarray(labels, dtype=np.int_), metadata
