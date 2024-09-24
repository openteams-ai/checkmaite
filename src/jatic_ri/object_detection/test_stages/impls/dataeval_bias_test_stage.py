from dataeval.metrics.bias import coverage, parity  # noqa
from typing import Any
from jatic_ri._common.test_stages.interfaces.test_stage import TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import SingleDatasetPlugin
import maite.protocols.object_detection as od
from maite.protocols import ArrayLike
from collections import defaultdict
import numpy as np


def read_dataset(dataset: od.Dataset) -> tuple[od.InputBatchType, od.TargetBatchType, list[dict[str, Any]]]:
    """Reads maite.od.Dataset and extracts individual components"""
    images: list[ArrayLike] = []
    od_targets: list[od.ObjectDetectionTarget] = []
    metadata: list[dict[str, Any]] = []

    for image, od_target, meta in dataset:
        images.append(image)
        od_targets.append(od_target)
        metadata.append(meta)

    return images, od_targets, metadata


class DatasetBiasTest(TestStage, SingleDatasetPlugin):
    """
    Tests for bias in a dataset

    Bias is determined through coverage of images as well as
    parity and balance between metadata and labels
    """

    outputs = None

    def run(self, use_cache: bool = False) -> None:
        """Run bias analysis using coverage and parity"""

        if use_cache:
            return

        # Separate data into individual lists
        images, targets, metadata = read_dataset(self.dataset)

        # Aggregate each metadata factor into lists:
        # dict[factor] = List[factor_values]
        factor_lists = defaultdict(list)
        labels = []

        # Flattens all targets into one array and
        # copies metadata for an image into all of its targets
        for target, mdata in zip(targets, metadata):
            # Generates flat list of all labels
            tlabels = np.array(target.labels)
            labels.extend(tlabels.tolist())

            # Aggregates list for each metadata factor
            for k, v in mdata.items():
                v = [v] * len(tlabels)
                factor_lists[k].extend(v)

        # Parity considers labels as metadata
        factor_lists["class"] = labels

        # Convert all lists into ArrayLike
        data_factors = {k: np.array(v) for k, v in factor_lists.items()}

        if self.outputs is None:
            self.outputs = {}

        self.outputs["coverage"] = coverage(images, k=5)
        self.outputs["parity"] = parity(data_factors)

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect consumables"""
        if self.outputs is None:
            return []

        # Creates a dict slide for each metric
        return [{k: v} for k, v in self.outputs.items()]
