import re
import shutil
from pathlib import Path

import torch

from jatic_ri.object_detection.datasets import CocoDetectionDataset, DetectionTarget


class TestCocoDetectionDataset:
    ROOT = Path("tests/testing_utilities/example_data/coco_dataset")
    ANN_FILE = str(ROOT / "ann_file.json")

    def test_metadata_default(self):
        dataset = CocoDetectionDataset(root=self.ROOT, ann_file=self.ANN_FILE)
        assert re.match(r"coco_[0-9a-f]{8}$", dataset.metadata["id"])
        dataset2 = CocoDetectionDataset(root=self.ROOT, ann_file=self.ANN_FILE)
        # Assert that the two datasets have same ID
        assert dataset.metadata["id"] == dataset2.metadata["id"]

    def test_different_dirs_no_id_match(self, tmp_path):
        # Create first dataset
        dataset1 = CocoDetectionDataset(root=self.ROOT, ann_file=self.ANN_FILE)

        # Create a temporary directory with copy of the data
        temp_root = tmp_path / "temp_coco"
        temp_root.mkdir(parents=True)

        # Copy ann_file and images
        shutil.copytree(self.ROOT, temp_root, dirs_exist_ok=True)

        # Create second dataset with the temporary directory
        dataset2 = CocoDetectionDataset(root=temp_root, ann_file=temp_root / "ann_file.json")

        # Assert that the two datasets have different IDs
        assert dataset1.metadata["id"] != dataset2.metadata["id"]

    def test_coco_dataset(self):
        coco_dataset = CocoDetectionDataset(
            root=self.ROOT,
            ann_file=self.ANN_FILE,
        )
        assert len(coco_dataset) == 4
        element = coco_dataset[0]
        assert isinstance(element[0], torch.Tensor)
        assert element[0].ndim == 3
        assert isinstance(element[1], DetectionTarget)
        assert element[1].boxes.ndim == 2
        assert element[1].labels.ndim == 1
        assert element[1].scores.ndim == 1
        assert element[1].scores.shape == (14,)
        assert isinstance(element[2], dict)
        assert coco_dataset.metadata["index2label"][1] == "person"
        assert coco_dataset.metadata["index2label"][2] == "bicycle"
