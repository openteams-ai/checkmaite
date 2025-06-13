import re
import shutil

import torch

from jatic_ri.object_detection.datasets import DetectionTarget, YoloDetectionDataset


class TestYoloDetectionDataset:
    YAML_DATASET = "tests/test_object_detection/data/yolo_dataset/dataset.yaml"
    ANN_DIR = "tests/test_object_detection/data/yolo_dataset/ann_dir"

    def test_metadata_default(self):
        dataset = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)
        assert re.match(r"yolo_[0-9a-f]{8}$", dataset.metadata["id"])
        dataset2 = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)
        # Assert that the two datasets have same ID
        assert dataset.metadata["id"] == dataset2.metadata["id"]

    def test_different_dirs_no_id_match(self, tmp_path):
        # Create first dataset
        dataset1 = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)

        # Create a temporary directory with copy of the data
        temp_root = tmp_path / "temp_yolo"
        temp_root.mkdir(parents=True)

        # Copy dataset.yaml and ann_dir
        shutil.copy(self.YAML_DATASET, temp_root / "dataset.yaml")
        shutil.copytree(self.ANN_DIR, temp_root / "ann_dir", dirs_exist_ok=True)

        # Create second dataset with the temporary directory
        dataset2 = YoloDetectionDataset(
            yaml_dataset=str(temp_root / "dataset.yaml"), ann_dir=str(temp_root / "ann_dir")
        )

        # Assert that the two datasets have different IDs
        assert dataset1.metadata["id"] != dataset2.metadata["id"]

    def test_yolo_dataset(self):
        yolo_dataset = YoloDetectionDataset(
            yaml_dataset=self.YAML_DATASET,
            ann_dir=self.ANN_DIR,
        )
        assert len(yolo_dataset) == 4
        element = yolo_dataset[0]
        assert isinstance(element[0], torch.Tensor)
        assert element[0].ndim == 3
        assert isinstance(element[1], DetectionTarget)
        assert element[1].boxes.ndim == 2
        assert element[1].labels.ndim == 1
        assert element[1].scores.ndim == 1
        assert element[1].scores.shape == (14,)
        assert element[2] == {"id": 0}
        assert yolo_dataset.metadata["index2label"][0] == "person"
        assert yolo_dataset.metadata["index2label"][1] == "bicycle"
