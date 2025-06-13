import re
import shutil
from pathlib import Path

import torch

from jatic_ri.object_detection.datasets import DetectionTarget, VisdroneDetectionDataset


class TestVisdroneDetectionDataset:
    ROOT = Path(__file__).parents[2] / "testing_utilities" / "example_data" / "visdrone_dataset"

    def test_metadata_id(self):
        id = "sentinel"

        dataset = VisdroneDetectionDataset(self.ROOT, dataset_id=id)

        assert dataset.metadata["id"] == id

    def test_metadata_default(self):
        dataset = VisdroneDetectionDataset(self.ROOT)
        assert re.match(r"visdrone_[0-9a-f]{8}$", dataset.metadata["id"])
        dataset2 = VisdroneDetectionDataset(self.ROOT)
        # Assert that the two datasets have same ID
        assert dataset.metadata["id"] == dataset2.metadata["id"]

    def test_different_dirs_no_id_match(self, tmp_path):
        # Create first dataset with original ROOT
        dataset1 = VisdroneDetectionDataset(self.ROOT)

        # Create a temporary directory with subset of the data
        temp_root = tmp_path / "temp_visdrone"
        temp_root.mkdir(parents=True)

        # Move the dataset into a tmp dir
        for subdir in ["images", "annotations"]:
            (temp_root / subdir).mkdir()
            shutil.copytree(self.ROOT / subdir, temp_root / subdir, dirs_exist_ok=True)

        # Create second dataset with the temporary directory
        dataset2 = VisdroneDetectionDataset(temp_root)

        # Assert that the two datasets have different IDs
        assert dataset1.metadata["id"] != dataset2.metadata["id"]

    def test_metadata_index2label(self):
        dataset = VisdroneDetectionDataset(self.ROOT)

        assert len(dataset.metadata["index2label"]) == 12
        assert dataset.metadata["index2label"][0] == "ignored regions"
        assert dataset.metadata["index2label"][5] == "van"
        assert dataset.metadata["index2label"][11] == "others"

    def test_len(self):
        assert len(VisdroneDetectionDataset(self.ROOT)) == 3

    def test_getitem(self):
        dataset = VisdroneDetectionDataset(self.ROOT)

        assert dataset.metadata["index2label"][0] == "ignored regions"
        assert dataset.metadata["index2label"][5] == "van"
        assert dataset.metadata["index2label"][11] == "others"

        image, target, metadata = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.ndim == 3
        assert image.shape[0] == 3

        assert isinstance(target, DetectionTarget)
        assert isinstance(target.boxes, torch.Tensor)
        assert target.boxes.ndim == 2
        assert target.boxes.shape[-1] == 4
        assert isinstance(target.scores, torch.Tensor)
        assert target.scores.ndim == 1
        assert isinstance(target.labels, torch.Tensor)
        assert target.labels.ndim == 1
        assert target.boxes.shape[0] == target.scores.shape[0] == target.labels.shape[0]

        assert isinstance(metadata, dict)
        assert "image_path" in metadata
        assert isinstance(metadata["image_path"], str)
        # Fails if image_path is not a subpath
        Path(metadata["image_path"]).relative_to(self.ROOT)
        assert "annotation_path" in metadata
        assert isinstance(metadata["annotation_path"], str)
        Path(metadata["annotation_path"]).relative_to(self.ROOT)
        assert "truncations" in metadata
        assert len(metadata["truncations"]) == len(target.boxes)
        assert "occlusions" in metadata
        assert len(metadata["occlusions"]) == len(target.boxes)
