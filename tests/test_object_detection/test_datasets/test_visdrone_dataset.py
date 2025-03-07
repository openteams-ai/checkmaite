from jatic_ri.object_detection.datasets import VisdroneDetectionDataset, DetectionTarget
import torch
from pathlib import Path


class TestVisdroneDetectionDataset:
    ROOT = Path(__file__).parents[2] / "testing_utilities" / "example_data" / "visdrone_dataset"

    def test_metadata_id(self):
        id = "sentinel"

        dataset = VisdroneDetectionDataset(self.ROOT, dataset_id=id)

        assert dataset.metadata["id"] == id

    def test_metadata_default(self):
        dataset = VisdroneDetectionDataset(self.ROOT)

        assert dataset.metadata["id"] == "visdrone"

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
