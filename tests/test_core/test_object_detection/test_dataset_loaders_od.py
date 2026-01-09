import re
import shutil
import time
from pathlib import Path

import pytest
import torch

from jatic_ri.core.object_detection.dataset_loaders import (
    CocoDetectionDataset,
    DatasetSpecification,
    DetectionTarget,
    VisdroneDetectionDataset,
    YoloDetectionDataset,
    load_datasets,
)


class TestVisdroneDetectionDataset:
    ROOT = Path(__file__).parents[2] / "data_for_tests"
    VISDRONE_ROOT = ROOT / "visdrone_dataset"

    def test_metadata_id(self):
        id = "sentinel"

        dataset = VisdroneDetectionDataset(self.VISDRONE_ROOT, dataset_id=id)

        assert dataset.metadata["id"] == id

    def test_metadata_default(self):
        dataset = VisdroneDetectionDataset(self.VISDRONE_ROOT)
        assert re.match(r"visdrone_[0-9a-f]{8}$", dataset.metadata["id"])
        dataset2 = VisdroneDetectionDataset(self.VISDRONE_ROOT)
        assert dataset.metadata["id"] == dataset2.metadata["id"]

    def test_different_dirs_no_id_match(self, tmp_path):
        dataset1 = VisdroneDetectionDataset(self.VISDRONE_ROOT)
        temp_root = tmp_path / "temp_visdrone"
        temp_root.mkdir(parents=True)

        for subdir in ["images", "annotations"]:
            (temp_root / subdir).mkdir()
            shutil.copytree(self.VISDRONE_ROOT / subdir, temp_root / subdir, dirs_exist_ok=True)

        dataset2 = VisdroneDetectionDataset(temp_root)

        assert dataset1.metadata["id"] != dataset2.metadata["id"]

    def test_metadata_index2label(self):
        dataset = VisdroneDetectionDataset(self.VISDRONE_ROOT)

        assert len(dataset.metadata["index2label"]) == 12
        assert dataset.metadata["index2label"][0] == "ignored regions"
        assert dataset.metadata["index2label"][5] == "van"
        assert dataset.metadata["index2label"][11] == "others"

    def test_len(self):
        assert len(VisdroneDetectionDataset(self.VISDRONE_ROOT)) == 3

    def test_getitem(self):
        dataset = VisdroneDetectionDataset(self.VISDRONE_ROOT)

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
        Path(metadata["image_path"]).relative_to(self.VISDRONE_ROOT)
        assert "annotation_path" in metadata
        assert isinstance(metadata["annotation_path"], str)
        Path(metadata["annotation_path"]).relative_to(self.VISDRONE_ROOT)
        assert "truncations" in metadata
        assert len(metadata["truncations"]) == len(target.boxes)
        assert "occlusions" in metadata
        assert len(metadata["occlusions"]) == len(target.boxes)


class TestYoloDetectionDataset:
    ROOT = Path(__file__).parents[2] / "data_for_tests"
    YOLO_ROOT = ROOT / "yolo_dataset"
    YAML_DATASET = YOLO_ROOT / "dataset.yaml"
    ANN_DIR = YOLO_ROOT / "ann_dir"

    def test_metadata_is_dict(self):
        dataset = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)

        assert isinstance(dataset.metadata, dict)

    def test_metadata_default(self):
        dataset = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)
        assert re.match(r"yolo_[0-9a-f]{8}$", dataset.metadata["id"])
        dataset2 = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)
        assert dataset.metadata["id"] == dataset2.metadata["id"]

    def test_different_dirs_no_id_match(self, tmp_path):
        dataset1 = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)

        temp_root = tmp_path / "temp_yolo"
        temp_root.mkdir(parents=True)

        shutil.copy(self.YAML_DATASET, temp_root / "dataset.yaml")
        shutil.copytree(self.ANN_DIR, temp_root / "ann_dir", dirs_exist_ok=True)

        dataset2 = YoloDetectionDataset(
            yaml_dataset=str(temp_root / "dataset.yaml"), ann_dir=str(temp_root / "ann_dir")
        )

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


class TestCocoDetectionDataset:
    ROOT = Path(__file__).parents[2] / "data_for_tests"
    COCO_ROOT = ROOT / "coco_dataset"
    ANN_FILE = str(COCO_ROOT / "ann_file.json")

    def test_metadata_default(self):
        dataset = CocoDetectionDataset(root=self.COCO_ROOT, ann_file=self.ANN_FILE)
        assert re.match(r"coco_[0-9a-f]{8}$", dataset.metadata["id"])
        dataset2 = CocoDetectionDataset(root=self.COCO_ROOT, ann_file=self.ANN_FILE)
        assert dataset.metadata["id"] == dataset2.metadata["id"]

    def test_different_dirs_no_id_match(self, tmp_path):
        dataset1 = CocoDetectionDataset(root=self.COCO_ROOT, ann_file=self.ANN_FILE)

        temp_root = tmp_path / "temp_coco"
        temp_root.mkdir(parents=True)

        shutil.copytree(self.COCO_ROOT, temp_root, dirs_exist_ok=True)

        dataset2 = CocoDetectionDataset(root=temp_root, ann_file=temp_root / "ann_file.json")

        assert dataset1.metadata["id"] != dataset2.metadata["id"]

    def test_coco_dataset(self):
        coco_dataset = CocoDetectionDataset(
            root=self.COCO_ROOT,
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

    def test_coco_dataset_missing_image_id_raises_keyerror(self):
        """Simulate a mismatch where an image id is missing from the images index.

        We remove one id from the prebuilt _id_to_image map to trigger the error
        path and assert a helpful KeyError is raised.
        """
        coco_dataset = CocoDetectionDataset(
            root=self.COCO_ROOT,
            ann_file=self.ANN_FILE,
        )

        # Pick a valid id used by the underlying CocoDetection dataset
        missing_id = coco_dataset.dataset.ids[0]
        # Simulate inconsistent annotations by removing it from the index
        del coco_dataset._id_to_image[missing_id]

        with pytest.raises(KeyError, match=r"Image id .* not found"):
            _ = coco_dataset[0]


class TestDatasetLoader:
    ROOT = Path(__file__).parents[2] / "data_for_tests"
    YOLO_METADATA_PATH = ROOT / "yolo_dataset" / "dataset.yaml"
    YOLO_ANNOTATION_DIR = ROOT / "yolo_dataset" / "ann_dir"
    COCO_ANNOTATION_PATH = ROOT / "coco_dataset" / "ann_file.json"

    def test_load_datasets_yolo(self):
        spec: DatasetSpecification = {
            "dataset_type": "YoloDetectionDataset",
            "metadata_path": self.YOLO_METADATA_PATH,
            "data_dir": self.YOLO_ANNOTATION_DIR,
        }
        datasets = {
            "yolo_dataset1": spec,
        }
        loaded = load_datasets(datasets=datasets)
        assert "yolo_dataset1" in loaded

    def test_load_datasets_coco(self):
        spec: DatasetSpecification = {
            "dataset_type": "CocoDetectionDataset",
            "metadata_path": self.COCO_ANNOTATION_PATH,
            "data_dir": Path(self.COCO_ANNOTATION_PATH).parent.resolve(),
        }

        datasets = {
            "coco_dataset1": spec,
        }
        loaded = load_datasets(datasets=datasets)
        assert "coco_dataset1" in loaded

    def test_load_datasets_visdrone(self):
        spec: DatasetSpecification = {
            "dataset_type": "VisdroneDetectionDataset",
            "metadata_path": "",
            "data_dir": Path(__file__).parents[2] / "data_for_tests" / "visdrone_dataset",
        }

        key = "visdrone_dataset1"

        loaded = load_datasets(datasets={key: spec})
        assert key in loaded

    def test_load_datasets_invalid(self):
        spec: DatasetSpecification = {
            "dataset_type": "NonexistentClass",
        }
        datasets_invalid = {
            "dataset_invalid": spec,
        }
        with pytest.raises(RuntimeError, match=r"\bNonexistentClass\b"):
            load_datasets(datasets=datasets_invalid)

    def test_coco_dataset_metadata_lookup_performance(self):
        """Benchmark metadata lookup performance with O(1) dict access.

        This test verifies that repeated __getitem__ calls on the COCO dataset
        maintain consistent O(1) performance by measuring access times.
        """
        coco_dataset = CocoDetectionDataset(
            root=Path(self.COCO_ANNOTATION_PATH).parent.resolve(),
            ann_file=str(self.COCO_ANNOTATION_PATH),
        )

        # Warm up: single access
        _ = coco_dataset[0]

        # Measure repeated accesses
        n_iterations = 100
        start_time = time.perf_counter()

        for i in range(n_iterations):
            _ = coco_dataset[i % len(coco_dataset)]

        elapsed_time = time.perf_counter() - start_time
        avg_access_time = elapsed_time / n_iterations

        # Assert that average access time is very fast (O(1) behavior)
        # Even with 100 iterations, should be sub-millisecond per access on modern hardware
        # This is a loose bound to accommodate various system speeds
        assert avg_access_time < 0.01, (
            f"Average metadata lookup time {avg_access_time:.6f}s exceeds threshold. "
            f"Expected O(1) dict access, got {avg_access_time * 1000:.3f}ms per access."
        )
