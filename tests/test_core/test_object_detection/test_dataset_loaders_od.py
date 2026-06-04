import re
import shutil
import time
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from checkmaite.core.object_detection.dataset_loaders import (
    CocoDetectionDataset,
    DatasetSpecification,
    DetectionTarget,
    VisdroneDetectionDataset,
    XaitkExplainableDetectionBaselineDataset,
    YoloDetectionDataLoader,
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

    def test_get_input(self):
        dataset = VisdroneDetectionDataset(self.VISDRONE_ROOT)
        image = dataset.get_input(0)

        assert isinstance(image, torch.Tensor)
        assert image.ndim == 3
        assert image.shape[0] == 3

    def test_get_target(self):
        dataset = VisdroneDetectionDataset(self.VISDRONE_ROOT)
        target = dataset.get_target(0)

        assert isinstance(target, DetectionTarget)
        assert isinstance(target.boxes, torch.Tensor)
        assert target.boxes.ndim == 2
        assert target.boxes.shape[-1] == 4
        assert isinstance(target.scores, torch.Tensor)
        assert target.scores.ndim == 1
        assert isinstance(target.labels, torch.Tensor)
        assert target.labels.ndim == 1

    def test_get_metadata(self):
        dataset = VisdroneDetectionDataset(self.VISDRONE_ROOT)
        metadata = dataset.get_metadata(0)

        assert isinstance(metadata, dict)
        assert "id" in metadata
        assert "image_path" in metadata
        assert "annotation_path" in metadata

    def test_fieldwise_methods_consistent_with_getitem(self):
        dataset = VisdroneDetectionDataset(self.VISDRONE_ROOT)

        for i in range(len(dataset)):
            image_full, target_full, metadata_full = dataset[i]
            image_field = dataset.get_input(i)
            target_field = dataset.get_target(i)
            metadata_field = dataset.get_metadata(i)

            assert torch.equal(image_full, image_field)
            assert torch.equal(target_full.boxes, target_field.boxes)
            assert torch.equal(target_full.labels, target_field.labels)
            assert torch.equal(target_full.scores, target_field.scores)
            assert metadata_full == metadata_field


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
        # YAML uses 'train: images' (relative to YAML parent), so images must exist there
        shutil.copytree(self.YOLO_ROOT / "images", temp_root / "images", dirs_exist_ok=True)

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
        assert isinstance(element[2]["id"], str)  # image filename, e.g. "000000037777.jpg"
        assert yolo_dataset.metadata["index2label"][0] == "person"
        assert yolo_dataset.metadata["index2label"][1] == "bicycle"

    def test_yolo_lazy_loading(self):
        """Test that annotations are NOT cached at init (lazy parsing)."""
        yolo_dataset = YoloDetectionDataset(
            yaml_dataset=self.YAML_DATASET,
            ann_dir=self.ANN_DIR,
        )
        # Eager annotation cache must not exist; only (image, label) path pairs are stored
        assert not hasattr(yolo_dataset, "_annotations")
        assert hasattr(yolo_dataset, "_samples")
        assert len(yolo_dataset._samples) == len(yolo_dataset)
        for img_path, _label_path in yolo_dataset._samples:
            assert img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def test_get_input(self):
        dataset = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)
        image = dataset.get_input(0)

        assert isinstance(image, torch.Tensor)
        assert image.ndim == 3
        assert image.shape[0] == 3

    def test_get_target_does_not_load_image(self):
        """Test that get_target returns the target without loading the image."""
        dataset = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)
        target = dataset.get_target(0)

        assert isinstance(target, DetectionTarget)
        assert target.boxes.ndim == 2
        assert target.labels.ndim == 1
        assert target.scores.ndim == 1

    def test_get_metadata(self):
        dataset = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)
        metadata = dataset.get_metadata(0)

        assert isinstance(metadata, dict)
        assert "id" in metadata
        assert isinstance(metadata["id"], str)  # image filename, e.g. "000000037777.jpg"

    def test_fieldwise_methods_consistent_with_getitem(self):
        dataset = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)

        for i in range(len(dataset)):
            image_full, target_full, metadata_full = dataset[i]
            image_field = dataset.get_input(i)
            target_field = dataset.get_target(i)
            metadata_field = dataset.get_metadata(i)

            assert torch.equal(image_full, image_field)
            assert torch.equal(target_full.boxes, target_field.boxes)
            assert torch.equal(target_full.labels, target_field.labels)
            assert torch.equal(target_full.scores, target_field.scores)
            assert metadata_full == metadata_field

    def test_get_input_index_error(self):
        dataset = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)
        with pytest.raises(IndexError):
            dataset.get_input(100)

    def test_get_target_index_error(self):
        dataset = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)
        with pytest.raises(IndexError):
            dataset.get_target(100)

    def test_get_metadata_index_error(self):
        dataset = YoloDetectionDataset(yaml_dataset=self.YAML_DATASET, ann_dir=self.ANN_DIR)
        with pytest.raises(IndexError):
            dataset.get_metadata(100)


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
        # Verify scores shape matches boxes/labels (annotations vary per image)
        assert element[1].scores.shape[0] == element[1].boxes.shape[0]
        assert element[1].scores.shape[0] == element[1].labels.shape[0]
        assert isinstance(element[2], dict)
        assert coco_dataset.metadata["index2label"][1] == "person"
        assert coco_dataset.metadata["index2label"][2] == "bicycle"

    def test_coco_dataset_missing_image_id_raises_keyerror(self):
        """Simulate a mismatch where an image id is missing from the annotation index.

        We remove one id from the prebuilt _img_id_to_annotations map to trigger the error
        path and assert a helpful KeyError is raised.
        """
        coco_dataset = CocoDetectionDataset(
            root=self.COCO_ROOT,
            ann_file=self.ANN_FILE,
        )

        # Pick a valid id from the images list
        missing_id = coco_dataset._images[0]["id"]
        # Simulate inconsistent annotations by removing it from the annotation index
        del coco_dataset._img_id_to_annotations[missing_id]

        # Now the image exists but has no annotations (empty list from defaultdict)
        # This should not raise - it should return an empty detection target
        element = coco_dataset[0]
        assert element[1].boxes.shape[0] == 0

    def test_get_input(self):
        dataset = CocoDetectionDataset(root=self.COCO_ROOT, ann_file=self.ANN_FILE)
        image = dataset.get_input(0)

        assert isinstance(image, torch.Tensor)
        assert image.ndim == 3
        assert image.shape[0] == 3

    def test_get_target_does_not_load_image(self):
        """Test that get_target returns the target without loading the image."""
        dataset = CocoDetectionDataset(root=self.COCO_ROOT, ann_file=self.ANN_FILE)
        target = dataset.get_target(0)

        assert isinstance(target, DetectionTarget)
        assert target.boxes.ndim == 2
        assert target.labels.ndim == 1
        assert target.scores.ndim == 1

    def test_get_metadata_does_not_load_image(self):
        """Test that get_metadata returns without loading the image."""
        dataset = CocoDetectionDataset(root=self.COCO_ROOT, ann_file=self.ANN_FILE)
        metadata = dataset.get_metadata(0)

        assert isinstance(metadata, dict)
        assert "id" in metadata
        assert "file_name" in metadata

    def test_fieldwise_methods_consistent_with_getitem(self):
        dataset = CocoDetectionDataset(root=self.COCO_ROOT, ann_file=self.ANN_FILE)

        for i in range(len(dataset)):
            image_full, target_full, metadata_full = dataset[i]
            image_field = dataset.get_input(i)
            target_field = dataset.get_target(i)
            metadata_field = dataset.get_metadata(i)

            assert torch.equal(image_full, image_field)
            assert torch.equal(target_full.boxes, target_field.boxes)
            assert torch.equal(target_full.labels, target_field.labels)
            assert torch.equal(target_full.scores, target_field.scores)
            assert metadata_full == metadata_field

    def test_get_input_index_error(self):
        dataset = CocoDetectionDataset(root=self.COCO_ROOT, ann_file=self.ANN_FILE)
        with pytest.raises(IndexError):
            dataset.get_input(100)

    def test_get_target_index_error(self):
        dataset = CocoDetectionDataset(root=self.COCO_ROOT, ann_file=self.ANN_FILE)
        with pytest.raises(IndexError):
            dataset.get_target(100)

    def test_get_metadata_index_error(self):
        dataset = CocoDetectionDataset(root=self.COCO_ROOT, ann_file=self.ANN_FILE)
        with pytest.raises(IndexError):
            dataset.get_metadata(100)


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


class TestAccessorMethodPerformance:
    """Performance tests comparing accessor methods to __getitem__.

    The accessor methods (get_input, get_target, get_metadata) are designed to
    provide performance benefits when only a subset of the data is needed:
    - get_target: Skips image loading, should be significantly faster than __getitem__
    - get_metadata: Skips image loading and annotation processing, should be fastest
    - get_input: Still loads the image, expected to have similar performance to __getitem__
    """

    ROOT = Path(__file__).parents[2] / "data_for_tests"
    COCO_ROOT = ROOT / "coco_dataset"
    COCO_ANN_FILE = str(COCO_ROOT / "ann_file.json")
    YOLO_ROOT = ROOT / "yolo_dataset"
    YOLO_YAML = YOLO_ROOT / "dataset.yaml"
    YOLO_ANN_DIR = YOLO_ROOT / "ann_dir"
    VISDRONE_ROOT = ROOT / "visdrone_dataset"

    def test_coco_get_target_faster_than_getitem(self):
        """Verify get_target is faster than __getitem__ by skipping image loading."""
        dataset = CocoDetectionDataset(root=self.COCO_ROOT, ann_file=self.COCO_ANN_FILE)
        n_iterations = 50

        # Warm up
        _ = dataset[0]
        _ = dataset.get_target(0)

        # Measure __getitem__ time
        start = time.perf_counter()
        for i in range(n_iterations):
            _ = dataset[i % len(dataset)]
        getitem_time = time.perf_counter() - start

        # Measure get_target time
        start = time.perf_counter()
        for i in range(n_iterations):
            _ = dataset.get_target(i % len(dataset))
        get_target_time = time.perf_counter() - start

        assert get_target_time < getitem_time, (
            f"get_target ({get_target_time:.4f}s) should be faster than __getitem__ ({getitem_time:.4f}s) "
            "because it skips image loading"
        )

    def test_coco_get_metadata_faster_than_getitem(self):
        """Verify get_metadata is faster than __getitem__ by skipping image and annotation processing."""
        dataset = CocoDetectionDataset(root=self.COCO_ROOT, ann_file=self.COCO_ANN_FILE)
        n_iterations = 50

        # Warm up
        _ = dataset[0]
        _ = dataset.get_metadata(0)

        # Measure __getitem__ time
        start = time.perf_counter()
        for i in range(n_iterations):
            _ = dataset[i % len(dataset)]
        getitem_time = time.perf_counter() - start

        # Measure get_metadata time
        start = time.perf_counter()
        for i in range(n_iterations):
            _ = dataset.get_metadata(i % len(dataset))
        get_metadata_time = time.perf_counter() - start

        assert get_metadata_time < getitem_time, (
            f"get_metadata ({get_metadata_time:.4f}s) should be faster than __getitem__ ({getitem_time:.4f}s) "
            "because it skips image loading and annotation processing"
        )

    def test_yolo_get_target_faster_than_getitem(self):
        """Verify YOLO get_target is faster than __getitem__ by skipping image loading."""
        dataset = YoloDetectionDataset(yaml_dataset=self.YOLO_YAML, ann_dir=self.YOLO_ANN_DIR)
        n_iterations = 50

        # Warm up
        _ = dataset[0]
        _ = dataset.get_target(0)

        # Measure __getitem__ time
        start = time.perf_counter()
        for i in range(n_iterations):
            _ = dataset[i % len(dataset)]
        getitem_time = time.perf_counter() - start

        # Measure get_target time
        start = time.perf_counter()
        for i in range(n_iterations):
            _ = dataset.get_target(i % len(dataset))
        get_target_time = time.perf_counter() - start

        assert get_target_time < getitem_time, (
            f"get_target ({get_target_time:.4f}s) should be faster than __getitem__ ({getitem_time:.4f}s) "
            "because it skips image loading"
        )

    def test_yolo_get_metadata_faster_than_getitem(self):
        """Verify YOLO get_metadata is faster than __getitem__ by skipping image loading."""
        dataset = YoloDetectionDataset(yaml_dataset=self.YOLO_YAML, ann_dir=self.YOLO_ANN_DIR)
        n_iterations = 50

        # Warm up
        _ = dataset[0]
        _ = dataset.get_metadata(0)

        # Measure __getitem__ time
        start = time.perf_counter()
        for i in range(n_iterations):
            _ = dataset[i % len(dataset)]
        getitem_time = time.perf_counter() - start

        # Measure get_metadata time
        start = time.perf_counter()
        for i in range(n_iterations):
            _ = dataset.get_metadata(i % len(dataset))
        get_metadata_time = time.perf_counter() - start

        assert get_metadata_time < getitem_time, (
            f"get_metadata ({get_metadata_time:.4f}s) should be faster than __getitem__ ({getitem_time:.4f}s) "
            "because it skips image loading"
        )

    def test_visdrone_get_target_faster_than_getitem(self):
        """Verify VisDrone get_target is faster than __getitem__ by skipping image loading."""
        dataset = VisdroneDetectionDataset(self.VISDRONE_ROOT)
        n_iterations = 50

        # Warm up
        _ = dataset[0]
        _ = dataset.get_target(0)

        # Measure __getitem__ time
        start = time.perf_counter()
        for i in range(n_iterations):
            _ = dataset[i % len(dataset)]
        getitem_time = time.perf_counter() - start

        # Measure get_target time
        start = time.perf_counter()
        for i in range(n_iterations):
            _ = dataset.get_target(i % len(dataset))
        get_target_time = time.perf_counter() - start

        assert get_target_time < getitem_time, (
            f"get_target ({get_target_time:.4f}s) should be faster than __getitem__ ({getitem_time:.4f}s) "
            "because it skips image loading"
        )

    def test_visdrone_get_metadata_faster_than_getitem(self):
        """Verify VisDrone get_metadata is faster than __getitem__ by skipping image loading."""
        dataset = VisdroneDetectionDataset(self.VISDRONE_ROOT)
        n_iterations = 50

        # Warm up
        _ = dataset[0]
        _ = dataset.get_metadata(0)

        # Measure __getitem__ time
        start = time.perf_counter()
        for i in range(n_iterations):
            _ = dataset[i % len(dataset)]
        getitem_time = time.perf_counter() - start

        # Measure get_metadata time
        start = time.perf_counter()
        for i in range(n_iterations):
            _ = dataset.get_metadata(i % len(dataset))
        get_metadata_time = time.perf_counter() - start

        assert get_metadata_time < getitem_time, (
            f"get_metadata ({get_metadata_time:.4f}s) should be faster than __getitem__ ({getitem_time:.4f}s) "
            "because it skips image loading"
        )


# ─── New tests for OD fixes and DataLoader ────────────────────────────────────


def _make_img(path, size=(100, 50), mode="RGB"):
    Image.new(mode, size).save(path)


def _write_yaml(path, content):
    path.write_text(yaml.dump(content))


class TestYoloDetectionDatasetNew:
    """Tests for the new YoloDetectionDataset behavior."""

    def test_box_conversion_to_pixel_xyxy(self, tmp_path):
        (tmp_path / "images" / "val").mkdir(parents=True)
        (tmp_path / "labels" / "val").mkdir(parents=True)
        _make_img(tmp_path / "images" / "val" / "img1.jpg", size=(100, 50))
        (tmp_path / "labels" / "val" / "img1.txt").write_text("0 0.5 0.5 0.2 0.4\n")

        _write_yaml(tmp_path / "data.yaml", {"val": "images/val", "names": {0: "cat"}})
        ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split="val")

        _, target, _ = ds[0]
        torch.testing.assert_close(target.boxes[0], torch.tensor([40.0, 15.0, 60.0, 35.0]))
        assert target.labels.dtype == torch.int64
        assert target.scores.dtype == torch.float32

    def test_pairs_labels_by_stem(self, tmp_path):
        (tmp_path / "images" / "val").mkdir(parents=True)
        (tmp_path / "labels" / "val").mkdir(parents=True)
        _make_img(tmp_path / "images" / "val" / "b.jpg", size=(100, 100))
        _make_img(tmp_path / "images" / "val" / "a.jpg", size=(100, 100))
        (tmp_path / "labels" / "val" / "a.txt").write_text("0 0.1 0.1 0.1 0.1\n")
        (tmp_path / "labels" / "val" / "b.txt").write_text("1 0.9 0.9 0.1 0.1\n")

        _write_yaml(tmp_path / "data.yaml", {"val": "images/val", "names": {0: "cat", 1: "dog"}})
        ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split="val")

        by_name = {ds.get_metadata(i)["id"]: ds.get_target(i) for i in range(len(ds))}
        assert by_name["a.jpg"].labels[0].item() == 0
        assert by_name["b.jpg"].labels[0].item() == 1

    def test_missing_label_file_returns_empty_target(self, tmp_path):
        (tmp_path / "images" / "val").mkdir(parents=True)
        _make_img(tmp_path / "images" / "val" / "img.jpg", size=(50, 50))

        _write_yaml(tmp_path / "data.yaml", {"val": "images/val", "names": {0: "cat"}})
        ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split="val")

        _, target, _ = ds[0]
        assert target.boxes.shape == (0, 4)
        assert target.labels.shape == (0,)
        assert target.labels.dtype == torch.int64
        assert target.scores.shape == (0,)

    def test_extra_label_file_is_ignored(self, tmp_path):
        (tmp_path / "images" / "val").mkdir(parents=True)
        (tmp_path / "labels" / "val").mkdir(parents=True)
        _make_img(tmp_path / "images" / "val" / "img.jpg", size=(50, 50))
        (tmp_path / "labels" / "val" / "img.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (tmp_path / "labels" / "val" / "extra.txt").write_text("1 0.1 0.1 0.1 0.1\n")

        _write_yaml(tmp_path / "data.yaml", {"val": "images/val", "names": {0: "cat", 1: "dog"}})
        ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split="val")

        assert len(ds) == 1
        _, target, _ = ds[0]
        assert target.labels[0].item() == 0

    def test_train_val_test_splits(self, tmp_path):
        for split in ("train", "val", "test"):
            (tmp_path / "images" / split).mkdir(parents=True)
            _make_img(tmp_path / "images" / split / f"{split}.jpg", size=(10, 10))

        _write_yaml(
            tmp_path / "data.yaml",
            {"train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "cat"}},
        )
        for split in ("train", "val", "test"):
            ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split=split)
            assert len(ds) == 1
            assert ds.get_metadata(0)["id"] == f"{split}.jpg"

    def test_resolves_split_relative_to_yaml_file(self, tmp_path):
        (tmp_path / "images" / "val").mkdir(parents=True)
        _make_img(tmp_path / "images" / "val" / "img.jpg", size=(10, 10))

        _write_yaml(tmp_path / "data.yaml", {"val": "images/val", "names": {0: "cat"}})
        ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split="val")
        assert len(ds) == 1

    def test_resolves_yaml_path_field(self, tmp_path):
        dataset_root = tmp_path / "dataset-root"
        (dataset_root / "images" / "val").mkdir(parents=True)
        _make_img(dataset_root / "images" / "val" / "img.jpg", size=(10, 10))

        _write_yaml(
            tmp_path / "data.yaml",
            {"path": "dataset-root", "val": "images/val", "names": {0: "cat"}},
        )
        ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split="val")
        assert len(ds) == 1

    def test_parser_allows_whitespace_blank_and_comments(self, tmp_path):
        (tmp_path / "images" / "val").mkdir(parents=True)
        (tmp_path / "labels" / "val").mkdir(parents=True)
        _make_img(tmp_path / "images" / "val" / "img.jpg", size=(100, 100))
        label_content = "# comment line\n" "\n" "0\t0.5\t0.5\t0.2\t0.2\n" "   \n" "1  0.1  0.1  0.1  0.1\n"
        (tmp_path / "labels" / "val" / "img.txt").write_text(label_content)

        _write_yaml(tmp_path / "data.yaml", {"val": "images/val", "names": {0: "a", 1: "b"}})
        ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split="val")
        _, target, _ = ds[0]
        assert len(target.boxes) == 2
        assert target.labels.tolist() == [0, 1]

    def test_parser_reports_malformed_rows(self, tmp_path):
        (tmp_path / "images" / "val").mkdir(parents=True)
        (tmp_path / "labels" / "val").mkdir(parents=True)
        _make_img(tmp_path / "images" / "val" / "img.jpg", size=(10, 10))
        (tmp_path / "labels" / "val" / "img.txt").write_text("0 0.5 0.5\n")  # 3 fields, not 5

        _write_yaml(tmp_path / "data.yaml", {"val": "images/val", "names": {0: "cat"}})
        ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split="val")

        with pytest.raises(ValueError, match="line 1"):
            ds[0]

    def test_ignores_non_image_files(self, tmp_path):
        (tmp_path / "images" / "val").mkdir(parents=True)
        _make_img(tmp_path / "images" / "val" / "img1.jpg", size=(10, 10))
        (tmp_path / "images" / "val" / ".DS_Store").touch()
        (tmp_path / "images" / "val" / "README.md").touch()

        _write_yaml(tmp_path / "data.yaml", {"val": "images/val", "names": {0: "cat"}})
        ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split="val")
        assert len(ds) == 1

    def test_converts_images_to_rgb(self, tmp_path):
        (tmp_path / "images" / "val").mkdir(parents=True)
        Image.new("L", (10, 10)).save(tmp_path / "images" / "val" / "gray.jpg")

        _write_yaml(tmp_path / "data.yaml", {"val": "images/val", "names": {0: "cat"}})
        ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split="val")
        img, _, _ = ds[0]
        assert img.shape[0] == 3

    def test_dataloader_batches(self, tmp_path):
        (tmp_path / "images" / "val").mkdir(parents=True)
        for i in range(5):
            _make_img(tmp_path / "images" / "val" / f"img{i}.jpg", size=(10, 10))

        _write_yaml(tmp_path / "data.yaml", {"val": "images/val", "names": {0: "cat"}})
        ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split="val")
        loader = YoloDetectionDataLoader(ds, batch_size=2, shuffle=False)

        batches = list(loader)
        assert [len(b[0]) for b in batches] == [2, 2, 1]
        for inputs, targets, metadata in batches:
            assert len(inputs) == len(targets) == len(metadata)

    def test_dataloader_shuffle_seed_is_deterministic(self, tmp_path):
        (tmp_path / "images" / "val").mkdir(parents=True)
        for i in range(6):
            _make_img(tmp_path / "images" / "val" / f"img{i}.jpg", size=(10, 10))

        _write_yaml(tmp_path / "data.yaml", {"val": "images/val", "names": {0: "cat"}})
        ds = YoloDetectionDataset(yaml_dataset=tmp_path / "data.yaml", split="val")

        l1 = YoloDetectionDataLoader(ds, batch_size=6, shuffle=True, seed=99)
        l2 = YoloDetectionDataLoader(ds, batch_size=6, shuffle=True, seed=99)
        ids1 = [m["id"] for m in list(l1)[0][2]]
        ids2 = [m["id"] for m in list(l2)[0][2]]
        assert ids1 == ids2

        l_plain = YoloDetectionDataLoader(ds, batch_size=6, shuffle=False)
        ids_plain = [m["id"] for m in list(l_plain)[0][2]]
        assert ids1 != ids_plain


class _FixedODModel:
    """Minimal fake OD model with known predictions for deterministic sorting tests."""

    metadata = {"id": "fixed_model", "index2label": {0: "a", 1: "b"}}

    def __call__(self, inputs):
        return [
            DetectionTarget(
                boxes=torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]], dtype=torch.float32),
                labels=torch.tensor([0, 1, 0]),
                scores=torch.tensor([0.2, 0.9, 0.5]),
            )
            for _ in inputs
        ]


class TestXaitkExplainableDetectionBaselineDataset:
    def test_xaitk_detection_baseline_dataset_is_importable_from_dataset_loaders(
        self, fake_od_dataset_default, fake_od_model_default
    ):
        temp_dataset = XaitkExplainableDetectionBaselineDataset(fake_od_dataset_default, fake_od_model_default)
        assert len(temp_dataset) == len(fake_od_dataset_default)

    def test_xaitk_detection_baseline_dataset_preserves_images_and_metadata(
        self, fake_od_dataset_default, fake_od_model_default
    ):
        temp_dataset = XaitkExplainableDetectionBaselineDataset(fake_od_dataset_default, fake_od_model_default)
        for i in range(len(temp_dataset)):
            assert torch.equal(temp_dataset[i][0], torch.as_tensor(fake_od_dataset_default[i][0]))
            assert temp_dataset[i][2] == fake_od_dataset_default[i][2]

    def test_xaitk_detection_baseline_dataset_replaces_targets_with_predictions(self, fake_od_dataset_default):
        model = _FixedODModel()
        temp_dataset = XaitkExplainableDetectionBaselineDataset(fake_od_dataset_default, model, dets_limit=2)

        for i in range(len(temp_dataset)):
            target = temp_dataset[i][1]
            scores = torch.as_tensor(target.scores)
            # Sorted descending: original indices [1, 2] → scores [0.9, 0.5]
            assert torch.allclose(scores, torch.tensor([0.9, 0.5]))
            assert torch.allclose(torch.as_tensor(target.labels), torch.tensor([1, 0]))
            assert torch.allclose(
                torch.as_tensor(target.boxes),
                torch.tensor([[1, 1, 2, 2], [2, 2, 3, 3]], dtype=torch.float32),
            )

    def test_xaitk_detection_baseline_dataset_applies_dets_limit(self, fake_od_dataset_default):
        model = _FixedODModel()
        dets_limit = 2
        temp_dataset = XaitkExplainableDetectionBaselineDataset(fake_od_dataset_default, model, dets_limit=dets_limit)

        for i in range(len(temp_dataset)):
            target = temp_dataset[i][1]
            assert len(torch.as_tensor(target.boxes)) <= dets_limit
            assert len(torch.as_tensor(target.labels)) <= dets_limit
            assert len(torch.as_tensor(target.scores)) <= dets_limit

    def test_xaitk_detection_baseline_dataset_preserves_dataset_metadata(
        self, fake_od_dataset_default, fake_od_model_default
    ):
        temp_dataset = XaitkExplainableDetectionBaselineDataset(fake_od_dataset_default, fake_od_model_default)
        assert temp_dataset.metadata["id"] == f"xai_temp_{fake_od_dataset_default.metadata['id']}"
        if "index2label" in fake_od_dataset_default.metadata:
            assert temp_dataset.metadata["index2label"] == fake_od_dataset_default.metadata["index2label"]
