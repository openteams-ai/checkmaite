"""Unit tests for fsspec-based dataset loading.

These tests verify that the dataset loaders correctly use the fsspec filesystem
contract, using fsspec's in-memory filesystem. This ensures our code will work
with any fsspec-compatible backend (S3, GCS, Azure, etc.) without needing to
test against actual cloud services.
"""

import io
from pathlib import Path

import fsspec
import numpy as np
import pytest
import torch
from PIL import Image
from upath import UPath

from checkmaite.core.image_classification.dataset_loaders import (
    YoloClassificationDataset,
)
from checkmaite.core.object_detection.dataset_loaders import (
    CocoDetectionDataset,
    DetectionTarget,
    VisdroneDetectionDataset,
    YoloDetectionDataset,
)

# Test data directory
TEST_DATA_DIR = Path(__file__).parents[1] / "data_for_tests"


@pytest.fixture
def memory_coco_dataset():
    """Create a COCO dataset in fsspec memory filesystem."""
    fs = fsspec.filesystem("memory")

    # Copy test data to memory filesystem
    local_coco = TEST_DATA_DIR / "coco_dataset"

    # Copy annotation file
    ann_file = local_coco / "ann_file.json"
    fs.pipe_file("memory://test-bucket/coco_dataset/ann_file.json", ann_file.read_bytes())

    # Copy images
    for img_path in local_coco.glob("*.jpg"):
        fs.pipe_file(f"memory://test-bucket/coco_dataset/{img_path.name}", img_path.read_bytes())
    for img_path in local_coco.glob("*.png"):
        fs.pipe_file(f"memory://test-bucket/coco_dataset/{img_path.name}", img_path.read_bytes())

    yield {
        "root": "memory://test-bucket/coco_dataset",
        "ann_file": "memory://test-bucket/coco_dataset/ann_file.json",
    }

    # Cleanup
    fs.rm("memory://test-bucket", recursive=True)


@pytest.fixture
def memory_visdrone_dataset():
    """Create a VisDrone dataset in fsspec memory filesystem."""
    fs = fsspec.filesystem("memory")

    local_visdrone = TEST_DATA_DIR / "visdrone_dataset"

    # Copy images
    images_dir = local_visdrone / "images"
    for img_path in images_dir.glob("*"):
        if img_path.is_file():
            fs.pipe_file(f"memory://test-bucket/visdrone_dataset/images/{img_path.name}", img_path.read_bytes())

    # Copy annotations
    ann_dir = local_visdrone / "annotations"
    for ann_path in ann_dir.glob("*"):
        if ann_path.is_file():
            fs.pipe_file(f"memory://test-bucket/visdrone_dataset/annotations/{ann_path.name}", ann_path.read_bytes())

    yield {"root": "memory://test-bucket/visdrone_dataset"}

    fs.rm("memory://test-bucket", recursive=True)


@pytest.fixture
def memory_yolo_dataset():
    """Create a YOLO dataset in fsspec memory filesystem."""
    fs = fsspec.filesystem("memory")

    local_yolo = TEST_DATA_DIR / "yolo_dataset"

    # Copy yaml file
    yaml_file = local_yolo / "dataset.yaml"
    # Modify the yaml to use memory:// paths
    yaml_content = yaml_file.read_text()
    yaml_content = yaml_content.replace(str(local_yolo), "memory://test-bucket/yolo_dataset")
    fs.pipe_file("memory://test-bucket/yolo_dataset/dataset.yaml", yaml_content.encode())

    # Copy images directory
    images_dir = local_yolo / "images"
    if images_dir.exists():
        for img_path in images_dir.rglob("*"):
            if img_path.is_file():
                rel_path = img_path.relative_to(local_yolo)
                fs.pipe_file(f"memory://test-bucket/yolo_dataset/{rel_path}", img_path.read_bytes())

    # Copy annotations directory
    ann_dir = local_yolo / "ann_dir"
    if ann_dir.exists():
        for ann_path in ann_dir.rglob("*"):
            if ann_path.is_file():
                rel_path = ann_path.relative_to(local_yolo)
                fs.pipe_file(f"memory://test-bucket/yolo_dataset/{rel_path}", ann_path.read_bytes())

    yield {
        "yaml": "memory://test-bucket/yolo_dataset/dataset.yaml",
        "ann_dir": "memory://test-bucket/yolo_dataset/ann_dir",
    }

    fs.rm("memory://test-bucket", recursive=True)


class TestMemoryFilesystemDatasetLoading:
    """Tests for loading datasets using fsspec's memory filesystem.

    This verifies our code correctly uses the fsspec contract without
    needing actual cloud credentials or mocks.
    """

    def test_coco_detection_from_memory_fs(self, memory_coco_dataset):
        """Test loading CocoDetectionDataset from memory filesystem."""
        dataset = CocoDetectionDataset(
            root=memory_coco_dataset["root"],
            ann_file=memory_coco_dataset["ann_file"],
            dataset_id="memory_coco_test",
        )

        assert len(dataset) == 4
        assert dataset.metadata["id"] == "memory_coco_test"

        image, target, metadata = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.ndim == 3
        assert isinstance(target, DetectionTarget)
        assert isinstance(metadata, dict)

    def test_visdrone_detection_from_memory_fs(self, memory_visdrone_dataset):
        """Test loading VisdroneDetectionDataset from memory filesystem."""
        dataset = VisdroneDetectionDataset(
            root=memory_visdrone_dataset["root"],
            dataset_id="memory_visdrone_test",
        )

        assert len(dataset) == 3
        assert dataset.metadata["id"] == "memory_visdrone_test"

        image, target, metadata = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(target, DetectionTarget)
        assert isinstance(metadata, dict)

    def test_yolo_detection_from_memory_fs(self, memory_yolo_dataset):
        """Test loading YoloDetectionDataset from memory filesystem."""
        dataset = YoloDetectionDataset(
            yaml_dataset=memory_yolo_dataset["yaml"],
            ann_dir=memory_yolo_dataset["ann_dir"],
            dataset_id="memory_yolo_test",
        )

        assert len(dataset) == 4
        assert dataset.metadata["id"] == "memory_yolo_test"

        image, target, metadata = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.ndim == 3
        assert isinstance(target, DetectionTarget)
        assert isinstance(metadata, dict)


class TestYoloClassificationWithFsspec:
    """Tests for YoloClassificationDataset with fsspec filesystems."""

    @pytest.fixture
    def memory_yolo_classification_dataset(self):
        """Create a YOLO classification dataset in memory filesystem."""
        fs = fsspec.filesystem("memory")
        classes = ["cat", "dog"]

        for split in ["test", "train"]:
            for class_name in classes:
                for i in range(2):
                    img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    fs.pipe_file(
                        f"memory://test-bucket/yolo_classification/{split}/{class_name}/{i}_{class_name}.jpg",
                        buf.getvalue(),
                    )

        yield {"root": "memory://test-bucket/yolo_classification", "classes": classes}

        fs.rm("memory://test-bucket", recursive=True)

    def test_yolo_classification_from_memory_fs(self, memory_yolo_classification_dataset):
        """Test YoloClassificationDataset with memory filesystem."""
        dataset = YoloClassificationDataset(
            root_dir=memory_yolo_classification_dataset["root"],
            split="test",
            dataset_id="memory_classification_test",
        )

        assert len(dataset) == len(memory_yolo_classification_dataset["classes"]) * 2
        assert dataset.metadata["id"] == "memory_classification_test"

        image, _, metadata = dataset[0]
        assert image.shape[0] == 3  # CHW format
        assert isinstance(metadata, dict)


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with local filesystem paths."""

    COCO_ROOT = str(TEST_DATA_DIR / "coco_dataset")
    COCO_ANN_FILE = str(TEST_DATA_DIR / "coco_dataset" / "ann_file.json")
    VISDRONE_ROOT = str(TEST_DATA_DIR / "visdrone_dataset")
    YOLO_ROOT = TEST_DATA_DIR / "yolo_dataset"
    YOLO_YAML = str(YOLO_ROOT / "dataset.yaml")
    YOLO_ANN_DIR = str(YOLO_ROOT / "ann_dir")

    def test_coco_detection_local_path_still_works(self):
        """Verify CocoDetectionDataset still works with local string paths."""
        dataset = CocoDetectionDataset(
            root=self.COCO_ROOT,
            ann_file=self.COCO_ANN_FILE,
            dataset_id="local_coco_test",
        )

        assert len(dataset) == 4
        image, target, metadata = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(target, DetectionTarget)
        assert isinstance(metadata, dict)

    def test_visdrone_detection_local_path_still_works(self):
        """Verify VisdroneDetectionDataset still works with local paths."""
        dataset = VisdroneDetectionDataset(
            root=self.VISDRONE_ROOT,
            dataset_id="local_visdrone_test",
        )

        assert len(dataset) == 3
        image, target, metadata = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(target, DetectionTarget)
        assert isinstance(metadata, dict)

    def test_yolo_detection_local_path_still_works(self):
        """Verify YoloDetectionDataset still works with local paths."""
        dataset = YoloDetectionDataset(
            yaml_dataset=self.YOLO_YAML,
            ann_dir=self.YOLO_ANN_DIR,
            dataset_id="local_yolo_test",
        )

        assert len(dataset) == 4
        assert dataset.metadata["id"] == "local_yolo_test"
        image, target, metadata = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(target, DetectionTarget)
        assert isinstance(metadata, dict)


class TestImageLoadingEquivalence:
    """Tests to verify fsspec-based image loading produces identical results to direct PIL loading."""

    def test_upath_loading_matches_direct_pil_loading(self, tmp_path):
        """Verify UPath-based loading produces byte-for-byte identical results to Image.open(path)."""
        img_path = tmp_path / "test.png"
        test_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        test_img.save(img_path)

        # Original direct PIL approach
        img_direct = Image.open(img_path)
        result_direct = np.array(img_direct).transpose(2, 0, 1)

        # Current fsspec-compatible approach (mirrors dataset_loaders implementation)
        upath = UPath(img_path)
        with upath.open("rb") as f, Image.open(f) as img_upath:
            img_upath = img_upath.convert("RGB")
            arr = np.asarray(img_upath)
        result_upath = np.moveaxis(arr, -1, 0)

        np.testing.assert_array_equal(result_direct, result_upath)
