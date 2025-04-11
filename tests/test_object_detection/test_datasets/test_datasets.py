from pathlib import Path

import pytest

from jatic_ri.object_detection.datasets import DatasetSpecification, load_datasets

YOLO_METADATA_PATH = "tests/test_object_detection/data/yolo_dataset/dataset.yaml"
YOLO_ANNOTATION_DIR = "tests/test_object_detection/data/yolo_dataset/ann_dir"
COCO_ANNOTATION_PATH = "tests/testing_utilities/example_data/coco_dataset/ann_file.json"


def test_load_datasets_yolo():
    """Test loading a yolo dataset through the programmatic loader"""
    spec: DatasetSpecification = {
        "dataset_type": "YoloDetectionDataset",
        "metadata_path": YOLO_METADATA_PATH,
        "data_dir": YOLO_ANNOTATION_DIR,
    }
    datasets = {
        "yolo_dataset1": spec,
    }
    loaded = load_datasets(datasets=datasets)
    assert "yolo_dataset1" in loaded


def test_load_datasets_coco():
    """Test loading a coco dataset through the programmatic loader"""
    spec: DatasetSpecification = {
        "dataset_type": "CocoDetectionDataset",
        "metadata_path": COCO_ANNOTATION_PATH,
        "data_dir": Path(COCO_ANNOTATION_PATH).parent.resolve(),
    }

    datasets = {
        "coco_dataset1": spec,
    }
    loaded = load_datasets(datasets=datasets)
    assert "coco_dataset1" in loaded


def test_load_datasets_visdrone():
    """Test loading a visdrone dataset through the programmatic loader"""
    spec: DatasetSpecification = {
        "dataset_type": "VisdroneDetectionDataset",
        "metadata_path": "",
        "data_dir": Path(__file__).parents[2] / "testing_utilities" / "example_data" / "visdrone_dataset",
    }

    key = "visdrone_dataset1"
    loaded = load_datasets(datasets={key: spec})
    assert key in loaded


def test_load_datasets_invalid():
    """Test loading an invalid dataset"""
    spec: DatasetSpecification = {
        "dataset_type": "NonexistentClass",
    }
    datasets_invalid = {
        "dataset_invalid": spec,
    }
    with pytest.raises(RuntimeError, match=r"\bNonexistentClass\b"):
        load_datasets(datasets=datasets_invalid)
