from __future__ import annotations

import pytest

from jatic_ri.image_classification.datasets import (
    DatasetSpecification,
    MissingYoloDataSplitError,
    YoloClassificationDataset,
    load_datasets,
)


def test_dataset_initialization(fake_dataset):
    dataset_root, classes, num_images_per_class, _ = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    assert len(dataset) == len(classes) * num_images_per_class
    assert dataset.metadata["id"] == "test_dataset"
    assert dataset.metadata["index2label"] == dict(enumerate(classes))
    assert dataset.metadata["index2label"][0] == "cat"
    assert dataset.metadata["index2label"][1] == "dog"


def test_get_item(fake_dataset):
    dataset_root, classes, _, image_shape = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    img, label, metadata = dataset[0]

    width, height = image_shape
    assert img.shape == (3, height, width)  # CHW format
    assert len(label) == len(classes)
    assert label.sum() == 1  # One-hot encoded
    assert metadata["id"] == "test_dataset"


def test_iteration_over_dataset(fake_dataset):
    dataset_root, _, _, _ = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    count = 0
    for _ in dataset:
        count += 1

    assert count == len(dataset), "Iteration count does not match dataset length"


def test_missing_data_split(fake_dataset):
    with pytest.raises(
        MissingYoloDataSplitError,
        match="The following data split subdirectory does not exist",
    ):
        YoloClassificationDataset(dataset_id="test_dataset", root_dir=fake_dataset, split="train")


def test_load_datasets(fake_dataset):
    dataset_root, _, _, _ = fake_dataset
    spec: DatasetSpecification = {
        "dataset_type": "YoloClassificationDataset",
        "data_dir": str(dataset_root),
        "split_folder": "test",
    }
    datasets = {
        "dataset1": spec,
    }
    loaded = load_datasets(datasets=datasets)
    assert loaded


def test_different_splits_no_id_match(fake_dataset, tmp_path):
    """Test that datasets generate auto-generated IDs and they are not the same for different splits."""
    dataset_root, classes, num_images_per_class, _ = fake_dataset

    # Create first dataset with test split
    dataset1 = YoloClassificationDataset(root_dir=dataset_root, split="test")

    # Create second dataset with train split
    dataset2 = YoloClassificationDataset(root_dir=dataset_root, split="train")

    # Assert that the two datasets have different IDs
    assert dataset1.metadata["id"] != dataset2.metadata["id"]
