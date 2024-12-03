from __future__ import annotations

import os

import pytest
from PIL import Image

from jatic_ri.image_classification.datasets import MissingYoloDataSplitError, YoloClassificationDataset

CLASSES = ["cat", "dog"]
NUM_IMAGES_PER_CLASS = 3
IMG_SHAPE = (64, 128)


def create_fake_yolo_dataset(root_dir, split, classes, num_images_per_class):
    os.makedirs(root_dir / split, exist_ok=True)
    for class_name in classes:
        class_dir = root_dir / split / class_name
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_images_per_class):
            img = Image.new("RGB", IMG_SHAPE, color=(i, i, i))
            img.save(class_dir / f"{i}_{class_name}.jpg")


@pytest.fixture
def fake_dataset(tmpdir):
    dataset_root = tmpdir / "yolo_dataset"
    create_fake_yolo_dataset(
        root_dir=dataset_root,
        split="test",
        classes=CLASSES,
        num_images_per_class=NUM_IMAGES_PER_CLASS,
    )
    return dataset_root


def test_dataset_initialization(fake_dataset):
    dataset = YoloClassificationDataset(dataset_name="test_dataset", root_dir=fake_dataset, split="test")
    assert len(dataset) == len(CLASSES) * NUM_IMAGES_PER_CLASS
    assert dataset.metadata["id"] == "test_dataset"
    assert dataset.metadata["index2label"] == dict(enumerate(CLASSES))


def test_get_item(fake_dataset):
    dataset = YoloClassificationDataset(dataset_name="test_dataset", root_dir=fake_dataset, split="test")
    img, label, metadata = dataset[0]

    width, height = IMG_SHAPE
    assert img.shape == (3, height, width)  # CHW format
    assert len(label) == len(CLASSES)
    assert label.sum() == 1  # One-hot encoded
    assert metadata["id"] == "test_dataset"


def test_iteration_over_dataset(fake_dataset):
    dataset = YoloClassificationDataset(dataset_name="test_dataset", root_dir=fake_dataset, split="test")
    count = 0
    for _ in dataset:
        count += 1

    assert count == len(dataset), "Iteration count does not match dataset length"


def test_missing_data_split(fake_dataset):
    with pytest.raises(
        MissingYoloDataSplitError,
        match="The following data split subdirectory does not exist",
    ):
        YoloClassificationDataset(dataset_name="test_dataset", root_dir=fake_dataset, split="train")
