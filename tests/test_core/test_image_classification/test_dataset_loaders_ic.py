import os

import pytest
from PIL import Image

from jatic_ri.core.image_classification.dataset_loaders import (
    MissingYoloDataSplitError,
    YoloClassificationDataset,
    load_datasets,
)

CLASSES = ["cat", "dog"]
NUM_IMAGES_PER_CLASS = 4
IMG_SHAPE = (64, 128)


def create_fake_yolo_dataset(
    root_dir,
    split,
    classes,
    num_images_per_class,
    image_shape,
) -> None:
    """Create a fake YOLO dataset structure.

    Parameters
    ----------
    root_dir
        The root directory where the dataset will be created.
    split
        The dataset split (e.g., "train", "test").
    classes
        A list of class names.
    num_images_per_class
        The number of images to create for each class.
    image_shape
        The shape (width, height) of the images to create.
    """
    os.makedirs(root_dir / split, exist_ok=True)
    for class_name in classes:
        class_dir = root_dir / split / class_name
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_images_per_class):
            img = Image.new("RGB", image_shape, color=(i, i, i))
            img.save(class_dir / f"{i}_{class_name}.jpg")


@pytest.fixture(scope="session")
def fake_dataset(
    tmp_path_factory,
):
    """Create a fake YOLO dataset for testing.

    Parameters
    ----------
    tmp_path_factory
        Pytest fixture for creating temporary directories.

    Returns
    -------
        A tuple containing:
        - The root directory of the created dataset.
        - The list of class names.
        - The number of images per class.
        - The shape of the images.
    """
    dataset_root = tmp_path_factory.mktemp("yolo_dataset")

    for split in ["test", "train"]:
        create_fake_yolo_dataset(
            root_dir=dataset_root,
            split=split,
            classes=CLASSES,
            num_images_per_class=NUM_IMAGES_PER_CLASS,
            image_shape=IMG_SHAPE,
        )
    return str(dataset_root), CLASSES, NUM_IMAGES_PER_CLASS, IMG_SHAPE


def test_yolo_dataset_initialization(fake_dataset):
    dataset_root, classes, num_images_per_class, _ = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    assert len(dataset) == len(classes) * num_images_per_class
    assert dataset.metadata["id"] == "test_dataset"
    assert dataset.metadata["index2label"] == dict(enumerate(classes))
    assert dataset.metadata["index2label"][0] == CLASSES[0]
    assert dataset.metadata["index2label"][1] == CLASSES[1]


def test_yolo_get_item(fake_dataset):
    dataset_root, classes, _, image_shape = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    img, label, metadata = dataset[0]

    width, height = image_shape
    assert img.shape == (3, height, width)  # CHW format
    assert len(label) == len(classes)
    assert label.sum() == 1  # One-hot encoded
    assert metadata["id"] == "test_dataset"


def test_yolo_iteration_over_dataset(fake_dataset):
    dataset_root, _, _, _ = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    count = 0
    for _ in dataset:
        count += 1

    assert count == len(dataset), "Iteration count does not match dataset length"


def test_yolo_missing_data_split(fake_dataset):
    with pytest.raises(
        MissingYoloDataSplitError,
        match="The following data split subdirectory does not exist",
    ):
        YoloClassificationDataset(dataset_id="test_dataset", root_dir=fake_dataset, split="train")


def test_yolo_load_datasets(fake_dataset):
    dataset_root, _, _, _ = fake_dataset
    spec = {
        "dataset_type": "YoloClassificationDataset",
        "data_dir": str(dataset_root),
        "split_folder": "test",
    }
    datasets = {
        "dataset1": spec,
    }
    loaded = load_datasets(datasets=datasets)

    assert loaded


def test_yolo_different_splits_no_id_match(fake_dataset):
    dataset_root, _, _, _ = fake_dataset

    dataset1 = YoloClassificationDataset(root_dir=dataset_root, split="test")

    dataset2 = YoloClassificationDataset(root_dir=dataset_root, split="train")

    assert dataset1.metadata["id"] != dataset2.metadata["id"]
