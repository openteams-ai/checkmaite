import os
import time

import numpy as np
import pytest
from PIL import Image

from checkmaite.core.image_classification.dataset_loaders import (
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
    assert isinstance(metadata["id"], str)
    assert "/" in metadata["id"]  # Should be in format "class/filename"


def test_yolo_unique_datum_ids(fake_dataset):
    dataset_root, _, _, _ = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")

    datum_ids = set()
    for i in range(len(dataset)):
        _, _, metadata = dataset[i]
        datum_id = metadata["id"]
        assert datum_id not in datum_ids, f"Duplicate datum ID found: {datum_id}"
        datum_ids.add(datum_id)

    assert len(datum_ids) == len(dataset), "Not all datums have unique IDs"


def test_yolo_iteration_over_dataset(fake_dataset):
    dataset_root, _, _, _ = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    count = 0
    for _ in dataset:
        count += 1

    assert count == len(dataset), "Iteration count does not match dataset length"


def test_yolo_missing_data_split(tmp_path):
    # Create a directory without the expected split subdirectory
    dataset_root = tmp_path / "empty_dataset"
    dataset_root.mkdir()
    with pytest.raises(
        MissingYoloDataSplitError,
        match="The following data split subdirectory does not exist",
    ):
        YoloClassificationDataset(dataset_id="test_dataset", root_dir=str(dataset_root), split="validation")


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


def test_get_input(fake_dataset):
    dataset_root, _, _, image_shape = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    image = dataset.get_input(0)

    width, height = image_shape
    assert isinstance(image, np.ndarray)
    assert image.shape == (3, height, width)


def test_get_target_does_not_load_image(fake_dataset):
    """Test that get_target returns the label without loading the image."""
    dataset_root, classes, _, _ = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    target = dataset.get_target(0)

    assert isinstance(target, np.ndarray)
    assert len(target) == len(classes)
    assert target.sum() == 1  # One-hot encoded


def test_get_metadata_does_not_load_image(fake_dataset):
    """Test that get_metadata returns without loading the image."""
    dataset_root, _, _, _ = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    metadata = dataset.get_metadata(0)

    assert isinstance(metadata, dict)
    assert "id" in metadata
    assert "/" in metadata["id"]  # Should be in format "class/filename"


def test_fieldwise_methods_consistent_with_getitem(fake_dataset):
    dataset_root, _, _, _ = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")

    for i in range(len(dataset)):
        image_full, target_full, metadata_full = dataset[i]
        image_field = dataset.get_input(i)
        target_field = dataset.get_target(i)
        metadata_field = dataset.get_metadata(i)

        assert np.array_equal(image_full, image_field)
        assert np.array_equal(target_full, target_field)
        assert metadata_full == metadata_field


def test_get_input_index_error(fake_dataset):
    dataset_root, _, _, _ = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    with pytest.raises(IndexError):
        dataset.get_input(100)


def test_get_target_index_error(fake_dataset):
    dataset_root, _, _, _ = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    with pytest.raises(IndexError):
        dataset.get_target(100)


def test_get_metadata_index_error(fake_dataset):
    dataset_root, _, _, _ = fake_dataset
    dataset = YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")
    with pytest.raises(IndexError):
        dataset.get_metadata(100)


class TestAccessorMethodPerformance:
    """Performance tests comparing accessor methods to __getitem__.

    The accessor methods (get_input, get_target, get_metadata) are designed to
    provide performance benefits when only a subset of the data is needed:
    - get_target: Skips image loading, should be significantly faster than __getitem__
    - get_metadata: Skips image loading, should be significantly faster than __getitem__
    - get_input: Still loads the image, expected to have similar performance to __getitem__
    """

    def test_get_target_faster_than_getitem(self, fake_dataset):
        """Verify get_target is faster than __getitem__ by skipping image loading."""
        dataset_root, _, _, _ = fake_dataset
        dataset = YoloClassificationDataset(dataset_id="test_perf", root_dir=dataset_root, split="test")
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

    def test_get_metadata_faster_than_getitem(self, fake_dataset):
        """Verify get_metadata is faster than __getitem__ by skipping image loading."""
        dataset_root, _, _, _ = fake_dataset
        dataset = YoloClassificationDataset(dataset_id="test_perf", root_dir=dataset_root, split="test")
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
