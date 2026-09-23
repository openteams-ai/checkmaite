"""Tests for checkmaite's native-datamaite IC loading boundary."""

from pathlib import Path

import numpy as np
import pytest
from datamaite.image_classification import ImageClassificationDataset
from PIL import Image

from checkmaite.core.image_classification.dataset_loaders import (
    MissingYoloDataSplitError,
    YoloClassificationDataLoader,
    load_datasets,
    load_yolo_classification_dataset,
)


def _make_dataset(root: Path) -> None:
    for split in ("train", "test", "val"):
        for class_name in ("cat", "dog"):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            Image.new("RGB", (16, 12), color=(1, 2, 3)).save(class_dir / f"{class_name}.jpg")


def test_factory_returns_native_datamaite_dataset(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    dataset = load_yolo_classification_dataset(tmp_path, split="test", dataset_id="ic-test")

    assert type(dataset) is ImageClassificationDataset
    assert dataset.metadata["id"] == "ic-test"
    assert dataset.metadata["index2label"] == {0: "cat", 1: "dog"}

    image, target, metadata = dataset[0]
    assert isinstance(image, np.ndarray)
    assert image.shape == (3, 12, 16)
    assert isinstance(target, np.ndarray)
    assert target.dtype == np.float32
    assert target.sum() == 1
    # The datum id is relative to the dataset root, so it carries the split:
    # checkmaite now hands datamaite the root plus ``split=`` instead of the
    # split directory as its own root.
    assert metadata["id"] == "test/cat/cat.jpg"


def test_validation_alias_uses_val_folder(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    dataset = load_yolo_classification_dataset(tmp_path, split="validation")
    assert len(dataset) == 2


def test_missing_split_preserves_actionable_error(tmp_path: Path) -> None:
    # datamaite warns and yields an empty dataset; checkmaite's policy is to
    # fail loudly, and to say which splits the root actually has.
    with pytest.raises(MissingYoloDataSplitError, match="data split 'test'"):
        load_yolo_classification_dataset(tmp_path, split="test")


def test_empty_split_directory_also_fails_loudly(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    for image in (tmp_path / "test").rglob("*.jpg"):
        image.unlink()
    with pytest.raises(MissingYoloDataSplitError, match=r"Split subdirectories present: \['test', 'train', 'val'\]"):
        load_yolo_classification_dataset(tmp_path, split="test")


def test_nested_images_below_class_directories_are_discovered(tmp_path: Path) -> None:
    # Regression boundary for the reader this migration removed: images nested
    # below a class directory keep the top-level directory as their class and
    # keep the nested path in the datum id (datamaite #90). A nested-only split
    # must not load as an empty dataset.
    for class_name, subdir in (("cat", "roll-01"), ("dog", "roll-02/day-1")):
        nested = tmp_path / "test" / class_name / subdir
        nested.mkdir(parents=True)
        Image.new("RGB", (16, 12), color=(1, 2, 3)).save(nested / f"{class_name}.jpg")

    dataset = load_yolo_classification_dataset(tmp_path, split="test")

    assert len(dataset) == 2
    assert dataset.metadata["index2label"] == {0: "cat", 1: "dog"}
    datum_ids = sorted(dataset.get_metadata(index)["id"] for index in range(len(dataset)))
    assert datum_ids == ["test/cat/roll-01/cat.jpg", "test/dog/roll-02/day-1/dog.jpg"]
    # The nested directories are path, not taxonomy.
    assert "roll-01" not in dataset.metadata["index2label"].values()


def test_mixed_flat_and_nested_images_load_together(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    nested = tmp_path / "test" / "cat" / "roll-01"
    nested.mkdir()
    Image.new("RGB", (16, 12), color=(4, 5, 6)).save(nested / "nested.jpg")

    dataset = load_yolo_classification_dataset(tmp_path, split="test")

    # Flat 2 (cat, dog) + 1 nested: a mixed split must not silently drop the
    # nested image.
    assert len(dataset) == 3


def test_native_dataset_provides_maite_fieldwise_access(tmp_path: Path) -> None:
    # datamaite >=0.4.0 implements the MAITE fieldwise accessors natively;
    # checkmaite intentionally has no adapter layer re-adding them.
    _make_dataset(tmp_path)
    dataset = load_yolo_classification_dataset(tmp_path, split="test")
    image, target, metadata = dataset[0]
    np.testing.assert_array_equal(dataset.get_input(0), image)
    np.testing.assert_array_equal(dataset.get_target(0), target)
    assert dataset.get_metadata(0) == metadata


def test_taxonomy_includes_empty_class_directories(tmp_path: Path) -> None:
    # datamaite >=0.4.0 derives the split-local taxonomy from all class
    # directories, not just image-bearing ones, so label indices stay stable
    # when one split is missing a class's images.
    _make_dataset(tmp_path)
    (tmp_path / "test" / "zebra").mkdir()
    dataset = load_yolo_classification_dataset(tmp_path, split="test")
    assert dataset.metadata["index2label"] == {0: "cat", 1: "dog", 2: "zebra"}
    assert len(dataset) == 2


def test_load_datasets_dispatches_by_format_not_class_name(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    loaded = load_datasets(
        {
            "evaluation": {
                "dataset_format": "yolo",
                "data_dir": str(tmp_path),
                "split_folder": "test",
            }
        }
    )
    assert type(loaded["evaluation"]) is ImageClassificationDataset


def test_batch_loader_accepts_native_maite_dataset(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    dataset = load_yolo_classification_dataset(tmp_path, split="test")
    batches = list(YoloClassificationDataLoader(dataset, batch_size=1))
    assert len(batches) == len(dataset)
    assert all(len(inputs) == len(targets) == len(metadata) == 1 for inputs, targets, metadata in batches)
