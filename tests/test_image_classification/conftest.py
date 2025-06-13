from __future__ import annotations

import os

import pytest
from PIL import Image

CLASSES = ["cat", "dog"]
NUM_IMAGES_PER_CLASS = 4
IMG_SHAPE = (64, 128)


def create_fake_yolo_dataset(root_dir, split, classes, num_images_per_class, image_shape):
    os.makedirs(root_dir / split, exist_ok=True)
    for class_name in classes:
        class_dir = root_dir / split / class_name
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_images_per_class):
            img = Image.new("RGB", image_shape, color=(i, i, i))
            img.save(class_dir / f"{i}_{class_name}.jpg")


@pytest.fixture(scope="session")
def fake_dataset(tmp_path_factory):
    dataset_root = tmp_path_factory.mktemp("yolo_dataset")
    # Create both test and train splits
    for split in ["test", "train"]:
        create_fake_yolo_dataset(
            root_dir=dataset_root,
            split=split,
            classes=CLASSES,
            num_images_per_class=NUM_IMAGES_PER_CLASS,
            image_shape=IMG_SHAPE,
        )
    return str(dataset_root), CLASSES, NUM_IMAGES_PER_CLASS, IMG_SHAPE
