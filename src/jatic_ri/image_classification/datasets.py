"""datasets"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
from PIL import Image

from jatic_ri.vendor.maite import DatasetMetadata, DatumMetadata

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ClassificationDatasetWrapperError(Exception):
    """Base class for catching errors in image classification dataset wrapper."""

    pass


class MissingYoloDataSplitError(ClassificationDatasetWrapperError):
    """The provided YOLO dataset is missing the requested data split."""

    pass


class YoloClassificationDataset:
    """
    A dataset handler for YOLO image classification datasets.

    This class is designed to load datasets formatted as per the YOLO image
    classification dataset specification. See the official documentation at
    https://docs.ultralytics.com/datasets/classify/ for more details.

    Attributes
    ----------
    metadata : DatasetMetadata
        A typed dictionary containing at least an 'id' field of type str
        and a index2label mapping from id to labels.

    Methods
    -------
    __getitem__(self, ind: int) -> tuple[NDArray, NDArray, DatumMetadataType]
        Provide map-style access to dataset elements. Returned tuple elements
        correspond to model input type, model target type, and datum-specific metadata type,
        respectively.

    __len__(self) -> int
        Return the number of data elements in the dataset.
    """

    def __init__(self, dataset_id: str, root_dir: str, split: Literal["train", "test", "validation"] = "test") -> None:
        """
        Args:
            dataset_id: Identifier for dataset that will be stored in dataset metadata. Name should
                be chosen to help users quickly identify what the dataset contains and/or how it was created.
            root_dir: Root directory of the dataset.
            split: Dataset split to use (e.g., "train", "test", "validation"). Defaults to "test".
        """

        try:
            # convention adopted is to order labels alphabetically
            self._images = sorted(self._get_filepaths_by_split(Path(f"{root_dir}/{split}")))
            labels = sorted(os.listdir(Path(f"{root_dir}/{split}")))
        except FileNotFoundError:
            raise MissingYoloDataSplitError(
                f"The following data split subdirectory does not exist {root_dir}/{split}"
            ) from None

        self._index2label = dict(enumerate(labels))  # 0-indexing
        self._label2index = {val: idx for idx, val in enumerate(labels)}  # 0-indexing

        self._metadata = DatasetMetadata({"id": dataset_id, "index2label": self._index2label})

    @staticmethod
    def _get_filepaths_by_split(dataset_split: Path) -> list[Path]:
        """
        Get the filepaths for images in a YOLO classification dataset structure.

        Args:
            dataset_split: Path to the dataset split directory e.g. "<dataset_root>/test"

        Returns:
            List of filepaths relative to dataset_split for all images in dataset
        """
        filepaths = []
        for class_dir in Path(dataset_split).iterdir():
            if class_dir.is_dir():
                filepaths.extend([filepath for filepath in class_dir.glob("*") if filepath.is_file()])
        return filepaths

    @property
    def metadata(self) -> DatasetMetadata:
        """Return typed dictionary with dataset metadata."""
        return self._metadata

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[NDArray[Any], NDArray[Any], DatumMetadata]:
        """Get `index`-th element from dataset."""

        try:
            image_path = self._images[index]
        except IndexError as e:
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self)}",
            ) from e

        img = Image.open(image_path)
        # PIL loads data as HWC, but MAITE requires CHW
        img_chw = np.array(img).transpose(2, 0, 1)

        one_hot_encode = np.zeros([len(self._label2index)])
        label = image_path.parent.name
        one_hot_encode[self._label2index[label]] = 1

        metadata: DatumMetadata = {"id": self.metadata["id"]}

        return img_chw, one_hot_encode, metadata


class DatasetSpecification(TypedDict):
    """Dataset metadata required for loading datasets via the RI wrappers"""

    # Dataset class as a string
    # TO DO hard-coded due to https://github.com/microsoft/pyright/issues/9194 and maite pyright<=1.1.320
    dataset_type: Literal["YoloClassificationDataset"]
    # Full filepath to the data directory to use. For yolo, this is the split dir.
    # The root directory of images is expected to be the parent of this directory.
    data_dir: str | Path


def load_datasets(datasets: dict[str, DatasetSpecification]) -> dict[str, YoloClassificationDataset]:
    """Simplified programmatic loading of datasets from on dictionary of
    DatasetSpecifications."""
    loaded = {}
    for name, dataset_metadata in datasets.items():
        if dataset_metadata["dataset_type"] == "YoloClassificationDataset":
            split_dir = Path(dataset_metadata["data_dir"]).stem
            root_path = Path(dataset_metadata["data_dir"]).parent.resolve()
            loaded[name] = YoloClassificationDataset(
                dataset_id="_".join([str(root_path), str(split_dir)]),
                root_dir=str(root_path),
                split=str(split_dir),  # type: ignore
            )
        else:
            raise RuntimeError(f"Dataset type {dataset_metadata['dataset_type']} is not supported.")
    return loaded
