from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
from maite.protocols import DatasetMetadata, DatumMetadata
from PIL import Image
from upath import UPath

from jatic_ri.core._utils import id_hash

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ClassificationDatasetWrapperError(Exception):
    """Base class for catching errors in image classification dataset wrapper."""

    pass


class MissingYoloDataSplitError(ClassificationDatasetWrapperError):
    """The provided YOLO dataset is missing the requested data split."""

    pass


class YoloClassificationDataset:
    """A dataset handler for YOLO image classification datasets.

    This class is designed to load datasets formatted as per the YOLO image
    classification dataset specification. See the official documentation at
    https://docs.ultralytics.com/datasets/classify/ for more details.

    Attributes
    ----------
    metadata
        A typed dictionary containing at least an 'id' field of type str
        and a index2label mapping from id to labels.

    """

    def __init__(
        self, root_dir: str, dataset_id: str | None = None, split: Literal["train", "test", "validation"] = "test"
    ) -> None:
        """Initialize YoloClassificationDataset.

        Parameters
        ----------
        root_dir
            Root directory of the dataset.
        dataset_id
            Optional identifier for dataset. If omitted, a unique one will be
            generated from the other input arguments. By default None.
        split
            Dataset split to use (e.g., "train", "test", "validation").
            Defaults to "test".

        Raises
        ------
        If the specified data split subdirectory does not exist.
        """
        split_path = UPath(root_dir) / split

        if not split_path.exists():
            raise MissingYoloDataSplitError(f"The following data split subdirectory does not exist {split_path}")

        # convention adopted is to order labels alphabetically
        self._images = sorted(self._get_filepaths_by_split(split_path))
        labels = sorted([p.name for p in split_path.iterdir() if p.is_dir()])

        self._index2label = dict(enumerate(labels))  # 0-indexing
        self._label2index = {val: idx for idx, val in enumerate(labels)}  # 0-indexing

        # Generate dataset_id if not provided
        if dataset_id is None:
            dataset_id = f"yolo_classification_{id_hash(root_dir=root_dir, split=split)}"
        self._metadata = DatasetMetadata({"id": dataset_id, "index2label": self._index2label})

    @staticmethod
    def _get_filepaths_by_split(dataset_split: UPath) -> list[UPath]:
        """Get the filepaths for images in a YOLO classification dataset structure.

        Parameters
        ----------
        dataset_split : UPath
            Path to the dataset split directory e.g. "<dataset_root>/test".

        list[UPath]
            List of filepaths relative to `dataset_split` for all images in dataset.
        """
        filepaths: list[UPath] = []
        for class_dir in dataset_split.iterdir():
            if class_dir.is_dir():
                filepaths.extend([filepath for filepath in class_dir.iterdir() if filepath.is_file()])
        return filepaths

    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata.

        Returns
        -------
        Typed dictionary with dataset metadata.
        """
        return self._metadata

    def __len__(self) -> int:
        """Length of the dataset.

        Returns
        -------
        The number of items in the dataset.
        """
        return len(self._images)

    def __getitem__(self, index: int) -> tuple["NDArray[Any]", "NDArray[Any]", DatumMetadata]:
        """Get `index`-th element from dataset.

        Parameters
        ----------
        index
            Index of the element to retrieve.

        Returns
        -------
            A tuple containing:
            - Image data as a NumPy array (CHW format).
            - One-hot encoded label as a NumPy array.
            - Datum metadata.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        try:
            image_path = self._images[index]
        except IndexError as e:
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self)}",
            ) from e

        # Use UPath to support both local and remote filesystems
        with image_path.open("rb") as f, Image.open(f) as img:
            img = img.convert("RGB")
            arr = np.asarray(img)  # HWC

        # PIL loads data as HWC, but MAITE requires CHW
        img_chw = np.moveaxis(arr, -1, 0)

        one_hot_encode = np.zeros([len(self._label2index)])
        label = image_path.parent.name
        one_hot_encode[self._label2index[label]] = 1

        metadata: DatumMetadata = {"id": f"{label}/{image_path.name}"}

        return img_chw, one_hot_encode, metadata


class DatasetSpecification(TypedDict):
    """Dataset metadata required for loading datasets via the RI wrappers.

    Attributes
    ----------
    dataset_type : Literal["YoloClassificationDataset"]
        Dataset class as a string.
        TODO: hard-coded due to https://github.com/microsoft/pyright/issues/9194
        and maite pyright<=1.1.320
    data_dir : str
        Full filepath to the data directory to use. For YOLO, this is the split
        directory. The root directory of images is expected to be the parent of
        this directory.
    split_folder : Literal["train", "test", "validation"]
        Folder name of the split folder to load inside of the root dataset
        directory.
    """

    # Dataset class as a string
    # TODO: hard-coded due to https://github.com/microsoft/pyright/issues/9194 and maite pyright<=1.1.320
    dataset_type: Literal["YoloClassificationDataset"]
    # Full filepath to the data directory to use. For YOLO, this is the split dir.
    # The root directory of images is expected to be the parent of this directory.
    data_dir: str
    # Folder name of the split folder to load inside of the root dataset directory.
    split_folder: Literal["train", "test", "validation"]


def load_datasets(datasets: dict[str, DatasetSpecification]) -> dict[str, YoloClassificationDataset]:
    """Simplified programmatic loading of datasets from a dictionary of DatasetSpecifications.

    Parameters
    ----------
    datasets : dict[str, DatasetSpecification]
        A dictionary where keys are dataset names and values are DatasetSpecification
        objects.

    Returns
    -------
    dict[str, YoloClassificationDataset]
        A dictionary of loaded datasets, where keys are dataset names and values
        are YoloClassificationDataset instances.

    Raises
    ------
    RuntimeError
        If an unsupported dataset type is encountered.
    """
    loaded = {}
    for name, dataset_metadata in datasets.items():
        if dataset_metadata["dataset_type"] == "YoloClassificationDataset":
            loaded[name] = YoloClassificationDataset(
                root_dir=dataset_metadata["data_dir"],
                split=dataset_metadata["split_folder"],
            )
        else:
            raise RuntimeError(f"Dataset type {dataset_metadata['dataset_type']} is not supported.")
    return loaded
