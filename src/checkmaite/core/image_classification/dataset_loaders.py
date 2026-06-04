import random
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
from maite.protocols import DatasetMetadata, DatumMetadata
from maite.protocols.image_classification import FieldwiseDataset
from PIL import Image
from upath import UPath

from checkmaite.core._common.dataset_utils import _is_image_path
from checkmaite.core._utils import id_hash

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ClassificationDatasetWrapperError(Exception):
    """Base class for catching errors in image classification dataset wrapper."""

    pass


class MissingYoloDataSplitError(ClassificationDatasetWrapperError):
    """The provided YOLO dataset is missing the requested data split."""

    pass


class YoloClassificationDataset(FieldwiseDataset):
    """A dataset handler for YOLO image classification datasets.

    This class is designed to load datasets formatted as per the YOLO image
    classification dataset specification. See the official documentation at
    https://docs.ultralytics.com/datasets/classify/ for more details.

    Attributes
    ----------
    metadata
        A typed dictionary containing at least an 'id' field of type str
        and a index2label mapping from id to labels.

    Methods
    -------
    __getitem__(index)
        Provide mapping-style access to dataset elements.
    __len__()
        Return the number of data elements in the dataset.
    get_input(index)
        Get only the image data at the given index.
    get_target(index)
        Get only the target label at the given index.
    get_metadata(index)
        Get only the metadata at the given index.

    """

    def __init__(
        self,
        root_dir: str | UPath,
        dataset_id: str | None = None,
        split: Literal["train", "val", "test", "validation"] = "test",
    ) -> None:
        """Initialize YoloClassificationDataset.

        Parameters
        ----------
        root_dir
            Root directory of the YOLO classification dataset containing split
            folders (e.g., train/, val/, test/).
        dataset_id
            Optional identifier for dataset. If omitted, a unique one will be
            generated from the other input arguments. By default None.
        split
            Dataset split to use. Accepted values: "train", "val", "test".
            The value "validation" is also accepted for backward compatibility:
            if a ``validation/`` directory exists it is used directly, otherwise
            it falls back to ``val/``.
            Defaults to "test".

        Raises
        ------
        MissingYoloDataSplitError
            If the resolved split subdirectory does not exist.
        """
        root = UPath(root_dir)
        split_path = root / split

        if split == "validation" and not split_path.exists():
            split_path = root / "val"

        if not split_path.exists():
            raise MissingYoloDataSplitError(f"The following data split subdirectory does not exist {split_path}")

        self._split_path = split_path

        # convention adopted is to order labels alphabetically
        self._images = sorted(self._get_filepaths_by_split(split_path))
        labels = sorted([p.name for p in split_path.iterdir() if p.is_dir()])

        self._index2label = dict(enumerate(labels))  # 0-indexing
        self._label2index = {val: idx for idx, val in enumerate(labels)}  # 0-indexing

        # Generate dataset_id if not provided
        if dataset_id is None:
            dataset_id = f"yolo_classification_{id_hash(root_dir=root_dir, split=split)}"
        self.metadata = DatasetMetadata({"id": dataset_id, "index2label": self._index2label})

    @staticmethod
    def _get_filepaths_by_split(dataset_split: UPath) -> list[UPath]:
        """Get the filepaths for images in a YOLO classification dataset structure.

        Parameters
        ----------
        dataset_split : UPath
            Path to the dataset split directory e.g. "<dataset_root>/val".

        Returns
        -------
        list[UPath]
            List of image filepaths for all images under the split directory.
            Non-image files (e.g. .DS_Store, README.md) are excluded.
            Nested subdirectories under each class directory are scanned
            recursively; the class label is always the first directory under
            the split folder.
        """
        filepaths: list[UPath] = []
        for class_dir in dataset_split.iterdir():
            if class_dir.is_dir():
                filepaths.extend(p for p in class_dir.rglob("*") if p.is_file() and _is_image_path(p))
        return filepaths

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

        rel = image_path.relative_to(self._split_path)
        label = rel.parts[0]  # first dir under split — always the class, even for nested images

        one_hot_encode = np.zeros([len(self._label2index)])
        one_hot_encode[self._label2index[label]] = 1

        metadata: DatumMetadata = {"id": str(rel)}

        return img_chw, one_hot_encode, metadata

    def get_input(self, index: int, /) -> "NDArray[Any]":
        """Get only the image data at the given index.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.

        Returns
        -------
        NDArray[Any]
            The image data as a NumPy array in CHW format.

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

        with image_path.open("rb") as f, Image.open(f) as img:
            img = img.convert("RGB")
            arr = np.asarray(img)

        return np.moveaxis(arr, -1, 0)

    def get_target(self, index: int, /) -> "NDArray[Any]":
        """Get only the target label at the given index without loading the image.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.

        Returns
        -------
        NDArray[Any]
            The one-hot encoded label as a NumPy array.

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

        label = image_path.relative_to(self._split_path).parts[0]

        one_hot_encode = np.zeros([len(self._label2index)])
        one_hot_encode[self._label2index[label]] = 1

        return one_hot_encode

    def get_metadata(self, index: int, /) -> DatumMetadata:
        """Get only the metadata at the given index without loading the image.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.

        Returns
        -------
        DatumMetadata
            The metadata dictionary for the datum.

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

        rel = image_path.relative_to(self._split_path)
        return {"id": str(rel)}


class DatasetSpecification(TypedDict):
    """Dataset metadata required for loading datasets via checkmaite wrappers.

    Attributes
    ----------
    dataset_type : Literal["YoloClassificationDataset"]
        Dataset class as a string.
        TODO: hard-coded due to https://github.com/microsoft/pyright/issues/9194
        and maite pyright<=1.1.320
    data_dir : str
        Root directory of the YOLO classification dataset containing split
        folders (e.g., train/, val/, test/). This is passed as ``root_dir``
        to ``YoloClassificationDataset``; the split folder is appended
        internally.
    split_folder : Literal["train", "val", "test", "validation"]
        Name of the split folder to load inside the root dataset directory.
        Use "val" for the standard YOLO validation split. "validation" is
        accepted for backward compatibility and falls back to "val/" if a
        "validation/" directory does not exist.
    """

    # Dataset class as a string
    # TODO: hard-coded due to https://github.com/microsoft/pyright/issues/9194 and maite pyright<=1.1.320
    dataset_type: Literal["YoloClassificationDataset"]
    # Root directory of the YOLO classification dataset containing split folders
    # (e.g., train/, val/, test/). Passed as root_dir to YoloClassificationDataset.
    data_dir: str
    # Name of the split folder to load (e.g., "train", "val", "test").
    # "validation" is accepted for backward compatibility.
    split_folder: Literal["train", "val", "test", "validation"]


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


class YoloClassificationDataLoader:
    """MAITE-compliant DataLoader for YoloClassificationDataset.

    Yields batches of ``(inputs, targets, metadata)`` tuples where each
    element is a list of the corresponding per-sample values from the dataset.
    The loader is re-iterable: calling ``list(loader)`` twice produces the same
    result (with ``shuffle=False``) or an equivalently shuffled result when the
    same seed is supplied.

    Parameters
    ----------
    dataset : YoloClassificationDataset
        The dataset to iterate over.
    batch_size : int, optional
        Number of samples per batch.  Must be >= 1.  Defaults to 1.
    shuffle : bool, optional
        Whether to shuffle sample order at the start of each iteration.
        Defaults to ``False``.
    seed : int or None, optional
        Random seed used for shuffling.  When provided, repeated iterations
        produce the same order.  Defaults to ``None``.

    Examples
    --------
    >>> dataset = YoloClassificationDataset("path/to/root", split="val")
    >>> loader = YoloClassificationDataLoader(dataset, batch_size=4, shuffle=True, seed=0)
    >>> for inputs, targets, metadata in loader:
    ...     assert len(inputs) == len(targets) == len(metadata)
    """

    def __init__(
        self,
        dataset: YoloClassificationDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed

    def __iter__(self) -> Iterator[tuple[list, list, list]]:
        indices = list(range(len(self._dataset)))
        if self._shuffle:
            rng = random.Random(self._seed)  # noqa: S311  # nosec B311
            rng.shuffle(indices)
        for start in range(0, len(indices), self._batch_size):
            batch_indices = indices[start : start + self._batch_size]
            batch = [self._dataset[i] for i in batch_indices]
            inputs, targets, metadata = zip(*batch, strict=True)
            yield list(inputs), list(targets), list(metadata)

    def __len__(self) -> int:
        """Return the number of batches."""
        return (len(self._dataset) + self._batch_size - 1) // self._batch_size
