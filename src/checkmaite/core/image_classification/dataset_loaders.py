"""Load native datamaite image-classification datasets for checkmaite.

Unlike the former CheckMAITE dataset wrappers, the factories in this module
return :class:`datamaite.image_classification.ImageClassificationDataset`
instances directly. Datamaite datasets already satisfy MAITE structurally:
``dataset[index]`` lazily decodes an image and returns the MAITE
``(input, target, metadata)`` tuple.

Split selection, split-name aliasing (``validation`` -> ``val``), split-local
taxonomies including empty class directories, and recursive image discovery
below each class directory are all datamaite loader behaviour. The only policy
kept here is CheckMAITE's fail-loud one: datamaite warns and yields an empty
dataset where CheckMAITE raises.
"""

import random
from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any, Literal, TypedDict

import maite.protocols.image_classification as ic
from datamaite import load_ic
from datamaite.image_classification import ImageClassificationDataset
from upath import UPath

from checkmaite.core._common.dataset_utils import (
    collect_source_warnings,
    datamaite_root,
    enforce_source_integrity,
)
from checkmaite.core._utils import id_hash

StorageOptions = Mapping[str, Any] | None


class ClassificationDatasetWrapperError(Exception):
    """Base class retained for callers handling dataset-load errors."""


class MissingYoloDataSplitError(ClassificationDatasetWrapperError):
    """The provided YOLO dataset is missing the requested data split."""


def _available_splits(root: UPath) -> list[str]:
    """Top-level directory names under ``root``, for an actionable error."""
    try:
        return sorted(child.name for child in root.iterdir() if child.is_dir())
    except (OSError, ValueError):  # pragma: no cover - unreadable or missing root
        return []


def load_yolo_classification_dataset(
    root_dir: str | UPath,
    dataset_id: str | None = None,
    split: str = "test",
    *,
    storage_options: StorageOptions = None,
    strict_annotations: bool = True,
) -> ImageClassificationDataset:
    """Load one YOLO classification split as a native datamaite dataset.

    Images and targets remain in datamaite's representation. In particular,
    this function does not introduce another CheckMAITE dataset object and does
    not convert MAITE-compatible NumPy arrays to Torch tensors.

    ``root_dir`` is the dataset root (the directory holding the split folders)
    and may be local or a remote URL; ``split`` is passed straight to datamaite,
    which owns the alias handling and builds a split-local taxonomy. Images
    nested below a class directory (``<split>/<class>/**/<image>``) are
    discovered recursively, with the top-level directory as the class and the
    nested relative path preserved in the datum id. A configured ``UPath`` keeps
    its filesystem options.

    ``strict_annotations`` (default) promotes recognized datamaite row-rejection
    warnings to :class:`DatasetSourceError`. It is a best-effort guard and does
    not currently guarantee complete source integrity.
    """
    root = root_dir if isinstance(root_dir, UPath) else UPath(root_dir)
    with collect_source_warnings() as source_warnings:
        dataset = load_ic(
            # Pass path objects through: stringifying a configured UPath would
            # drop its filesystem options (credentials) before datamaite sees it.
            datamaite_root(root_dir),
            dataset_format="yolo",
            split=split,
            # CheckMAITE's API is explicitly root-plus-split, so the layout is
            # never inferred: a class directory that happens to be named like a
            # split cannot flip the interpretation.
            layout="split",
            storage_options=storage_options,
        )
    enforce_source_integrity(
        source_warnings, source=f"YOLO classification dataset {root} (split {split!r})", strict=strict_annotations
    )
    if not dataset.samples:
        raise MissingYoloDataSplitError(
            f"No images were loaded for data split {split!r} under {root} — the split subdirectory is "
            f"missing or empty. Split subdirectories present: {_available_splits(root) or 'none'}."
        )

    if dataset_id is None:
        dataset_id = f"yolo_classification_{id_hash(root_dir=root_dir, split=split)}"
    storage = dataset._runtime_storage_options  # noqa: SLF001  # datamaite's documented runtime accessor
    return replace(dataset, dataset_id=dataset_id).with_storage_options(storage)


class DatasetSpecification(TypedDict):
    """Configuration for a native datamaite image-classification dataset."""

    dataset_format: Literal["yolo"]
    data_dir: str
    split_folder: str


def load_datasets(datasets: dict[str, DatasetSpecification]) -> dict[str, ImageClassificationDataset]:
    """Load configured datasets, returning datamaite objects directly."""
    loaded: dict[str, ImageClassificationDataset] = {}
    for name, specification in datasets.items():
        if specification["dataset_format"] != "yolo":
            raise RuntimeError(f"Dataset format {specification['dataset_format']} is not supported.")
        loaded[name] = load_yolo_classification_dataset(
            root_dir=specification["data_dir"],
            split=specification["split_folder"],
        )
    return loaded


class YoloClassificationDataLoader:
    """Small re-iterable batcher for any MAITE image-classification dataset.

    The historical name is retained because this is a batching utility, not an
    on-disk dataset implementation. Its input is the MAITE protocol rather than
    the removed ``YoloClassificationDataset`` wrapper.
    """

    def __init__(
        self,
        dataset: ic.Dataset,
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
            batch = [self._dataset[index] for index in indices[start : start + self._batch_size]]
            inputs, targets, metadata = zip(*batch, strict=True)
            yield list(inputs), list(targets), list(metadata)

    def __len__(self) -> int:
        """Return the number of batches."""
        return (len(self._dataset) + self._batch_size - 1) // self._batch_size
