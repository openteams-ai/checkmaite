"""Load native datamaite object-detection datasets for checkmaite.

The factories here return :class:`datamaite.object_detection.ObjectDetectionDataset`
objects directly. Datamaite owns format parsing, source records, lazy image
decoding, MAITE target construction, and dataset metadata. CheckMAITE does not
wrap those objects or eagerly duplicate their annotations as Torch tensors.

What stays CheckMAITE-side is policy, not parsing: relative annotation paths keep
their historical caller-relative meaning, and recognized datamaite row-rejection
warnings are promoted to errors -- a best-effort guard, not a guarantee of
complete source integrity (see :mod:`checkmaite.core._common.dataset_utils`).
"""

import random
from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any, Literal, TypedDict

import maite.protocols.object_detection as od
import numpy as np
from datamaite import load_od
from datamaite.object_detection import ObjectDetectionDataset
from maite.protocols import DatasetMetadata
from modelmaite.object_detection import DetectionTarget
from torch import Tensor, as_tensor
from typing_extensions import NotRequired
from upath import UPath

from checkmaite.core._common.dataset_utils import (
    collect_source_warnings,
    datamaite_root,
    enforce_source_integrity,
    is_remote,
    to_caller_anchored_path,
)
from checkmaite.core._utils import id_hash

StorageOptions = Mapping[str, Any] | None


def _with_dataset_id(dataset: ObjectDetectionDataset, dataset_id: str) -> ObjectDetectionDataset:
    """Return the immutable datamaite dataset with its checkmaite-facing id.

    datamaite carries its runtime storage options (the fsspec credentials a
    remote root was opened with) in an ``InitVar``, which ``dataclasses.replace``
    resets to its default. Re-binding them keeps a cloud-backed dataset readable
    after the id swap.
    """
    storage_options = dataset._runtime_storage_options  # noqa: SLF001  # datamaite's documented runtime accessor
    return replace(dataset, dataset_id=dataset_id).with_storage_options(storage_options)


def load_coco_detection_dataset(
    root: str | UPath,
    ann_file: str | UPath,
    dataset_id: str | None = None,
    *,
    storage_options: StorageOptions = None,
    strict_annotations: bool = True,
) -> ObjectDetectionDataset:
    """Load COCO directly into datamaite's native MAITE OD dataset.

    ``root`` may be local or a remote URL (``s3://``, ``gs://``, ``az://``,
    ``memory://``); install the matching provider extra for cloud roots.
    ``root`` may also be a configured ``UPath``; it is passed through unchanged,
    so its filesystem options (credentials, endpoints) reach datamaite. A
    relative local ``ann_file`` is resolved against the working directory, as it
    always has been in CheckMAITE, even under a remote root.

    ``strict_annotations`` (default) promotes recognized datamaite row-rejection
    warnings to :class:`DatasetSourceError`. It is a best-effort guard and does
    not currently guarantee complete source integrity.
    """
    with collect_source_warnings() as source_warnings:
        dataset = load_od(
            datamaite_root(root),
            dataset_format="coco",
            annotation_file=to_caller_anchored_path(ann_file),
            images_dir=".",
            storage_options=storage_options,
        )
    enforce_source_integrity(source_warnings, source=f"COCO dataset {ann_file}", strict=strict_annotations)
    if not dataset.samples:
        raise ValueError(f"No COCO samples were loaded from {ann_file}.")
    if dataset_id is None:
        dataset_id = f"coco_{id_hash(root=str(root), ann_file=ann_file)}"
    return _with_dataset_id(dataset, dataset_id)


def load_yolo_detection_dataset(
    yaml_dataset: str | UPath,
    ann_dir: str | UPath | None = None,
    dataset_id: str | None = None,
    split: str = "train",
    *,
    storage_options: StorageOptions = None,
    strict_annotations: bool = True,
) -> ObjectDetectionDataset:
    """Load a YOLO dataset and return a native datamaite object.

    ``yaml_dataset`` may use any file name (not just ``data.yaml``); ``ann_dir``
    overrides the conventional ``labels/`` directory; ``split`` limits discovery
    to one split. All three are datamaite loader options — CheckMAITE stages no
    temporary dataset trees. A relative ``ann_dir`` is resolved against the
    working directory rather than the dataset root, preserving CheckMAITE's
    historical meaning (also under a remote root); misresolving it would load
    images with no detections at all rather than fail. A configured ``UPath``
    passed as ``yaml_dataset`` keeps its filesystem options.

    ``strict_annotations`` (default) promotes recognized datamaite row-rejection
    warnings to :class:`DatasetSourceError`. It is a best-effort guard and does
    not currently guarantee complete source integrity.
    """
    yaml_path = yaml_dataset if isinstance(yaml_dataset, UPath) else UPath(yaml_dataset)
    # A string root stays a string (datamaite applies ``storage_options`` to it);
    # a configured path object is passed through so its own options survive.
    yaml_root = str(yaml_path.parent) if isinstance(yaml_dataset, str) else yaml_path.parent
    if ann_dir is not None and is_remote(yaml_root) and not is_remote(ann_dir):
        # datamaite 0.5.0 records each label file relative to the dataset root and
        # cannot relate a local label path to a remote root. Refuse up front
        # rather than fail mid-load (or, before the file:// anchoring, silently
        # load every image with no labels).
        raise ValueError(
            f"A local ann_dir ({ann_dir}) under a remote YOLO root ({yaml_path}) is not supported by "
            "datamaite 0.5.0. Upload the labels next to the images, or point ann_dir at the same "
            "remote store."
        )
    with collect_source_warnings() as source_warnings:
        dataset = load_od(
            datamaite_root(yaml_root),
            dataset_format="yolo",
            yaml_file=yaml_path.name,
            ann_dir=to_caller_anchored_path(ann_dir) if ann_dir is not None else None,
            split=split,
            storage_options=storage_options,
        )
    enforce_source_integrity(
        source_warnings, source=f"YOLO dataset {yaml_path} (split {split!r})", strict=strict_annotations
    )
    if not dataset.samples:
        raise ValueError(f"No YOLO samples were loaded for split {split!r} from {yaml_path}.")

    if dataset_id is None:
        dataset_id = f"yolo_{id_hash(yaml_dataset=str(yaml_dataset), ann_dir=str(ann_dir), split=split)}"
    return _with_dataset_id(dataset, dataset_id)


def load_visdrone_detection_dataset(
    root: str | UPath,
    *,
    dataset_id: str | None = None,
    storage_options: StorageOptions = None,
    strict_annotations: bool = True,
) -> ObjectDetectionDataset:
    """Load VisDrone directly into datamaite's native MAITE OD dataset.

    ``root`` may be local, a remote URL, or a configured ``UPath`` (passed through
    unchanged so its filesystem options survive). ``strict_annotations``
    (default) promotes recognized datamaite row-rejection warnings to
    :class:`DatasetSourceError`; it is a best-effort guard and does not
    currently guarantee complete source integrity.
    """
    with collect_source_warnings() as source_warnings:
        dataset = load_od(datamaite_root(root), dataset_format="visdrone", storage_options=storage_options)
    enforce_source_integrity(source_warnings, source=f"VisDrone dataset {root}", strict=strict_annotations)
    if not dataset.samples:
        raise ValueError(f"No VisDrone samples were loaded from {root}.")
    if dataset_id is None:
        dataset_id = f"visdrone_{id_hash(root=str(root))}"
    return _with_dataset_id(dataset, dataset_id)


class DatasetSpecification(TypedDict):
    """Configuration for a native datamaite object-detection dataset."""

    dataset_format: Literal["coco", "yolo", "visdrone"]
    metadata_path: str | UPath
    data_dir: str | UPath
    split_folder: NotRequired[str]
    storage_options: NotRequired[Mapping[str, Any]]


def load_datasets(datasets: dict[str, DatasetSpecification]) -> dict[str, ObjectDetectionDataset]:
    """Load configured object-detection datasets as native datamaite objects."""
    loaded: dict[str, ObjectDetectionDataset] = {}
    for name, specification in datasets.items():
        dataset_format = specification["dataset_format"]
        storage_options = specification.get("storage_options")
        if dataset_format == "coco":
            loaded[name] = load_coco_detection_dataset(
                root=specification["data_dir"],
                ann_file=specification["metadata_path"],
                storage_options=storage_options,
            )
        elif dataset_format == "yolo":
            ann_dir = specification.get("data_dir")
            loaded[name] = load_yolo_detection_dataset(
                yaml_dataset=specification["metadata_path"],
                ann_dir=ann_dir if ann_dir else None,
                split=specification.get("split_folder", "train"),
                storage_options=storage_options,
            )
        elif dataset_format == "visdrone":
            loaded[name] = load_visdrone_detection_dataset(
                root=specification["data_dir"], storage_options=storage_options
            )
        else:
            raise RuntimeError(f"Dataset format {dataset_format} is not supported.")
    return loaded


class XaitkExplainableDetectionBaselineDataset(od.Dataset):
    """Replace ground-truth targets with a model's top predictions for XAITK."""

    def __init__(self, dataset: od.Dataset, model: od.Model, dets_limit: int = 10) -> None:
        metadata_args: dict = {"id": f"xai_temp_{dataset.metadata['id']}"}
        if "index2label" in dataset.metadata:
            metadata_args["index2label"] = dataset.metadata["index2label"]
        self.metadata = DatasetMetadata(**metadata_args)
        self.items = self._construct_dataset(dataset, model, dets_limit)

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget, od.DatumMetadataType]:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _construct_dataset(
        dataset: od.Dataset,
        model: od.Model,
        dets_limit: int,
    ) -> list[tuple[Tensor, DetectionTarget, od.DatumMetadataType]]:
        items: list[tuple[Tensor, DetectionTarget, od.DatumMetadataType]] = []
        for image, _target, metadata in dataset:
            predictions = model([image])[0]
            sorted_indices = np.argsort(np.asarray(predictions.scores))[::-1][:dets_limit].copy()
            target = DetectionTarget(
                boxes=np.asarray(predictions.boxes)[sorted_indices],
                labels=np.asarray(predictions.labels)[sorted_indices],
                scores=np.asarray(predictions.scores)[sorted_indices],
            )
            items.append((as_tensor(image), target, metadata))
        return items


class YoloDetectionDataLoader:
    """Small re-iterable batcher for any MAITE object-detection dataset.

    The historical name is retained because this is a batching utility, not an
    on-disk YOLO dataset implementation.
    """

    def __init__(
        self,
        dataset: od.Dataset,
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
