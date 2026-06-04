import csv
import json
import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch
import yaml
from maite.protocols import ArrayLike, DatasetMetadata
from maite.protocols.object_detection import DatumMetadataType, FieldwiseDataset
from PIL import Image
from torchvision.ops.boxes import box_convert
from torchvision.transforms.functional import pil_to_tensor
from typing_extensions import NotRequired
from upath import UPath

from checkmaite.core._common.dataset_utils import _is_image_path
from checkmaite.core._utils import id_hash

SUPPORTED_DATASET_TYPES = ["CocoDetectionDataset"]


@dataclass
class DetectionTarget:
    """Detection Target.

    Parameters
    ----------
    boxes : ArrayLike
        Coordinates of bounding boxes where objects are detected, in xyxy format,
        shape: (n_boxes, 4).
    labels : ArrayLike
        Labels of detected images, shape: (n_boxes,).
    scores : ArrayLike
        How confident the model is in each detection, shape: (n_boxes, n_classes).

    """

    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike


class CocoDetectionDataset(FieldwiseDataset):
    """A dataset protocol for object detection ML subproblem providing datum-level data access.

    Indexing into or iterating over the an object detection dataset returns a `Tuple` of
    types `Tensor`, `DetectionTarget`, and `Dict[str, Any]`. These correspond to
    the model input type, model target type, and datum-level metadata, respectively.

    Parameters
    ----------
    root : str or UPath
        Root directory of the dataset.
    ann_file : str
        Full filepath to the annotation file for the dataset.
    dataset_id : str, optional
        Identifier for the dataset, by default "coco".

    Attributes
    ----------
    metadata : DatasetMetadata
        Metadata about the dataset, including id and label mapping.
    classes
        Mapping from ids to labels.

    Methods
    -------
    __getitem__(index)
        Provide mapping-style access to dataset elements.
    __len__()
        Return the number of data elements in the dataset.
    get_input(index)
        Get only the image tensor at the given index.
    get_target(index)
        Get only the detection target at the given index.
    get_metadata(index)
        Get only the metadata at the given index.

    Notes
    -----
    COCO `categories` are exposed as dataset-level `index2label` metadata.
    Each COCO `images` entry is returned as datum-level metadata, including any extra user-defined fields.
    Extra fields in `annotations`, `info`, `licenses`, or `categories` are not otherwise surfaced.

    """

    def __init__(self, root: str | UPath, ann_file: str, dataset_id: str | None = None) -> None:
        """Initialize the COCO detection dataset.

        Parameters
        ----------
        root : str or UPath
            Root directory of the dataset containing the images
        ann_file : str
            Full filepath to the annotation file for the dataset
        dataset_id : str, optional
            Optional identifier for dataset. If omitted,
                a unique one will be generated from the other input arguments.
        """
        self._root = UPath(root)

        ann_path = UPath(ann_file)
        with ann_path.open("r") as fd:
            content = json.load(fd)

        self._index2label = {x["id"]: x["name"] for x in content["categories"]}
        self._n_classes = len(self._index2label)
        self._images = content["images"]

        # Build annotation index by image_id
        self._img_id_to_annotations: dict[int, list[dict]] = defaultdict(list)
        for ann in content["annotations"]:
            self._img_id_to_annotations[ann["image_id"]].append(ann)

        # Generate dataset_id if not provided
        if dataset_id is None:
            dataset_id = f"coco_{id_hash(root=str(root), ann_file=ann_file)}"

        self.metadata = DatasetMetadata(id=dataset_id, index2label=self._index2label)

    def __len__(self) -> int:
        """Return length of dataset.

        Returns
        -------
        int
            The number of data elements in the dataset.

        """
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, DetectionTarget, DatumMetadataType]:
        """Get `index`-th element from dataset.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.

        Returns
        -------
        tuple[torch.Tensor, DetectionTarget, DatumMetadataType]
            A tuple containing the image tensor, detection target, and metadata.

        Raises
        ------
        IndexError
            If the index is out of range.

        """
        try:
            metadata = self._images[index]
        except IndexError as e:
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self._images)}",
            ) from e

        img_path = self._root / metadata["file_name"]
        with img_path.open("rb") as f, Image.open(f) as img:
            img.load()  # Force load before closing file handle
            img_pt = pil_to_tensor(img)

        # Get annotations for this image
        annotations = self._img_id_to_annotations[metadata["id"]]

        # format ground truth
        num_boxes = len(annotations)
        boxes = torch.zeros(num_boxes, 4)
        scores = torch.zeros(num_boxes)
        labels = []
        for i in range(num_boxes):
            ann = annotations[i]
            bbox = torch.as_tensor(ann["bbox"])
            boxes[i, :] = box_convert(bbox, in_fmt="xywh", out_fmt="xyxy")
            scores[i] = 1
            labels.append(ann["category_id"])

        return img_pt, DetectionTarget(boxes, torch.as_tensor(labels), scores), metadata

    def get_input(self, index: int, /) -> torch.Tensor:
        """Get only the image tensor at the given index.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.

        Returns
        -------
        torch.Tensor
            The image tensor.

        Raises
        ------
        IndexError
            If the index is out of range.

        """
        try:
            metadata = self._images[index]
        except IndexError as e:
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self._images)}",
            ) from e

        img_path = self._root / metadata["file_name"]
        with img_path.open("rb") as f, Image.open(f) as img:
            img.load()
            return pil_to_tensor(img)

    def get_target(self, index: int, /) -> DetectionTarget:
        """Get only the detection target at the given index without loading the image.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.

        Returns
        -------
        DetectionTarget
            The detection target containing boxes, labels, and scores.

        Raises
        ------
        IndexError
            If the index is out of range.

        """
        try:
            metadata = self._images[index]
        except IndexError as e:
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self._images)}",
            ) from e

        annotations = self._img_id_to_annotations[metadata["id"]]

        num_boxes = len(annotations)
        boxes = torch.zeros(num_boxes, 4)
        scores = torch.zeros(num_boxes)
        labels = []
        for i in range(num_boxes):
            ann = annotations[i]
            bbox = torch.as_tensor(ann["bbox"])
            boxes[i, :] = box_convert(bbox, in_fmt="xywh", out_fmt="xyxy")
            scores[i] = 1
            labels.append(ann["category_id"])

        return DetectionTarget(boxes, torch.as_tensor(labels), scores)

    def get_metadata(self, index: int, /) -> DatumMetadataType:
        """Get only the metadata at the given index without loading image or annotations.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.

        Returns
        -------
        DatumMetadataType
            The metadata dictionary for the datum.

        Raises
        ------
        IndexError
            If the index is out of range.

        """
        try:
            return self._images[index]
        except IndexError as e:
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self._images)}",
            ) from e


def _parse_yolo_label_file(label_path: "UPath | None", image_width: int, image_height: int) -> DetectionTarget:
    """Parse a YOLO annotation file and return pixel-space xyxy boxes.

    Parameters
    ----------
    label_path
        Path to the ``.txt`` annotation file, or ``None`` when no file exists.
    image_width
        Width of the corresponding image in pixels.
    image_height
        Height of the corresponding image in pixels.

    Returns
    -------
    DetectionTarget
        Bounding boxes in pixel-space xyxy format, integer class labels, and
        float scores (1.0 for every ground-truth box).  Returns empty tensors
        when *label_path* is ``None`` or the file does not exist.

    Raises
    ------
    ValueError
        If a non-blank, non-comment row does not have exactly 5 whitespace-
        separated fields, or if numeric conversion of any field fails.  The
        error message always includes the file path and 1-based line number.
    """
    empty = DetectionTarget(
        boxes=torch.empty((0, 4), dtype=torch.float32),
        labels=torch.empty((0,), dtype=torch.int64),
        scores=torch.empty((0,), dtype=torch.float32),
    )

    if label_path is None:
        return empty

    try:
        with label_path.open("r") as f:
            raw_lines = f.readlines()
    except FileNotFoundError:
        return empty

    boxes: list[list[float]] = []
    labels: list[int] = []

    for line_num, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped or stripped[0] == "#":
            continue

        fields = stripped.split()
        if len(fields) != 5:
            raise ValueError(
                f"Malformed row in '{label_path}' at line {line_num}: {stripped!r} "
                f"(expected 5 fields, got {len(fields)})"
            )

        try:
            cls = int(fields[0])
            cx, cy, w, h = float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4])
        except ValueError as exc:
            raise ValueError(
                f"Could not parse numeric values in '{label_path}' at line {line_num}: {stripped!r}"
            ) from exc

        # Convert normalized YOLO coordinates to pixel-space cxcywh before box_convert
        boxes.append([cx * image_width, cy * image_height, w * image_width, h * image_height])
        labels.append(cls)

    if not boxes:
        return empty

    return DetectionTarget(
        boxes=box_convert(torch.tensor(boxes, dtype=torch.float32), in_fmt="cxcywh", out_fmt="xyxy"),
        labels=torch.tensor(labels, dtype=torch.int64),
        scores=torch.ones(len(labels), dtype=torch.float32),
    )


class YoloDetectionDataset(FieldwiseDataset):
    """A dataset handler for YOLO object detection datasets.

    Loads image/label pairs from a YOLO-format dataset directory tree described
    by a YAML configuration file.  See
    https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format for
    the specification.

    Expected YAML layout::

        path: ./dataset-root   # optional; resolved relative to the YAML file
        train: images/train
        val: images/val
        test: images/test
        names:
            0: cat
            1: dog

    Expected directory layout::

        dataset-root/
            images/train/
                img1.jpg
            labels/train/
                img1.txt    # YOLO format: class cx cy w h  (normalized 0-1)

    Parameters
    ----------
    yaml_dataset : str or UPath
        Path to the YAML file that describes the dataset.
    ann_dir : str, UPath, or None, optional
        Explicit path to the directory containing ``.txt`` annotation files.
        When provided, it overrides the auto-inferred labels directory.
        Pass this to keep backward-compatible callers working or to use a
        non-standard labels location.
    dataset_id : str, optional
        Human-readable identifier for this dataset instance.  Auto-generated
        when omitted.
    split : {"train", "val", "test"}, optional
        Dataset split to load.  The value is used as a key in the YAML file to
        locate the image directory.  Defaults to ``"train"``.

    Attributes
    ----------
    metadata : DatasetMetadata
        Dataset-level metadata including ``id`` and ``index2label`` mapping.

    Methods
    -------
    __getitem__(index)
        Return ``(image_tensor, DetectionTarget, datum_metadata)`` for one sample.
    __len__()
        Return the number of image samples in the dataset.
    get_input(index)
        Return only the CHW RGB image tensor.
    get_target(index)
        Return only the DetectionTarget (opens image for size; no full decode).
    get_metadata(index)
        Return only the datum metadata (no I/O).

    Usage example
    -------------
    >>> dataset = YoloDetectionDataset("data.yaml", split="val")
    >>> loader = YoloDetectionDataLoader(dataset, batch_size=4)
    >>> for inputs, targets, metadata in loader:
    ...     pass
    """

    def __init__(
        self,
        yaml_dataset: str | UPath,
        ann_dir: str | UPath | None = None,
        dataset_id: str | None = None,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        """Initialize the YOLO detection dataset.

        Parameters
        ----------
        yaml_dataset : str or UPath
            Full path to the YAML file containing dataset metadata.
        ann_dir : str, UPath, or None, optional
            Explicit path to the annotation (``.txt``) directory.  When
            ``None``, the labels directory is inferred from the image directory
            using the standard ``images/ → labels/`` YOLO layout.
        dataset_id : str, optional
            Optional identifier for this dataset instance.  Auto-generated
            when omitted.
        split : {"train", "val", "test"}, optional
            Dataset split to load.  Defaults to ``"train"``.

        Raises
        ------
        ValueError
            If the requested split key is absent from the YAML file.
        """
        self._yaml_path = UPath(yaml_dataset)
        with self._yaml_path.open("r") as fd:
            content = yaml.safe_load(fd)

        self._n_classes = len(content["names"])
        yaml_parent = self._yaml_path.parent

        # Resolve the optional 'path:' field to find the dataset root directory
        if "path" in content and content["path"]:
            path_field = str(content["path"])
            raw = UPath(path_field)
            dataset_root = raw if raw.is_absolute() or "://" in path_field else yaml_parent / path_field
        else:
            dataset_root = yaml_parent

        # Locate the image directory for the requested split
        if split not in content or not content[split]:
            available = [k for k in content if k not in ("names", "path", "nc")]
            raise ValueError(f"Split '{split}' not found in '{self._yaml_path}'. " f"Available splits: {available}")
        split_value = str(content[split])
        raw_split = UPath(split_value)
        if raw_split.is_absolute() or "://" in split_value:
            self._image_dir = raw_split
        else:
            self._image_dir = dataset_root / split_value

        # Resolve label directory: explicit override or infer from image path
        if ann_dir is not None:
            self._label_dir: UPath | None = UPath(ann_dir)
        else:
            self._label_dir = self._infer_label_dir(self._image_dir)

        # Build sample list: (image_path, label_path | None) paired by file stem
        images = sorted(p for p in self._image_dir.iterdir() if p.is_file() and _is_image_path(p))
        self._samples: list[tuple[UPath, UPath | None]] = [
            (img, (self._label_dir / f"{img.stem}.txt") if self._label_dir is not None else None) for img in images
        ]

        if dataset_id is None:
            dataset_id = f"yolo_{id_hash(yaml_dataset=str(yaml_dataset), ann_dir=str(ann_dir), split=split)}"

        self.metadata = DatasetMetadata(id=dataset_id, index2label=content["names"])

    @staticmethod
    def _infer_label_dir(image_dir: UPath) -> UPath | None:
        """Infer labels directory from image directory using YOLO convention.

        Replaces the ``images`` component of the path with ``labels``.
        For example, ``.../images/train`` → ``.../labels/train``.
        Returns ``None`` when the image directory does not follow this layout.
        """
        if image_dir.parent.name == "images":
            return image_dir.parent.parent / "labels" / image_dir.name
        return None

    def __len__(self) -> int:
        """Return the number of image samples in the dataset."""
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, DetectionTarget, DatumMetadataType]:
        """Return the image tensor, detection target, and metadata for one sample.

        Parameters
        ----------
        index : int
            Zero-based sample index.

        Returns
        -------
        tuple[torch.Tensor, DetectionTarget, DatumMetadataType]
            CHW uint8 RGB tensor, pixel-space xyxy DetectionTarget, metadata dict.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        try:
            img_path, label_path = self._samples[index]
        except IndexError as e:
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self)}",
            ) from e

        with img_path.open("rb") as f, Image.open(f) as img:
            img = img.convert("RGB")
            width, height = img.size
            img_pt = pil_to_tensor(img)

        target = _parse_yolo_label_file(label_path, width, height)
        metadata: DatumMetadataType = {"id": img_path.name}
        return img_pt, target, metadata

    def get_input(self, index: int, /) -> torch.Tensor:
        """Return only the image tensor at the given index.

        Parameters
        ----------
        index : int
            Zero-based sample index.

        Returns
        -------
        torch.Tensor
            CHW uint8 RGB image tensor.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        try:
            img_path, _ = self._samples[index]
        except IndexError as e:
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self)}",
            ) from e

        with img_path.open("rb") as f, Image.open(f) as img:
            img = img.convert("RGB")
            return pil_to_tensor(img)

    def get_target(self, index: int, /) -> DetectionTarget:
        """Return only the detection target at the given index.

        Opens the image file only to read its pixel dimensions; the full image
        data is not decoded.

        Parameters
        ----------
        index : int
            Zero-based sample index.

        Returns
        -------
        DetectionTarget
            Pixel-space xyxy boxes, integer class labels, float scores.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        try:
            img_path, label_path = self._samples[index]
        except IndexError as e:
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self)}",
            ) from e

        with img_path.open("rb") as f, Image.open(f) as img:
            width, height = img.size  # header-only read; no full pixel decode

        return _parse_yolo_label_file(label_path, width, height)

    def get_metadata(self, index: int, /) -> DatumMetadataType:
        """Return only the datum metadata at the given index (no I/O).

        Parameters
        ----------
        index : int
            Zero-based sample index.

        Returns
        -------
        DatumMetadataType
            Dict with at least an ``"id"`` key (the image filename).

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        if index < 0 or index >= len(self):
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self)}",
            )
        img_path, _ = self._samples[index]
        return {"id": img_path.name}


class VisdroneDetectionDataset(FieldwiseDataset):
    """A dataset protocol for object detection ML subproblem providing datum-level data access.

    Indexing into or iterating over the object detection dataset returns a `Tuple` of
    types `Tensor`, `DetectionTarget`, and `Dict[str, Any]`. These correspond to
    the model input type, model target type, and datum-level metadata, respectively.

    Parameters
    ----------
    root : str or UPath
        Root directory of the dataset.
    dataset_id : str, optional
        Identifier for the dataset, by default "visdrone".

    Attributes
    ----------
    root : UPath
        Resolved path to the root directory of the dataset.
    metadata : dict
        Metadata about the dataset, including id and label mapping.
    classes
        Mapping from ids to labels.

    Methods
    -------
    __getitem__(index)
        Provide mapping-style access to dataset elements.
    __len__()
        Return the number of data elements in the dataset.
    get_input(index)
        Get only the image tensor at the given index.
    get_target(index)
        Get only the detection target at the given index.
    get_metadata(index)
        Get only the metadata at the given index.

    """

    def __init__(self, root: str | UPath, *, dataset_id: str | None = None) -> None:
        """Initialize the VisDrone detection dataset.

        Parameters
        ----------
        root : str or UPath
            Root directory of the dataset containing images and annotations folders
        dataset_id : str, optional
            Optional identifier for dataset. If omitted,
                a unique one will be generated from the other input arguments.
        """
        self.root = UPath(root)
        # Generate dataset_id if not provided
        if dataset_id is None:
            dataset_id = f"visdrone_{id_hash(root=str(self.root))}"

        self.metadata = {
            "id": dataset_id,
            # See https://github.com/VisDrone/VisDrone2018-DET-toolkit
            "index2label": {
                0: "ignored regions",
                1: "pedestrian",
                2: "people",
                3: "bicycle",
                4: "car",
                5: "van",
                6: "truck",
                7: "tricycle",
                8: "awning-tricycle",
                9: "bus",
                10: "motor",
                11: "others",
            },
        }

        self._samples = self._load_samples(self.root)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, DetectionTarget, DatumMetadataType]:
        """Get `index`-th element from dataset.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.

        Returns
        -------
        tuple[torch.Tensor, DetectionTarget, DatumMetadataType]
            A tuple containing the image tensor, detection target, and metadata.
            Note: metadata includes additional fields beyond DatumMetadataType.

        """
        image_path, target, metadata = self._samples[index]
        with image_path.open("rb") as f, Image.open(f) as img:
            img.load()  # Force load before closing file handle
            image = pil_to_tensor(img)
        return image, target, metadata  # pyright: ignore[reportReturnType]

    def __len__(self) -> int:
        """Return length of dataset.

        Returns
        -------
        int
            The number of data elements in the dataset.

        """
        return len(self._samples)

    def _load_samples(self, root: UPath) -> list[tuple[UPath, DetectionTarget, dict]]:
        """Load samples from the VisDrone dataset structure.

        Parameters
        ----------
        root : UPath
            The root directory of the VisDrone dataset.

        Returns
        -------
        list[tuple[UPath, DetectionTarget, dict]]
            A list of samples, where each sample is a tuple containing the image path,
            detection target, and metadata.

        """
        images_folder = root / "images"
        annotations_folder = root / "annotations"

        samples: list[tuple[UPath, DetectionTarget, dict]] = []

        # Get all annotation files
        annotation_files = sorted(
            [p for p in annotations_folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
        )

        for annotation_path in annotation_files:
            boxes: list[tuple[float, float, float, float]] = []
            scores: list[float] = []
            labels: list[int] = []
            truncations: list[int] = []
            occlusions: list[int] = []

            with annotation_path.open("r") as f:
                r = csv.reader(f, delimiter=",")
                # See https://github.com/VisDrone/VisDrone2018-DET-toolkit
                # Discards any data after 8th value of a row (e.g. due to trailing comma).  See:
                # https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/merge_requests/287
                for x, y, h, w, score, label, truncation, occlusion, *_ in r:
                    boxes.append((float(x), float(y), float(h), float(w)))
                    scores.append(float(score))
                    labels.append(int(label))
                    truncations.append(int(truncation))
                    occlusions.append(int(occlusion))

            # Construct image path from annotation filename
            ann_stem = annotation_path.name.rsplit(".", 1)[0]
            image_path = images_folder / f"{ann_stem}.jpg"
            target = DetectionTarget(
                boxes=box_convert(torch.tensor(boxes), in_fmt="xywh", out_fmt="xyxy"),
                scores=torch.tensor(scores),
                labels=torch.tensor(labels),
            )
            metadata = {
                "id": ann_stem,
                "image_path": str(image_path),
                "annotation_path": str(annotation_path),
                "truncations": truncations,
                "occlusions": occlusions,
            }

            samples.append((image_path, target, metadata))

        return samples

    def get_input(self, index: int, /) -> torch.Tensor:
        """Get only the image tensor at the given index.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.

        Returns
        -------
        torch.Tensor
            The image tensor.

        Raises
        ------
        IndexError
            If the index is out of range.

        """
        image_path, _, _ = self._samples[index]
        with image_path.open("rb") as f, Image.open(f) as img:
            img.load()
            return pil_to_tensor(img)

    def get_target(self, index: int, /) -> DetectionTarget:
        """Get only the detection target at the given index without loading the image.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.

        Returns
        -------
        DetectionTarget
            The detection target containing boxes, labels, and scores.

        Raises
        ------
        IndexError
            If the index is out of range.

        """
        _, target, _ = self._samples[index]
        return target

    def get_metadata(self, index: int, /) -> DatumMetadataType:
        """Get only the metadata at the given index without loading the image.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.

        Returns
        -------
        DatumMetadataType
            The metadata dictionary for the datum.

        Raises
        ------
        IndexError
            If the index is out of range.

        """
        _, _, metadata = self._samples[index]
        return metadata  # pyright: ignore[reportReturnType]


class DatasetSpecification(TypedDict):
    """Dataset metadata required for loading datasets via checkmaite wrappers.

    Attributes
    ----------
    dataset_type : Literal["CocoDetectionDataset", "YoloDetectionDataset", "VisdroneDetectionDataset"]
        The type of the dataset.
        TODO: hard-coded due to https://github.com/microsoft/pyright/issues/9194 and maite pyright<=1.1.320
    metadata_path : str | UPath
        Full path to the metadata file.  For COCO this is the annotation JSON;
        for YOLO this is the dataset YAML.  Supports local paths and cloud URLs.
    data_dir : str | UPath
        Full path to the relevant data directory:

        - **YOLO**: explicit labels/annotation directory (passed as ``ann_dir``
          to ``YoloDetectionDataset``).  Omit or set to an empty string to let
          the loader infer the labels directory automatically.
        - **COCO**: the split directory containing images.
        - **VisDrone**: the root dataset directory.

        Supports local paths and cloud URLs (``s3://``, ``gs://``, ``az://``).
    split_folder : {"train", "val", "test"}, optional
        Split to load for YOLO detection datasets.  Defaults to ``"train"``
        when absent.  Has no effect on COCO or VisDrone datasets.

    """

    # TODO: hard-coded due to https://github.com/microsoft/pyright/issues/9194 and maite pyright<=1.1.320
    dataset_type: Literal["CocoDetectionDataset", "YoloDetectionDataset", "VisdroneDetectionDataset"]
    # Full path to the metadata file (COCO annotation JSON or YOLO YAML).
    metadata_path: str | UPath
    # Full path to the labels dir (YOLO), split dir (COCO), or root dir (VisDrone).
    data_dir: str | UPath
    # Optional split for YOLO detection; defaults to "train" when absent.
    split_folder: NotRequired[Literal["train", "val", "test"]]


def load_datasets(
    datasets: dict[str, DatasetSpecification],
) -> dict[str, CocoDetectionDataset | YoloDetectionDataset | VisdroneDetectionDataset]:
    """Simplified programmatic loading of datasets from a dictionary of DatasetSpecifications.

    Parameters
    ----------
    datasets : dict[str, DatasetSpecification]
        A dictionary where keys are dataset names and values are DatasetSpecification
        objects.

    Returns
    -------
    dict[str, CocoDetectionDataset | YoloDetectionDataset | VisdroneDetectionDataset]
        A dictionary of loaded datasets, where keys are dataset names and values are
        the corresponding dataset objects.

    Raises
    ------
    RuntimeError
        If an unsupported dataset type is encountered.

    """
    loaded: dict[str, CocoDetectionDataset | YoloDetectionDataset | VisdroneDetectionDataset] = {}
    for name, dataset_metadata in datasets.items():
        if dataset_metadata["dataset_type"] == "CocoDetectionDataset":
            loaded[name] = CocoDetectionDataset(
                root=str(dataset_metadata["data_dir"]), ann_file=str(dataset_metadata["metadata_path"])
            )
        elif dataset_metadata["dataset_type"] == "YoloDetectionDataset":
            ann_dir = dataset_metadata.get("data_dir")
            loaded[name] = YoloDetectionDataset(
                yaml_dataset=str(dataset_metadata["metadata_path"]),
                ann_dir=None if not ann_dir else str(ann_dir),
                split=dataset_metadata.get("split_folder", "train"),
            )
        elif dataset_metadata["dataset_type"] == "VisdroneDetectionDataset":
            loaded[name] = VisdroneDetectionDataset(root=str(dataset_metadata["data_dir"]))
        else:
            raise RuntimeError(f"Dataset type {dataset_metadata['dataset_type']} is not supported.")
    return loaded


class YoloDetectionDataLoader:
    """MAITE-compliant DataLoader for YoloDetectionDataset.

    Yields batches of ``(inputs, targets, metadata)`` tuples where each
    element is a list of the corresponding per-sample values from the dataset.
    The loader is re-iterable: calling ``list(loader)`` twice produces the same
    result (with ``shuffle=False``) or an equivalently shuffled result when the
    same seed is supplied.

    Parameters
    ----------
    dataset : YoloDetectionDataset
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
    >>> dataset = YoloDetectionDataset("data.yaml", split="val")
    >>> loader = YoloDetectionDataLoader(dataset, batch_size=4, shuffle=True, seed=0)
    >>> for inputs, targets, metadata in loader:
    ...     assert len(inputs) == len(targets) == len(metadata)
    """

    def __init__(
        self,
        dataset: YoloDetectionDataset,
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
