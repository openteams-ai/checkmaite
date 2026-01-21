import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch
import yaml
from maite.protocols import ArrayLike, DatasetMetadata
from maite.protocols.object_detection import Dataset, DatumMetadataType
from PIL import Image
from torchvision.ops.boxes import box_convert
from torchvision.transforms.functional import pil_to_tensor
from upath import UPath

from jatic_ri.core._utils import id_hash

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


class CocoDetectionDataset(Dataset):
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

    Notes
    -----
    The RI team follows the convention of using the `image` field to store all relevant metadata.
    If your dataset includes custom metadata fields, ensure they are placed inside the `image` field.

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


class YoloDetectionDataset(Dataset):
    """A dataset protocol for object detection ML subproblem providing datum-level data access.

    Indexing into or iterating over the an object detection dataset returns a `Tuple` of
    types `Tensor`, `DetectionTarget`, and `Dict[str, Any]`. These correspond to
    the model input type, model target type, and datum-level metadata, respectively.

    See https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format for
    specification.

    Parameters
    ----------
    yaml_dataset : str
        Full filepath to the yaml file containing dataset metadata.
    ann_dir : str
        Full path to the root directory containing the annotation folders.
    dataset_id : str, optional
        Identifier for the dataset, by default "yolo".

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

    """

    def __init__(self, yaml_dataset: str, ann_dir: str, dataset_id: str | None = None) -> None:
        """Initialize the YOLO detection dataset.

        Parameters
        ----------
        yaml_dataset : str
            Full filepath to the yaml file containing dataset metadata
        ann_dir : str
            Full path to the directory containing the annotation folders
        dataset_id : str, optional
            Optional identifier for dataset. If omitted,
                a unique one will be generated from the other input arguments.
        """
        self._yaml_path = UPath(yaml_dataset)
        with self._yaml_path.open("r") as fd:
            content = yaml.safe_load(fd)

        self._n_classes = len(content["names"])

        # Use UPath for remote filesystem support
        self._train_path = UPath(content["train"])
        self._images = sorted([p.name for p in self._train_path.iterdir() if p.is_file()])

        self._ann_path = UPath(ann_dir)
        annotation_files = sorted([p.name for p in self._ann_path.iterdir() if p.is_file()])

        # Cache all annotations
        self._annotations: list[tuple[list[list[float]], list[int]]] = []
        for ann_file in annotation_files:
            boxes: list[list[float]] = []
            labels: list[int] = []
            ann_file_path = self._ann_path / ann_file
            with ann_file_path.open("r") as fd:
                for line in fd:
                    label, x_center, y_center, width, height = line.split(" ")
                    labels.append(int(label))
                    boxes.append([float(x_center), float(y_center), float(width), float(height)])
            self._annotations.append((boxes, labels))

        # Generate dataset_id if not provided
        if dataset_id is None:
            dataset_id = f"yolo_{id_hash(yaml_dataset=yaml_dataset, ann_dir=ann_dir)}"

        self.metadata = DatasetMetadata(id=dataset_id, index2label=content["names"])

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
            image_name = self._images[index]
            boxes_raw, labels_raw = self._annotations[index]
        except IndexError as e:
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self)}",
            ) from e

        image_path = self._train_path / image_name
        with image_path.open("rb") as f, Image.open(f) as img:
            img.load()  # Force load before closing file handle
            img_pt = pil_to_tensor(img)

        # format ground truth
        num_boxes = len(boxes_raw)
        boxes = torch.zeros(num_boxes, 4)
        scores = torch.zeros(num_boxes)
        for i in range(num_boxes):
            boxes[i, :] = box_convert(
                torch.as_tensor(boxes_raw[i]),
                in_fmt="cxcywh",
                out_fmt="xyxy",
            )
            scores[i] = 1
        labels = torch.as_tensor(labels_raw)

        metadata: DatumMetadataType = {"id": index}
        return img_pt, DetectionTarget(boxes, labels, scores), metadata


class VisdroneDetectionDataset(Dataset):
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


class DatasetSpecification(TypedDict):
    """Dataset metadata required for loading datasets via the RI wrappers.

    Attributes
    ----------
    dataset_type : Literal["CocoDetectionDataset", "YoloDetectionDataset", "VisdroneDetectionDataset"]
        The type of the dataset.
        TODO: hard-coded due to https://github.com/microsoft/pyright/issues/9194 and maite pyright<=1.1.320
    metadata_path : str | UPath
        Full path to the metadata file. For Coco datasets, this is the annotation file.
        For yolo datasets, this is the yaml file. Supports local paths and cloud URLs.
    data_dir : str | UPath
        Full path to the directory containing:
        - the annotation files for yolo,
        - the split directory for coco, or
        - the root data directory for visdrone.
        Supports local paths and cloud URLs (s3://, gs://, az://).

    """

    # TO DO hard-coded due to https://github.com/microsoft/pyright/issues/9194 and maite pyright<=1.1.320
    dataset_type: Literal["CocoDetectionDataset", "YoloDetectionDataset", "VisdroneDetectionDataset"]
    # full path to the metadata file. For Coco datasets, this is the annotation file. For
    # yolo datasets, this is the yaml file. Supports cloud URLs.
    metadata_path: str | UPath
    # full path to the directory containing
    #   - the annotation files for yolo,
    #   - the split directory for coco, or
    #   - the root data directory for visdrone
    # Supports cloud URLs (s3://, gs://, az://)
    data_dir: str | UPath


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
            loaded[name] = YoloDetectionDataset(
                yaml_dataset=str(dataset_metadata["metadata_path"]), ann_dir=str(dataset_metadata["data_dir"])
            )
        elif dataset_metadata["dataset_type"] == "VisdroneDetectionDataset":
            loaded[name] = VisdroneDetectionDataset(root=str(dataset_metadata["data_dir"]))
        else:
            raise RuntimeError(f"Dataset type {dataset_metadata['dataset_type']} is not supported.")
    return loaded
