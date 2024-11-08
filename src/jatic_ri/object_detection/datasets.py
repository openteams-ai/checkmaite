"""datasets"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import yaml
from maite.protocols import ArrayLike
from maite.protocols.object_detection import Dataset
from PIL import Image
from torchvision.datasets import CocoDetection
from torchvision.ops.boxes import box_convert
from torchvision.transforms.functional import pil_to_tensor

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class DetectionTarget:
    """
    Detection Target.

    Attributes
    ----------
    boxes
        Coordinates of bounding boxes where objects are detected, in xyxy format,
        shape: (n_boxes, 4)
    labels
        Labels of detected images, shape: (n_boxes,)
    scores
        How confident the model is in each detection, shape: (n_boxes, n_classes)
    """

    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike


class CocoDetectionDataset(Dataset):
    """
    A dataset protocol for object detection ML subproblem providing datum-level
    data access.

    Indexing into or iterating over the an object detection dataset returns a `Tuple` of
    types `Tensor`, `DetectionTarget`, and `Dict[str, Any]`. These correspond to
    the model input type, model target type, and datum-level metadata, respectively.

    Attributes
    ----------
    classes
        Mapping from ids to labels.

    Methods
    -------
    __getitem__(self, index: int) -> Tuple[Tensor, DetectionTarget, Dict[str, Any]]
        Provide mapping-style access to dataset elements. Returned tuple elements
        correspond to input type, target type, and datum-specific metadata,
        respectively.

    __len__() -> int
        Return the number of data elements in the dataset.
    """

    def __init__(self, root: str | Path, ann_file: str) -> None:
        self.dataset: CocoDetection = CocoDetection(
            root,
            annFile=ann_file,
            transforms=lambda image, target: (pil_to_tensor(image), target),
        )
        with open(ann_file) as fd:
            content = json.load(fd)
        self._int_to_id = {i: x["id"] for i, x in enumerate(content["categories"])}
        self._id_to_int = {val: key for key, val in self._int_to_id.items()}
        self._classes = {i: x["name"] for i, x in enumerate(content["categories"])}
        self._n_classes = len(self._classes)
        self._images = content["images"]

    @property
    def classes(self) -> dict[int, str]:
        """Map ids to labels."""
        return self._classes

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, DetectionTarget, dict[str, Any]]:
        """Get `index`-th element from dataset."""
        try:
            # get original data item
            img_pt, annotations = self.dataset[index]
        except IndexError as e:
            # Here the underlying dataset is raising an IndexError since the index is beyond the
            # container's bounds. When wrapping custom datasets, wrappers are responsible to for
            # raising an IndexError in `__getitem__` when an index exceeds the container's bounds;
            # this enables iteration on the wrapper to properly terminate.
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self.dataset)}",
            ) from e

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
            labels.append(self._id_to_int[ann["category_id"]])

        # format metadata
        metadata = self._images[[image["id"] for image in self._images].index(self.dataset.ids[index])]

        return img_pt, DetectionTarget(boxes, torch.as_tensor(labels), scores), metadata


class YoloDetectionDataset(Dataset):
    """
    A dataset protocol for object detection ML subproblem providing datum-level
    data access.

    Indexing into or iterating over the an object detection dataset returns a `Tuple` of
    types `Tensor`, `DetectionTarget`, and `Dict[str, Any]`. These correspond to
    the model input type, model target type, and datum-level metadata, respectively.

    See https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format for
    specification.

    Attributes
    ----------
    classes
        Mapping from ids to labels.

    Methods
    -------
    __getitem__(self, index: int) -> Tuple[Tensor, DetectionTarget, Dict[str, Any]]
        Provide mapping-style access to dataset elements. Returned tuple elements
        correspond to input type, target type, and datum-specific metadata,
        respectively.

    __len__() -> int
        Return the number of data elements in the dataset.
    """

    def __init__(self, yaml_dataset: str, ann_dir: str) -> None:
        with open(yaml_dataset) as fd:
            content = yaml.safe_load(fd)

        self._int_to_id = dict(enumerate(content["names"]))
        self._id_to_int = {val: key for key, val in self._int_to_id.items()}
        self._classes = dict(enumerate(content["names"].values()))
        self._n_classes = len(self._classes)

        self._train = content["train"]
        self._images = sorted(os.listdir(content["train"]))
        self._ann_dir = ann_dir
        self._annotations = sorted(os.listdir(ann_dir))

    @property
    def classes(self) -> dict[int, str]:
        """Map ids to labels."""
        return self._classes

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, YoloDetectionDataset, dict[str, Any]]:
        """Get `index`-th element from dataset."""
        try:
            image = self._images[index]
            annotation = self._annotations[index]
        except IndexError as e:
            raise IndexError(
                f"The index number {index} is out of range for the dataset which has length {len(self.dataset)}",
            ) from e

        img_pt = pil_to_tensor(Image.open(os.path.join(self._train, image)))
        boxes = []
        labels = []
        with open(os.path.join(self._ann_dir, annotation)) as fd:
            for line in fd:
                label, x_center, y_center, width, height = line.split(" ")
                labels.append(int(label))
                boxes.append([float(x_center), float(y_center), float(width), float(height)])

        # format ground truth
        num_boxes = len(boxes)
        boxes = torch.zeros(num_boxes, 4)
        scores = torch.zeros(num_boxes)
        for i in range(num_boxes):
            boxes[i, :] = box_convert(
                torch.as_tensor(boxes[i]),
                in_fmt="cxcywh",
                out_fmt="xyxy",
            )
            scores[i] = 1
        labels = torch.as_tensor([self._id_to_int[label] for label in labels])

        metadata = {}
        return img_pt, DetectionTarget(boxes, labels, scores), metadata
