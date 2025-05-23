"""augmentation"""

import copy
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from maite.protocols import AugmentationMetadata
from maite.protocols.object_detection import (
    Augmentation,
    DatumMetadataBatchType,
    DatumMetadataType,
    InputBatchType,
    TargetBatchType,
)
from nrtk.interfaces.perturb_image import PerturbImage
from numpy.typing import NDArray

OBJ_DETECTION_BATCH_T = tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType]


@dataclass
class JATICDetectionTarget:
    """Dataclass for the datum-level JATIC output detection format."""

    boxes: np.ndarray[Any, Any]
    labels: np.ndarray[Any, Any]
    scores: np.ndarray[Any, Any]


class JATICDetectionAugmentation(Augmentation):
    """
    Implementation of JATIC Augmentation for NRTK perturbers operating
    on a MAITE-protocol compliant Object Detection dataset for a
    channels-first input image format.

    Parameters
    ----------
    augment : PerturbImage
        Augmentations to apply to an image.
    """

    def __init__(self, augment: PerturbImage, *, augmentation_id: str = "JATICDetection") -> None:
        self.augment = augment
        self.metadata = AugmentationMetadata(id=augmentation_id)

    def __extract_aug_img(self, img: NDArray | tuple[NDArray, Any]) -> NDArray[Any]:
        """
        Returned augmented images can be as NDArray or tuple of NDArray.
        If tuple, the first element is the augmented image and the second is the dtype.
        """
        if isinstance(img, tuple):
            return img[0]
        return img

    def __call__(
        self,
        batch: OBJ_DETECTION_BATCH_T,
    ) -> OBJ_DETECTION_BATCH_T:
        """Apply augmentations to the given data batch."""
        imgs, anns, metadata = batch

        # iterate over (parallel) elements in batch
        aug_imgs: list[NDArray[Any]] = []  # list of individual augmented inputs
        aug_dets: list[JATICDetectionTarget] = []  # list of individual object detection targets
        aug_metadata: list[DatumMetadataType] = []  # list of individual image-level metadata

        for img, ann, md in zip(imgs, anns, metadata, strict=False):
            # Perform augmentation
            aug_img = np.array(img, copy=True).transpose((1, 2, 0))
            height, width = aug_img.shape[0:2]
            aug_img = self.augment(image=aug_img, additional_params=cast(dict[str, Any], md))
            aug_img = self.__extract_aug_img(aug_img)
            aug_height, aug_width = aug_img.shape[0:2]
            if aug_img.ndim > 2:
                aug_img = np.transpose(aug_img, (2, 0, 1))  # Need to transpose it back
            aug_imgs.append(aug_img)

            # Resize bounding boxes
            y_aug_boxes = copy.deepcopy(np.asarray(ann.boxes))
            y_aug_labels = copy.deepcopy(np.asarray(ann.labels))
            y_aug_scores = copy.deepcopy(np.asarray(ann.scores))
            y_aug_boxes[:, 0] *= aug_width / width
            y_aug_boxes[:, 1] *= aug_height / height
            y_aug_boxes[:, 2] *= aug_width / width
            y_aug_boxes[:, 3] *= aug_height / height
            aug_dets.append(
                JATICDetectionTarget(
                    y_aug_boxes,
                    y_aug_labels,
                    y_aug_scores,
                ),
            )

            m_aug = cast(dict[str, Any], copy.deepcopy(md))
            m_aug.update({"nrtk::perturber": self.augment.get_config()})
            aug_metadata.append(cast(DatumMetadataType, m_aug))

        # return batch of augmented inputs, resized bounding boxes and updated metadata
        return aug_imgs, aug_dets, aug_metadata
