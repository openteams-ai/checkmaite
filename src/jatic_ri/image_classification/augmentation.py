"""augmentation"""

from __future__ import annotations

import copy
from typing import Any, cast

import numpy as np
from maite.protocols import AugmentationMetadata
from maite.protocols.image_classification import (
    Augmentation,
    DatumMetadataBatchType,
    DatumMetadataType,
    InputBatchType,
    TargetBatchType,
)
from nrtk.interfaces.perturb_image import PerturbImage

CLASSIFICATION_BATCH_T = tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType]


class JaticAugmentationMetadata(DatumMetadataType):
    """DatumMetadataType with extra key."""

    nrtk_perturber: dict[str, Any]


class JATICClassificationAugmentation(Augmentation):
    """
    Implementation of JATIC Augmentation for NRTK perturbers operating
    on a MAITE-protocol compliant Image Classification dataset for a
    channels-first input image format.

    Parameters
    ----------
    augment : nrtk.interfaces.perturb_image.PerturbImage
        Object used to apply augmentations to an image.
    """

    def __init__(self, augment: PerturbImage, augumentation_id: str = "JATICClassification") -> None:
        self.augment = augment
        self.metadata = AugmentationMetadata(id=augumentation_id)

    def __call__(
        self,
        batch: CLASSIFICATION_BATCH_T,
    ) -> CLASSIFICATION_BATCH_T:
        """Apply augmentations to the given data batch."""
        imgs, anns, metadata = batch
        imgs = np.asarray(imgs)
        imgs_new = np.transpose(imgs, (0, 2, 3, 1))

        # iterate over (parallel) elements in batch
        aug_imgs = []  # list of individual augmented inputs
        aug_dets = []  # list of individual object detection targets
        aug_metadata = []  # list of individual image-level metadata

        for img, ann, md in zip(imgs_new, anns, metadata, strict=False):
            # Perform augmentation
            aug_img = copy.deepcopy(img)
            aug_img = self.augment(aug_img, cast(dict[str, Any], md))
            if aug_img.ndim > 2:
                aug_img = np.transpose(aug_img, (2, 0, 1))  # Need to transpose it back
            aug_imgs.append(aug_img)

            m_aug = JaticAugmentationMetadata(id=md["id"], nrtk_perturber=self.augment.get_config())
            aug_metadata.append(m_aug)
            aug_dets.append(ann)

        # return batch of augmented inputs, resized bounding boxes and updated metadata
        return aug_imgs, anns, aug_metadata
