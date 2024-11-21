"""augmentation"""

import copy

import numpy as np
from maite.protocols.image_classification import Augmentation, DatumMetadataBatchType, InputBatchType, TargetBatchType
from nrtk.interfaces.perturb_image import PerturbImage

CLASSIFICATION_BATCH_T = tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType]


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

    def __init__(self, augment: PerturbImage) -> None:
        self.augment = augment

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

        for img, ann, md in zip(imgs_new, anns, metadata):
            # Perform augmentation
            aug_img = copy.deepcopy(img)
            aug_img = self.augment(aug_img, md)
            if aug_img.ndim > 2:
                aug_img = np.transpose(aug_img, (2, 0, 1))  # Need to transpose it back
            aug_imgs.append(aug_img)

            m_aug = copy.deepcopy(md)
            m_aug.update({"nrtk::perturber": self.augment.get_config()})
            aug_metadata.append(m_aug)
            aug_dets.append(ann)

        # return batch of augmented inputs, resized bounding boxes and updated metadata
        return aug_imgs, anns, aug_metadata
