"""augmentation"""

import copy
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from maite.protocols import AugmentationMetadata
from maite.protocols.image_classification import (
    Augmentation,
    DatumMetadataType,
    InputType,
    TargetType,
)
from nrtk.interfaces.perturb_image import PerturbImage
from numpy.typing import NDArray

CLASSIFICATION_BATCH_T = tuple[Sequence[InputType], Sequence[TargetType], Sequence[DatumMetadataType]]


class JaticAugmentationMetadata(DatumMetadataType):
    """DatumMetadataType with extra key.

    Attributes
    ----------
    nrtk_perturber : dict[str, Any]
        The NRTK perturber configuration.

    """

    nrtk_perturber: dict[str, Any]


class JATICClassificationAugmentation(Augmentation):
    """
    Implementation of JATIC Augmentation for NRTK perturbers operating
    on a MAITE-protocol compliant Image Classification dataset for a
    channels-first input image format.

    Parameters
    ----------
    augment : PerturbImage
        Object used to apply augmentations to an image.
    augumentation_id : str, optional
        Identifier for the augmentation, by default "JATICClassification".

    Attributes
    ----------
    augment : PerturbImage
        Object used to apply augmentations to an image.
    metadata : AugmentationMetadata
        Metadata for the augmentation.
    """

    def __init__(self, augment: PerturbImage, augumentation_id: str = "JATICClassification") -> None:
        self.augment = augment
        self.metadata = AugmentationMetadata(id=augumentation_id)

    def __extract_aug_img(self, img: NDArray | tuple[NDArray, Any]) -> NDArray[Any]:
        """Extract augmented image from NRTK output.

        Returned augmented images can be as NDArray or tuple of NDArray.
        If tuple, the first element is the augmented image and the second is the dtype.

        Parameters
        ----------
        img : NDArray | tuple[NDArray, Any]
            The output from an NRTK perturber.

        Returns
        -------
        NDArray[Any]
            The augmented image.
        """
        if isinstance(img, tuple):
            return img[0]
        return img

    def __call__(
        self,
        batch: CLASSIFICATION_BATCH_T,
    ) -> CLASSIFICATION_BATCH_T:
        """Apply augmentations to the given data batch.

        Parameters
        ----------
        batch : CLASSIFICATION_BATCH_T
            A batch of data containing images, annotations, and metadata.

        Returns
        -------
        CLASSIFICATION_BATCH_T
            A batch of augmented data containing augmented images, original
            annotations, and updated metadata.
        """
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
            aug_img = self.augment(image=aug_img, additional_params=cast(dict[str, Any], md))
            aug_img = self.__extract_aug_img(aug_img)
            if aug_img.ndim > 2:
                aug_img = np.transpose(aug_img, (2, 0, 1))  # Need to transpose it back
            aug_imgs.append(aug_img)

            m_aug = JaticAugmentationMetadata(id=md["id"], nrtk_perturber=self.augment.get_config())
            aug_metadata.append(m_aug)
            aug_dets.append(ann)

        # return batch of augmented inputs, resized bounding boxes and updated metadata
        return aug_imgs, anns, aug_metadata
