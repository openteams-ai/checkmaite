"""NRTKTestStage implementation"""

# Python generic imports

from typing import Any

import maite.protocols.image_classification as ic
from nrtk.interfaces.perturb_image import PerturbImage

# NRTK imports
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

# SMQTK imports
# Import TestStage
from jatic_ri._common.test_stages.impls.nrtk_test_stage import NRTKTestStageBase
from jatic_ri.image_classification.augmentation import JATICClassificationAugmentation
from jatic_ri.object_detection.augmentation import JATICDetectionAugmentation


class NRTKTestStage(NRTKTestStageBase[ic.Dataset, ic.Model, ic.Metric]):
    """
    NRTK Test Stage to perform augmentation on images in a dataset based
    on a given factory configuration.`
    """

    factory: PerturbImageFactory

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args)
        self._deck = "image_classification_dataset_evaluation"
        self._task = "image_classification"

    def _augmentation_wrapper(
        self, perturber: PerturbImage
    ) -> JATICDetectionAugmentation | JATICClassificationAugmentation:
        return JATICClassificationAugmentation(perturber)
