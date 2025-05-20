"""NRTKTestStage implementation"""

# Python generic imports
from __future__ import annotations

from typing import Any

import maite.protocols.object_detection as od
from nrtk.interfaces.perturb_image import PerturbImage

# NRTK imports
# SMQTK imports
# Import TestStage
from jatic_ri._common.test_stages.impls.nrtk_test_stage import NRTKTestStageBase
from jatic_ri.image_classification.augmentation import JATICClassificationAugmentation
from jatic_ri.object_detection.augmentation import JATICDetectionAugmentation


class NRTKTestStage(NRTKTestStageBase[od.Dataset, od.Model, od.Metric]):
    """
    NRTK Test Stage to perform augmentation on images in a dataset based
    on a given factory configuration.
    """

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args)
        self._deck = "object_detection_dataset_evaluation"
        self._task = "object_detection"

    def _augmentation_wrapper(
        self, perturber: PerturbImage
    ) -> JATICDetectionAugmentation | JATICClassificationAugmentation:
        return JATICDetectionAugmentation(perturber)
