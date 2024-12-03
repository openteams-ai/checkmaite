"""NRTKTestStage implementation"""

# Python generic imports
from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

import maite.protocols.image_classification as ic
from nrtk.interfaces.perturb_image import PerturbImage

# NRTK imports
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

# SMQTK imports
from smqtk_core.configuration import from_config_dict

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
        self.factory = from_config_dict(args["perturber_factory"], PerturbImageFactory.get_impls())
        self.factory_hash: str = sha256(json.dumps(args["perturber_factory"]).encode("utf-8")).hexdigest()

    def _augmentation_wrapper(
        self, perturber: PerturbImage
    ) -> JATICDetectionAugmentation | JATICClassificationAugmentation:
        return JATICClassificationAugmentation(perturber)
