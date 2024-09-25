"""NRTKTestStage implementation"""

# Python generic imports
from __future__ import annotations

from typing import Any

# MAITE imports
# 3rd party imports
# NRTK imports
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

# SMQTK imports
from smqtk_core.configuration import from_config_dict

# Import TestStage
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import (
    MetricThresholdPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
)
from jatic_ri.util.cache import JSONCache

# Not implemented yet. Keeping for future use
# from jatic_ri.object_detection.augmentation import JATICDetectionAugmentation
# from jatic_ri.image_classification.augmentation import JATICClassificationAugmentation

DECK_MAP = {"classification": "image_classification_model_evaluation", "detection": "object_detection_model_evaluation"}


class NRTKTestStage(TestStage[list[dict[str, Any]]], SingleDatasetPlugin, SingleModelPlugin, MetricThresholdPlugin):
    """
    Base NRTK Test Stage that takes in the necessary Sensor, Scenario and Image params
    needed to demo the JitterOTF Perturber.
    """

    config: dict[str, Any]
    stage_name: str
    factory: PerturbImageFactory
    outputs: list[dict[str, Any]] | None
    cache: Cache[list[dict[str, Any]]] | None = JSONCache()

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.outputs = None
        self.config = args
        self.stage_name = args["name"]
        self.factory = from_config_dict(args["perturber_factory"], PerturbImageFactory.get_impls())

    @property
    def cache_id(self) -> str:
        """Cache file for NRTK Test Stage"""
        return f"nrtk_{self.model_id}_{self.dataset_id}.json"

    def _run(self) -> None:
        """Run the test stage, and store any outputs of the evaluation in test stage"""
        # WIP: Method not tested and not completely fleshed out.
        # Run perturber factory (not implemented)

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report generated in the run method or in the
        load_cached_results method"""

        return []
