"""NRTKTestStage implementation"""

# Python generic imports
from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

# MAITE imports
from maite.workflows import evaluate

# MAITE imports
# 3rd party imports
# NRTK imports
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

# SMQTK imports
from smqtk_core.configuration import from_config_dict

# Import TestStage
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.object_detection.augmentation import JATICDetectionAugmentation
from jatic_ri.object_detection.test_stages.interfaces.plugins import (
    MetricPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    ThresholdPlugin,
)
from jatic_ri.util.cache import JSONCache


class NRTKTestStage(
    TestStage[list[dict[str, Any]]],
    SingleDatasetPlugin,
    SingleModelPlugin,
    MetricPlugin,
    ThresholdPlugin,
):
    """
    NRTK Test Stage to perform augmentation on images in a dataset based
    on a given factory configuration.
    """

    config: dict[str, Any]
    stage_name: str
    factory: PerturbImageFactory
    factory_hash: str
    outputs: list[dict[str, Any]] | None
    cache: Cache[list[dict[str, Any]]] | None = JSONCache()

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.outputs = None
        self.config = args
        self.stage_name = args["name"]
        self.factory = from_config_dict(args["perturber_factory"], PerturbImageFactory.get_impls())
        self.factory_hash = sha256(json.dumps(args["perturber_factory"]).encode("utf-8")).hexdigest()

    @property
    def cache_id(self) -> str:
        """Cache file for NRTK Test Stage"""
        return f"nrtk_{self.model_id}_{self.dataset_id}_{self.factory_hash}.json"

    def _run(self) -> None:
        """Run the test stage, and store any outputs of the evaluation in test stage"""

        self.outputs = []

        for perturber in self.factory:
            augmentation = JATICDetectionAugmentation(perturber)

            perturbed_metrics, _, _ = evaluate(
                model=self.model,
                dataset=self.dataset,
                metric=self.metric,
                batch_size=1,
                augmentation=augmentation,
                return_augmented_data=False,
                return_preds=False,
            )
            self.outputs.append(perturbed_metrics)

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report generated in the run method or in the
        load_cached_results method"""

        return []
