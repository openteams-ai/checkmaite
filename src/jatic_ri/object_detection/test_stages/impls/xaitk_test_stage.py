"""XAITKTestStage implementation"""

# Python generic imports
from __future__ import annotations

import json
from collections.abc import Hashable
from hashlib import sha256
from typing import Any

# SMQTK imports
from smqtk_core.configuration import from_config_dict

# XAITK imports
from xaitk_jatic.utils.sal_on_dets import sal_on_dets
from xaitk_saliency import GenerateObjectDetectorBlackboxSaliency

# Import TestStage
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import (
    MetricPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    ThresholdPlugin,
)
from jatic_ri.util.cache import JSONCache, NumpyEncoder


class XAITKTestStage(
    TestStage[list[dict[str, Any]]],
    SingleDatasetPlugin,
    SingleModelPlugin,
    MetricPlugin,
    ThresholdPlugin,
):
    """
    Base XAITK Test Stage that takes in the necessary saliency generator and id2label mapping
    to demo saliency map generation.
    """

    config: dict[str, Any]
    stage_name: str
    sal_generator: GenerateObjectDetectorBlackboxSaliency
    sal_generator_hash: str
    id2label: dict[int, Hashable]
    outputs: list[dict[str, Any]] | None
    cache: Cache[list[dict[str, Any]]] | None = JSONCache()

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.outputs = None
        self.config = args
        self.stage_name = args["name"]
        self.sal_generator = from_config_dict(
            args["GenerateObjectDetectorBlackboxSaliency"],
            GenerateObjectDetectorBlackboxSaliency.get_impls(),
        )
        self.id2label = args["id2label"]
        self.sal_generator_hash = sha256(
            json.dumps(args["GenerateObjectDetectorBlackboxSaliency"]).encode("utf-8"),
        ).hexdigest()

    @property
    def cache_id(self) -> str:
        """Cache file for XAITK Test Stage"""
        return f"xaitk_{self.model_id}_{self.dataset_id}_{self.sal_generator_hash}.json"

    def _run(self) -> None:
        """Run the test stage, and store any outputs of the saliency
        generation in test stage"""

        self.outputs = list()  # noqa: C408

        img_sal_maps, _ = sal_on_dets(
            dataset=self.dataset,
            sal_generator=self.sal_generator,
            detector=self.model,
            id_to_name=self.id2label,
        )

        sal_maps_json_serializable = {
            "saliency_map_" + str(i): NumpyEncoder().default(sal_map) for i, sal_map in enumerate(img_sal_maps)
        }

        self.outputs.append(sal_maps_json_serializable)

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report
        generated in the run method or in the load_cached_results method"""

        return []

    @property
    def name(self) -> str:
        """Returns classname as a string"""
        return self.__class__.__name__
