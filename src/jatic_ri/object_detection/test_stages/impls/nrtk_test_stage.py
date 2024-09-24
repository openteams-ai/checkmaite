"""NRTKTestStage implementation"""

# Python generic imports
from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path
from typing import Any

# MAITE imports
# 3rd party imports
# NRTK imports
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

# SMQTK imports
from smqtk_core.configuration import from_config_dict

# Import TestStage
from jatic_ri._common.test_stages.interfaces.test_stage import TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import (
    MetricThresholdPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
)

# Not implemented yet. Keeping for future use
# from jatic_ri.object_detection.augmentation import JATICDetectionAugmentation
# from jatic_ri.image_classification.augmentation import JATICClassificationAugmentation

DECK_MAP = {"classification": "image_classification_model_evaluation", "detection": "object_detection_model_evaluation"}

CODE_DIR = Path(os.path.abspath(__file__)).parent


class NRTKTestStage(TestStage, SingleDatasetPlugin, SingleModelPlugin, MetricThresholdPlugin):
    """
    Base NRTK Test Stage that takes in the necessary Sensor, Scenario and Image params
    needed to demo the JitterOTF Perturber.
    """

    CACHE_DIR = CODE_DIR / ".nrtk_cache"
    config: dict[str, Any]
    stage_name: str
    factory: PerturbImageFactory
    cur_cache_dir: str

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.outputs = None
        self.config = args
        self.stage_name = args["name"]
        self.factory = from_config_dict(args["perturber_factory"], PerturbImageFactory.get_impls())
        self.cur_cache_dir = ""

    def run(self, use_cache: bool = True) -> None:
        """Run the test stage, and store any outputs of the evaluation in test stage"""
        # WIP: Method not tested and not completely fleshed out.
        self.outputs = []

        cache_dirs = glob(str(Path(self.CACHE_DIR) / f"{self.model_id}_{self.dataset_id}_*"))

        if use_cache:
            # first try to load results from cache
            cache_hit = False
            for cd in cache_dirs:
                with open(Path(cd) / "config.json") as f:
                    c = json.load(f)
                    if c == self.config:
                        print("Cache Hit")
                        cache_hit = True
                        self.cur_cache_dir = cd
                        break
            if cache_hit:
                results_files = glob(str(Path(self.cur_cache_dir) / "result_*"))
                for i in range(len(results_files)):
                    # Make sure these are opened in order
                    result_file = Path(self.cur_cache_dir) / f"result_{i}.json"
                    with open(result_file) as f:
                        self.outputs.append(json.load(f))
                return
            print("No Cache Hit, running evaluation")

        # Run perturber factory (not implemented)

        self.cur_cache_dir = str(Path(self.CACHE_DIR) / f"{self.model_id}_{self.dataset_id}_{len(cache_dirs)}")
        os.makedirs(Path(self.cur_cache_dir), exist_ok=True)
        with open(Path(self.cur_cache_dir) / "config.json", "w") as f:
            json.dump(self.config, f)
        for i, res in enumerate(self.outputs):
            with open(Path(self.cur_cache_dir) / f"result_{i}.json", "w") as f:
                out = {}
                for key, val in res.items():
                    out[key] = float(val)
                json.dump(out, f)

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report generated in the run method or in the
        load_cached_results method"""

        return []
