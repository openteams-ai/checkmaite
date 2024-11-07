"""XAITKTestStage implementation"""

# Python generic imports
from __future__ import annotations

import json
import os
from collections.abc import Hashable
from hashlib import sha256
from typing import Any

import maite.protocols.object_detection as od

# 3rd Party Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle  # type: ignore
from PIL import Image  # type: ignore

# SMQTK imports
from smqtk_core.configuration import from_config_dict

# XAITK imports
from xaitk_jatic.utils.sal_on_dets import sal_on_dets
from xaitk_saliency import GenerateObjectDetectorBlackboxSaliency

# Import TestStage
from jatic_ri._common.test_stages.interfaces.plugins import (
    MetricPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    ThresholdPlugin,
)
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.util.cache import JSONCache, NumpyEncoder


class XAITKTestStage(
    TestStage[list[dict[str, Any]]],
    SingleDatasetPlugin[od.Dataset],
    SingleModelPlugin[od.Model],
    MetricPlugin[od.Metric],
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
    cache: Cache[list[dict[str, Any]]] | None = JSONCache()

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
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

    def _run(self) -> list[dict[str, Any]]:
        """Run the test stage, and store any outputs of the saliency
        generation in test stage"""

        outputs = []
        img_sal_maps, _ = sal_on_dets(
            dataset=self.dataset,
            sal_generator=self.sal_generator,
            detector=self.model,
            id_to_name=self.id2label,
        )

        sal_maps_json_serializable = {
            "saliency_map_" + str(i): NumpyEncoder().default(sal_map) for i, sal_map in enumerate(img_sal_maps)
        }

        outputs.append(sal_maps_json_serializable)

        return outputs

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report
        generated in the run method or in the load_cached_results method"""

        gradient_slides = []
        output_values = list(self.outputs[0].values())

        for dataset_idx, (ref_img, dets, _) in enumerate(self.dataset):
            sub_dir = os.path.join(self.cache_path[:-5], f"img_{dataset_idx}")

            os.makedirs(sub_dir, exist_ok=True)
            bboxes = np.asarray(dets.boxes)
            for sal_idx, bbox in enumerate(bboxes):
                sal_map = output_values[dataset_idx][sal_idx]
                if ref_img.shape[0] == 1:
                    gray_img = np.asarray(Image.fromarray(ref_img[0].numpy()).convert("L"))
                else:
                    gray_img = np.asarray(Image.fromarray(ref_img.numpy()).convert("L"))

                sal_map_path = os.path.join(sub_dir, f"det_{sal_idx}.png")

                fig = plt.figure()
                plt.axis("off")
                plt.imshow(gray_img, alpha=0.7, cmap="gray")
                plt.xticks(())
                plt.yticks(())

                plt.gca().add_patch(
                    Rectangle(
                        (bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                    ),
                )
                plt.imshow(sal_map, cmap="jet", alpha=0.3)
                plt.colorbar()
                plt.savefig(sal_map_path, bbox_inches="tight")
                plt.close(fig)

                gradient_slides.append(
                    {
                        "deck": "object_detection_dataset_evaluation",
                        "layout_name": "OneImageText",
                        "layout_arguments": {
                            "title": f"**XAITK Saliency Map**: {sal_idx} \n",
                            "text": (
                                f"Model: {self.model_id}\nImage: {dataset_idx}\n"
                                f"GT: {self.id2label[dets.labels[sal_idx].item()]}\n"
                                f"Pred: {self.id2label[torch.argmax(dets.scores).item()]}"
                            ),
                            "image_path": sal_map_path,
                        },
                    },
                )

        return gradient_slides

    @property
    def name(self) -> str:
        """Returns classname as a string"""
        return self.__class__.__name__
