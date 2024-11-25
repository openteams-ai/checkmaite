"""XAITKTestStage implementation"""

# Python generic imports
from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import maite.protocols.object_detection as od

# 3rd Party Imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle  # type: ignore
from PIL import Image  # type: ignore

# SMQTK imports
from smqtk_core.configuration import from_config_dict
from xaitk_jatic.utils.sal_on_dets import sal_on_dets

# XAITK imports
from xaitk_saliency.interfaces.gen_object_detector_blackbox_sal import GenerateObjectDetectorBlackboxSaliency

from jatic_ri._common.test_stages.impls.xaitk_test_stage import XAITKTestStageBase

# Import TestStage
from jatic_ri.util.cache import NumpyEncoder


class XAITKTestStage(XAITKTestStageBase[od.Model, od.Dataset, od.Metric]):
    """
    XAITKTestStage will generate saliency maps for every detections in all images from the dataset.


    Attributes:
        sal_generator (GenerateImageClassifierBlackboxSaliency):The saliency map generator used to
                                                                generate saliency maps.
    """

    sal_generator: GenerateObjectDetectorBlackboxSaliency
    _task: str = "od"

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args)
        self.sal_generator = from_config_dict(
            args["saliency_generator"],
            GenerateObjectDetectorBlackboxSaliency.get_impls(),
        )
        self.sal_generator_hash: str = sha256(
            json.dumps(args["saliency_generator"]).encode("utf-8"),
        ).hexdigest()

    def _run(self) -> dict[str, Any]:
        """Run the test stage, and store any outputs of the saliency
        generation in test stage"""
        img_sal_maps, _ = sal_on_dets(
            dataset=self.dataset,
            sal_generator=self.sal_generator,
            detector=self.model,
            id_to_name=self.model.index2label,  # type: ignore
        )

        return {"saliency_map_" + str(i): NumpyEncoder().default(sal_map) for i, sal_map in enumerate(img_sal_maps)}

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report
        generated in the run method or in the load_cached_results method.
        Each slide will have a short text description of the model used,
        the image id, the prediction, the ground truth, and the saliency
        map for target class."""

        gradient_slides = []
        output_values = list(self.outputs.values())

        for dataset_idx in range(len(self.dataset)):
            ref_img, dets, _ = self.dataset[dataset_idx]
            sub_dir = os.path.join(os.path.splitext(self.cache_path)[0], f"img_{dataset_idx}")

            os.makedirs(sub_dir, exist_ok=True)
            labels = np.asarray(dets.labels)
            bboxes = np.asarray(dets.boxes)
            scores = np.asarray(dets.scores)
            sal_maps = output_values[dataset_idx]
            for sal_idx, bbox in enumerate(bboxes):
                sal_map = sal_maps[sal_idx]
                ref_img_np = np.asarray(ref_img)
                if ref_img_np.shape[0] == 1:
                    gray_img = np.asarray(Image.fromarray(ref_img_np[0]).convert("L"))
                else:
                    gray_img = np.asarray(Image.fromarray(ref_img_np).convert("L"))

                sal_map_path = Path(sub_dir).joinpath(f"det_{sal_idx}.png")

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

                content = {
                    "deck": "object_detection_dataset_evaluation",
                    "layout_name": "OneImageText",
                    "layout_arguments": {
                        "title": f"**XAITK Saliency Map**: {sal_idx} \n",
                        "text": (
                            f"Model: {self.model_id}\nImage: {dataset_idx}\n"
                            f"GT: {self.model.index2label[int(labels[sal_idx])]}\n"  # type: ignore
                            f"Pred: {self.model.index2label[int(np.argmax(scores))]}"  # type: ignore
                        ),
                        "image_path": sal_map_path,
                    },
                }
                content["layout_arguments"]["text"] = content["layout_arguments"]["text"].replace("_", r"\_")
                gradient_slides.append(content)

        return gradient_slides
