"""XAITKTestStage implementation"""

# Python generic imports
from __future__ import annotations

import json
import os
from hashlib import sha256
from typing import Any

import maite.protocols.image_classification as ic

# 3rd party imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # type: ignore

# SMQTK imports
from smqtk_core.configuration import from_config_dict
from xaitk_jatic.interop.image_classification.model import JATICImageClassifier

# XAITK imports
from xaitk_saliency.interfaces.gen_image_classifier_blackbox_sal import GenerateImageClassifierBlackboxSaliency

# Import TestStage
from jatic_ri._common.test_stages.impls.xaitk_test_stage import XAITKTestStageBase
from jatic_ri.util.cache import NumpyEncoder


class XAITKTestStage(XAITKTestStageBase[ic.Model, ic.Dataset, ic.Metric]):
    """
    XAITKTestStage will generate saliency maps for each model target across all images in the dataset.


    Attributes:
        sal_generator (GenerateImageClassifierBlackboxSaliency):The saliency map generator used to
                                                                generate saliency maps.
    """

    sal_generator: GenerateImageClassifierBlackboxSaliency
    _task: str = "ic"

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args)
        self.sal_generator = from_config_dict(
            args["saliency_generator"],
            GenerateImageClassifierBlackboxSaliency.get_impls(),
        )
        self.sal_generator_hash: str = sha256(
            json.dumps(args["saliency_generator"]).encode("utf-8"),
        ).hexdigest()

    def _run(self) -> dict[str, Any]:
        """Run the test stage, and store any outputs of the saliency
        generation in test stage"""

        classifier = JATICImageClassifier(
            classifier=self.model,
            ids=sorted(self.model.index2label.keys()),  # type: ignore
            img_batch_size=self.img_batch_size,
        )
        img_sal_maps = [
            self.sal_generator(np.asarray(ref_img).transpose(1, 2, 0), classifier) for ref_img, _, _ in self.dataset
        ]

        return {"saliency_map_" + str(i): NumpyEncoder().default(sal_map) for i, sal_map in enumerate(img_sal_maps)}

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report
        generated in the run method or in the load_cached_results method.
        Each slide will have a short text description of the model used,
        the prediction, the ground truth, and the saliency map for the
        detection. The saliency map image will have the detection bounding
        box shown in red."""

        gradient_slides = []
        output_values = list(self.outputs.values())

        for dataset_idx in range(len(self.dataset)):
            ref_img, targets, _ = self.dataset[dataset_idx]
            sub_dir = os.path.join(os.path.splitext(self.cache_path)[0], f"img_{dataset_idx}")

            os.makedirs(sub_dir, exist_ok=True)
            ref_img = np.asarray(ref_img)
            if ref_img.shape[0] == 1:
                gray_img = np.asarray(Image.fromarray(ref_img[0]).convert("L"))
            else:
                gray_img = np.asarray(Image.fromarray(ref_img.transpose(1, 2, 0)).convert("L"))

            gt_label = self.model.index2label[int(np.argmax(targets))]  # type: ignore
            sal_maps = output_values[dataset_idx]
            for sal_idx, sal_map in enumerate(sal_maps):
                sal_map_path = os.path.join(sub_dir, f"class_{self.model.index2label[sal_idx]}.png")  # type: ignore

                fig = plt.figure()
                plt.axis("off")
                plt.imshow(gray_img, alpha=0.7, cmap="gray")
                plt.xticks(())
                plt.yticks(())

                plt.imshow(sal_map, cmap="jet", alpha=0.3)
                plt.colorbar()
                plt.savefig(sal_map_path, bbox_inches="tight")
                plt.close(fig)

                gradient_slides.append(
                    {
                        "deck": "image_classification_model_evaluation",
                        "layout_name": "OneImageText",
                        "layout_arguments": {
                            "title": f"**XAITK Saliency Map**: {sal_idx} \n",
                            "text": (
                                f"Model: {self.model_id}\n"
                                f"Image: {dataset_idx}\n"
                                f"GT: {gt_label}\n"
                                f"Pred: {self.model.index2label[sal_idx]}"  # type: ignore
                            ),
                            "image_path": sal_map_path,
                        },
                    },
                )

        return gradient_slides
