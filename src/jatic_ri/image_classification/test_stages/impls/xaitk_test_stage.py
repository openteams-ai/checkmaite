"""XAITKTestStage implementation"""

# Python generic imports

from pathlib import Path
from typing import Any

import maite.protocols.image_classification as ic

# 3rd party imports
import matplotlib.pyplot as plt
import numpy as np
import torch

# SMQTK imports
from torchvision.transforms.v2.functional import rgb_to_grayscale
from xaitk_jatic.interop.image_classification.model import JATICImageClassifier

# XAITK imports
from xaitk_saliency.interfaces.gen_image_classifier_blackbox_sal import GenerateImageClassifierBlackboxSaliency

# Import TestStage
from jatic_ri._common.test_stages.impls.xaitk_test_stage import XAITKTestStageBase
from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, OutputsBase, RunBase
from jatic_ri.util._types import DeSerializablePlugfigurable
from jatic_ri.util.utils import save_figure_to_tempfile


class XAITKOutputsIC(OutputsBase):
    """Each item in 'results' corresponds to one item in the input dataset"""

    # TODO: RISE algorithm is shape is (Cl, H, W) where Cl is the number of classes in the model index2label
    # Why does the saliency generator care about generating a sal for every single class in the model?  This must be
    # related to issue #345
    # Furthermore... MC-RISE is (X, Cl, H, W) where X is the number of different color masks specified in list in
    # MCRISEStack["fill_colors"]
    # REF: https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/345
    results: list[np.ndarray]


class XAITKConfigIC(ConfigBase):
    """Config class for XAITKTestStage"""

    name: str
    saliency_generator: DeSerializablePlugfigurable[GenerateImageClassifierBlackboxSaliency]
    img_batch_size: int


class XAITKRunIC(RunBase):
    """Run class for XAITKTestStage for OD"""

    config: XAITKConfigIC
    outputs: XAITKOutputsIC


class XAITKTestStage(XAITKTestStageBase[XAITKConfigIC, XAITKOutputsIC, ic.Model, ic.Dataset]):
    """
    XAITKTestStage will generate saliency maps for every detections in all images from the dataset.

    Attributes:
        config (XAITKConfigIC):The config used to run XAITK saliency test stage for IC.
    """

    _task: str = "ic"
    _RUN_TYPE = XAITKRunIC

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config: XAITKConfigIC = XAITKConfigIC.model_validate(config)

    def _run(self) -> XAITKOutputsIC:
        """Run the test stage, and store any outputs of the saliency
        generation in test stage"""
        if "index2label" not in self.model.metadata:
            raise (KeyError("'index2label' not found in model metadata but is required by XAITKTestStage"))

        classifier = JATICImageClassifier(
            classifier=self.model,
            ids=sorted(self.model.metadata["index2label"].keys()),
            img_batch_size=self.config.img_batch_size,
        )
        img_sal_maps = []
        for ref_img, _, _ in self.dataset:
            transposed_image = np.asarray(ref_img).transpose(1, 2, 0)
            if transposed_image.shape[2] == 1:
                transposed_image = np.concatenate((transposed_image,) * 3, axis=-1)
            img_sal_maps.append(self.config.saliency_generator(transposed_image, classifier))

        return XAITKOutputsIC.model_validate({"results": img_sal_maps})

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report
        generated in the run method or in the load_cached_results method.
        Each slide will have a short text description of the model used,
        the prediction, the ground truth, and the saliency map for the
        detection. The saliency map image will have the detection bounding
        box shown in red."""

        gradient_slides = []

        # TODO: This needs to be fixed to work from cache (self.dataset won't be available)
        # TODO: Also needs general refactoring once a determination is made on if/why the saliency maps are
        # being generated for every class in the model for every image.
        # REF: https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/345
        for i, sal_maps in enumerate(self.outputs.results):
            ref_img, targets, _ = self.dataset[i]

            gray_img = rgb_to_grayscale(torch.as_tensor(ref_img)).squeeze(0).numpy()

            gt_label = self.model.metadata["index2label"][int(np.argmax(targets))]  # type: ignore
            conf_dict = self.config.model_dump()

            if "MCRISEStack" in conf_dict["saliency_generator"]["type"]:
                fill_colors = conf_dict["saliency_generator"][conf_dict["saliency_generator"]["type"]]["fill_colors"]
                for color_idx, color_value in enumerate(fill_colors):
                    for sal_idx in self.model.metadata["index2label"]:  # type: ignore
                        fig = plt.figure()
                        plt.axis("off")
                        plt.imshow(gray_img, alpha=0.7, cmap="gray")
                        plt.xticks(())
                        plt.yticks(())
                        plt.imshow(np.asarray(sal_maps[color_idx][0]), cmap="jet", alpha=0.3)
                        plt.colorbar()
                        fig.tight_layout()
                        plt.close(fig)

                        content = {
                            "deck": "image_classification_model_evaluation",
                            "layout_name": "OneImageText",
                            "layout_arguments": {
                                "title": f"**XAITK Saliency Map**: {sal_idx} \n",
                                "text": (
                                    f"Model: {self.model_id}\n"
                                    f"Image: {i}\n"
                                    f"Fill Color: {color_value}\n"
                                    f"GT: {gt_label}\n"
                                    f"Pred: {self.model.metadata['index2label'][sal_idx]}"  # type: ignore
                                ),
                                "image_path": Path(save_figure_to_tempfile(fig)),
                            },
                        }
                        content["layout_arguments"]["text"] = content["layout_arguments"]["text"].replace("_", r"\_")
                        gradient_slides.append(content)
            else:
                for sal_idx, sal_map in enumerate(sal_maps):
                    fig = plt.figure()
                    plt.axis("off")
                    plt.imshow(gray_img, alpha=0.7, cmap="gray")
                    plt.xticks(())
                    plt.yticks(())

                    plt.imshow(sal_map, cmap="jet", alpha=0.3)
                    plt.colorbar()
                    fig.tight_layout()
                    plt.close(fig)

                    content = {
                        "deck": "image_classification_model_evaluation",
                        "layout_name": "OneImageText",
                        "layout_arguments": {
                            "title": f"**XAITK Saliency Map**: {sal_idx} \n",
                            "text": (
                                f"Model: {self.model_id}\n"
                                f"Image: {i}\n"
                                f"GT: {gt_label}\n"
                                f"Pred: {self.model.metadata['index2label'][sal_idx]}"  # type: ignore
                            ),
                            "image_path": Path(save_figure_to_tempfile(fig)),
                        },
                    }
                    content["layout_arguments"]["text"] = content["layout_arguments"]["text"].replace("_", r"\_")
                    gradient_slides.append(content)

        return gradient_slides
