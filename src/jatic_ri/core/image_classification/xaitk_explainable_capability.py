from pathlib import Path
from typing import Any

import maite.protocols.image_classification as ic
import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import Field
from smqtk_core.configuration import from_config_dict
from torchvision.transforms.v2.functional import rgb_to_grayscale
from xaitk_jatic.interop.image_classification.model import JATICImageClassifier
from xaitk_saliency.interfaces.gen_image_classifier_blackbox_sal import GenerateImageClassifierBlackboxSaliency

from jatic_ri.core._common.xaitk_explainable_capability import XaitkExplainableBase
from jatic_ri.core._types import DeSerializablePlugfigurable
from jatic_ri.core.capability_core import CapabilityConfigBase, CapabilityOutputsBase, CapabilityRunBase
from jatic_ri.core.report._plotting_utils import save_figure_to_tempfile


class XaitkExplainableOutputs(CapabilityOutputsBase):
    """Each item in 'results' corresponds to one item in the input dataset"""

    # TODO: RISE algorithm is shape is (Cl, H, W) where Cl is the number of classes in the model index2label
    # Why does the saliency generator care about generating a sal for every single class in the model?  This must be
    # related to issue #345
    # Furthermore... MC-RISE is (X, Cl, H, W) where X is the number of different color masks specified in list in
    # MCRISEStack["fill_colors"]
    # REF: https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/345
    results: list[np.ndarray]
    gray_imgs: list[np.ndarray]
    gt_labels: list[str]


def _default_saliency_factory() -> GenerateImageClassifierBlackboxSaliency:
    return from_config_dict(
        {
            "type": "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise.RISEStack",
            "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise.RISEStack": {
                "n": 50,
                "s": 7,
                "p1": 0.7,
                "seed": 42,
                "threads": 8,
                "debiased": True,
            },
        },
        GenerateImageClassifierBlackboxSaliency.get_impls(),
    )


class XaitkExplainableConfig(CapabilityConfigBase):
    """Config class for Xai capability"""

    name: str = "saliency_xaitk_app"
    saliency_generator: DeSerializablePlugfigurable[GenerateImageClassifierBlackboxSaliency] = Field(
        default_factory=_default_saliency_factory
    )
    img_batch_size: int = 1


class XaitkExplainableRun(CapabilityRunBase[XaitkExplainableConfig, XaitkExplainableOutputs]):
    """Run class for xai capability"""

    config: XaitkExplainableConfig
    outputs: XaitkExplainableOutputs

    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:  # noqa: ARG002
        """Access the in-depth data needed by Gradient to produce a report
        generated in the run method or in the load_cached_results method.
        Each slide will have a short text description of the model used,
        the prediction, the ground truth, and the saliency map for the
        detection. The saliency map image will have the detection bounding
        box shown in red.

        Parameters
        ----------
        threshold : float
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, where each dictionary represents a slide
            for the Gradient report.
        """

        outputs = self.outputs

        model_id = self.model_metadata[0]["id"]
        index2label = self.model_metadata[0]["index2label"]  # pyright: ignore[reportTypedDictNotRequiredAccess]

        gradient_slides = []

        # TODO: Needs general refactoring once a determination is made on if/why the saliency maps are
        # being generated for every class in the model for every image.
        # REF: https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/345
        i = -1
        for sal_maps, gray_img, gt_label in zip(outputs.results, outputs.gray_imgs, outputs.gt_labels, strict=False):
            i += 1

            conf_dict = self.config.model_dump()

            if "MCRISEStack" in conf_dict["saliency_generator"]["type"]:
                fill_colors = conf_dict["saliency_generator"][conf_dict["saliency_generator"]["type"]]["fill_colors"]
                for color_idx, color_value in enumerate(fill_colors):
                    for sal_idx in index2label:
                        fig = plt.figure()
                        plt.axis("off")
                        plt.imshow(gray_img, alpha=0.7, cmap="gray")
                        plt.xticks(())
                        plt.yticks(())
                        plt.imshow(np.asarray(sal_maps[color_idx][0]), cmap="seismic", alpha=0.3)
                        plt.colorbar()
                        fig.tight_layout()
                        plt.close(fig)

                        content = {
                            "deck": self.capability_id,
                            "layout_name": "OneImageText",
                            "layout_arguments": {
                                "title": f"**XAITK Saliency Map**: {sal_idx} \n",
                                "text": (
                                    f"Model: {model_id}\n"
                                    f"Image: {i}\n"
                                    f"Fill Color: {color_value}\n"
                                    f"GT: {gt_label}\n"
                                    f"Pred: {index2label[sal_idx]}"
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

                    plt.imshow(sal_map, cmap="seismic", alpha=0.3)
                    plt.colorbar()
                    fig.tight_layout()
                    plt.close(fig)

                    content = {
                        "deck": self.capability_id,
                        "layout_name": "OneImageText",
                        "layout_arguments": {
                            "title": f"**XAITK Saliency Map**: {sal_idx} \n",
                            "text": (
                                f"Model: {model_id}\n"
                                f"Image: {i}\n"
                                f"GT: {gt_label}\n"
                                f"Pred: {index2label[sal_idx]}"
                            ),
                            "image_path": Path(save_figure_to_tempfile(fig)),
                        },
                    }
                    content["layout_arguments"]["text"] = content["layout_arguments"]["text"].replace("_", r"\_")
                    gradient_slides.append(content)

        return gradient_slides


class XaitkExplainable(
    XaitkExplainableBase[XaitkExplainableOutputs, ic.Dataset, ic.Model, ic.Metric, XaitkExplainableConfig]
):
    """Xai will generate saliency maps for every detections in all images from the dataset."""

    _RUN_TYPE = XaitkExplainableRun

    @classmethod
    def _create_config(cls) -> XaitkExplainableConfig:
        return XaitkExplainableConfig()

    def _run(
        self,
        models: list[ic.Model],
        datasets: list[ic.Dataset],
        metrics: list[ic.Metric],
        config: XaitkExplainableConfig,
        use_prediction_and_evaluation_cache: bool,  # noqa: ARG002
    ) -> XaitkExplainableOutputs:
        """Run the capability, and store any outputs of the saliency generation in capability"""

        model = models[0]
        dataset = datasets[0]
        _ = metrics

        if "index2label" not in model.metadata:
            raise (KeyError("'index2label' not found in model metadata but is required by Xai"))

        classifier = JATICImageClassifier(
            classifier=model,
            ids=sorted(model.metadata["index2label"].keys()),
            img_batch_size=config.img_batch_size,
        )
        img_sal_maps = []
        gray_imgs = []
        gt_labels = []
        for ref_img, targets, _ in dataset:
            transposed_image = np.asarray(ref_img).transpose(1, 2, 0)
            if transposed_image.shape[2] == 1:
                transposed_image = np.concatenate((transposed_image,) * 3, axis=-1)
            img_sal_maps.append(config.saliency_generator(transposed_image, classifier))

            gray_imgs.append(rgb_to_grayscale(torch.as_tensor(ref_img)).squeeze(0).numpy())

            gt_labels.append(model.metadata["index2label"][int(np.argmax(targets))])

        return XaitkExplainableOutputs.model_validate(
            {"results": img_sal_maps, "gray_imgs": gray_imgs, "gt_labels": gt_labels}
        )
