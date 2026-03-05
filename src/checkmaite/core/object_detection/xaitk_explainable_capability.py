import logging
from pathlib import Path
from typing import Any

import maite.protocols.object_detection as od
import matplotlib.pyplot as plt
import numpy as np
from maite.protocols import DatasetMetadata
from matplotlib.patches import Rectangle
from PIL import Image
from pydantic import Field, model_validator
from smqtk_core.configuration import from_config_dict
from torch import Tensor, as_tensor
from xaitk_jatic.utils.sal_on_dets import sal_on_dets
from xaitk_saliency.interfaces.gen_object_detector_blackbox_sal import GenerateObjectDetectorBlackboxSaliency

from checkmaite.core._common.xaitk_explainable_capability import XaitkExplainableBase
from checkmaite.core._utils import deprecated, requires_optional_dependency
from checkmaite.core.capability_core import CapabilityConfigBase, CapabilityOutputsBase, CapabilityRunBase
from checkmaite.core.object_detection.dataset_loaders import DetectionTarget
from checkmaite.core.report._markdown import MarkdownOutput
from checkmaite.core.report._plotting_utils import save_figure_to_tempfile


class XaitkExplainableOutputDatum(CapabilityOutputsBase):
    """Xai outputs are stored as a list of per-datum outputs with these attributes"""

    img: np.ndarray
    img_id: int | str
    # These four properties correspond to 'N' detections each with a sal_map, box, label, and score.
    sal_maps: np.ndarray  # (H,W) pixel-wise saliency scores in [-1,1]
    boxes: np.ndarray  # [xmin,ymin,xmax,ymax]
    labels: np.ndarray
    scores: np.ndarray

    @model_validator(mode="after")
    def validate_lengths(self) -> "XaitkExplainableOutputDatum":
        """Validate that sal_maps, boxes, labels, and scores all have the same length."""
        lengths = [len(self.sal_maps), len(self.boxes), len(self.labels), len(self.scores)]
        if len(set(lengths)) > 1:
            raise ValueError("Lengths of sal_maps, boxes, labels, and scores must be equal")
        return self


class XaitkExplainableOutputs(CapabilityOutputsBase):
    """Each item in 'results' corresponds to one item in the input dataset"""

    results: list[XaitkExplainableOutputDatum]


def _default_saliency_factory() -> GenerateObjectDetectorBlackboxSaliency:
    return from_config_dict(
        {
            "type": "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack",
            "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack": {
                "n": 20,
                "s": 7,
                "p1": 0.7,
                "seed": 42,
                "threads": 8,
                "fill": [95, 96, 93],
            },
        },
        GenerateObjectDetectorBlackboxSaliency.get_impls(),
    )


class XaitkExplainableConfig(CapabilityConfigBase):
    """Config class for Xai capability."""

    name: str = "saliency_xai_app"
    saliency_generator: GenerateObjectDetectorBlackboxSaliency = Field(default_factory=_default_saliency_factory)
    img_batch_size: int = 1


class XaitkExplainableRun(CapabilityRunBase[XaitkExplainableConfig, XaitkExplainableOutputs]):
    """Run class for Xai for OD"""

    config: XaitkExplainableConfig
    outputs: XaitkExplainableOutputs

    # The order is important
    @requires_optional_dependency("gradient", install_hint="pip install '.[unsupported]'")
    @deprecated(replacement="collect_md_report")
    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:  # noqa: ARG002 # pragma: no cover
        """Access the in-depth data needed by Gradient to produce a report
        generated in the run method or in the load_cached_results method.
        Each slide will have a short text description of the model used,
        the image id, the prediction, the ground truth, and the saliency
        map for target class.

        Parameters
        ----------
        threshold
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
            A list of dictionaries, where each dictionary represents a slide
            for the Gradient report.
        """

        outputs = self.outputs

        model_id = self.model_metadata[0]["id"]
        index2label = self.model_metadata[0]["index2label"]  # pyright: ignore[reportTypedDictNotRequiredAccess]

        gradient_slides = []
        for idx, datum in enumerate(outputs.results):
            scores = datum.scores
            if len(datum.sal_maps) == 0:
                logging.warning(f"No detections found for image id {idx}")
                continue
            for sal_idx, bbox in enumerate(datum.boxes):
                sal_map = datum.sal_maps[sal_idx]
                # PIL throws error unless cast as uint8 - TypeError: Cannot handle this data type: (1, 1, 3), <i8
                ref_img_np = np.asarray(datum.img).astype(np.uint8)
                if ref_img_np.shape[0] == 1:
                    gray_img = np.asarray(Image.fromarray(ref_img_np[0]).convert("L"))
                else:
                    gray_img = np.asarray(Image.fromarray(ref_img_np.transpose(1, 2, 0)).convert("L"))

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
                plt.imshow(sal_map, cmap="seismic", alpha=0.3)
                cbar = plt.colorbar(location="bottom")
                cbar.set_label("Pixel relevance to detection")
                fig.tight_layout()
                plt.close(fig)

                content = {
                    "deck": self.capability_id,
                    "layout_name": "TwoItem",
                    "layout_arguments": {
                        "title": (f"Xai Saliency Map -- " f"Image ID: {datum.img_id}, " f"Detection: {sal_idx}"),
                        "left_item": Path(save_figure_to_tempfile(fig)),
                        "right_item": (
                            f"**Model:** {model_id}\n"
                            f"**Image ID**: {datum.img_id}\n"
                            f"**Prediction:** {index2label[int(datum.labels[sal_idx])]}\n"
                            f"**Confidence:** {scores[sal_idx]:.2f}\n\n\n"
                            f"Note: The Confidence is the metric score that the given detection had in the original "
                            f"(un-occluded) image.  Pixel relevance is normalized on scale from 0 to 1."
                        ),
                    },
                }
                gradient_slides.append(content)

        return gradient_slides

    def collect_md_report(self, threshold: float) -> str:  # noqa: ARG002
        """Generate Markdown report for XAITK saliency analysis.

        Parameters
        ----------
        threshold : float
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
        str
            Markdown-formatted report content.
        """
        outputs = self.outputs
        model_id = self.model_metadata[0]["id"]
        index2label = self.model_metadata[0]["index2label"]  # pyright: ignore[reportTypedDictNotRequiredAccess]

        md = MarkdownOutput("XAITK Saliency Maps - Object Detection")

        md.add_text(f"**Model**: {model_id}")
        md.add_text(
            "Saliency maps show which image regions most influenced each detection. "
            "Brighter areas indicate higher relevance. "
            "The confidence shown is the metric score from the original (un-occluded) image. "
            "Pixel relevance is normalized on scale from 0 to 1."
        )
        md.add_blank_line()

        for idx, datum in enumerate(outputs.results):
            scores = datum.scores
            if len(datum.sal_maps) == 0:
                logging.warning(f"No detections found for image id {idx}")
                continue

            for sal_idx, bbox in enumerate(datum.boxes):
                sal_map = datum.sal_maps[sal_idx]
                # PIL throws error unless cast as uint8
                ref_img_np = np.asarray(datum.img).astype(np.uint8)
                if ref_img_np.shape[0] == 1:
                    gray_img = np.asarray(Image.fromarray(ref_img_np[0]).convert("L"))
                else:
                    gray_img = np.asarray(Image.fromarray(ref_img_np.transpose(1, 2, 0)).convert("L"))

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
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none",
                    )
                )

                plt.imshow(sal_map, cmap="seismic", alpha=0.3)
                plt.colorbar()
                fig.tight_layout()

                img_path = save_figure_to_tempfile(fig=fig)

                plt.close(fig)

                md.add_section(heading=f"Detection {sal_idx + 1} - Image {datum.img_id}", level=3)
                md.add_text(f"**Model**: {model_id}")
                md.add_text(f"**Image ID**: {datum.img_id}")
                md.add_text(f"**Prediction**: {index2label[int(datum.labels[sal_idx])]}")
                md.add_text(f"**Confidence**: {scores[sal_idx]:.2f}")
                md.add_blank_line()
                md.add_image(img_path, alt_text=f"Saliency Map for Detection {sal_idx + 1}")

        return md.render()


class XaitkExplainable(
    XaitkExplainableBase[XaitkExplainableOutputs, od.Dataset, od.Model, od.Metric, XaitkExplainableConfig]
):
    """
    Xai will generate saliency maps for every detections in all images from the dataset.

    Attributes:
        config :The config used to run Xai saliency capability for OD.
    """

    _RUN_TYPE = XaitkExplainableRun

    @classmethod
    def _create_config(cls) -> XaitkExplainableConfig:
        return XaitkExplainableConfig()

    def _run(
        self,
        models: list[od.Model],
        datasets: list[od.Dataset],
        metrics: list[od.Metric],  # noqa: ARG002
        config: XaitkExplainableConfig,
        use_prediction_and_evaluation_cache: bool,  # noqa: ARG002
    ) -> XaitkExplainableOutputs:
        """Run the capability, and store any outputs of the saliency generation"""

        model = models[0]
        dataset = datasets[0]

        prediction_dataset = self.XaitkExplainableDetectionBaselineDataset(dataset, model)
        if "index2label" not in model.metadata:
            raise (KeyError("'index2label' not found in model metadata but is required by Xai"))
        all_dataset_sal_maps, _ = sal_on_dets(
            dataset=prediction_dataset,
            sal_generator=config.saliency_generator,
            detector=model,
            ids=sorted(model.metadata["index2label"].keys()),
            img_batch_size=config.img_batch_size,
        )

        return XaitkExplainableOutputs.model_validate(
            {
                "results": [
                    XaitkExplainableOutputDatum.model_validate(
                        {
                            "sal_maps": datum_sal_maps,
                            "img": prediction_dataset[i][0].numpy(),
                            "img_id": prediction_dataset[i][2]["id"],
                            "boxes": prediction_dataset[i][1].boxes,
                            "labels": prediction_dataset[i][1].labels,
                            "scores": prediction_dataset[i][1].scores,
                        }
                    )
                    for i, datum_sal_maps in enumerate(all_dataset_sal_maps)
                ]
            },
        )

    class XaitkExplainableDetectionBaselineDataset(od.Dataset):
        """
        A MAITE-compliant dataset wrapper class which is exclusive to the Xai for OD use case. The underlying
        assumption is that due to the nature of Xai and its OD saliency map algorithms, realistic use
        cases only involve small dataset and so re-creating it entirely in memory is feasible.

        Because `xaitk-saliency` evaluates the most salient locations with an image _for a given prediction_, the
        ground truth boxes and labels are not relevant.

        Since OD models may return many target boxes, this dataset will also save only the predictions where the model
        has the highest confidence.

        Parameters
        ----------
        dataset:
            A MAITE-compliant Object Detection dataset
        model:
            A MAITE-compliant Object Detection model
        dets_limit:
            The maximum number of detection targets that will be saved for an image, selected by highest confidence
            score.  Hardcoded to 10 for now, but could be made configurable in the future.

        Attributes
        ----------
        items:
            The whole dataset stored in memory.  The tuple structure is: image, boxes/labels/scores, datum metadata.
        """

        def __init__(self, dataset: od.Dataset, model: od.Model, dets_limit: int = 10) -> None:
            metadata_args: dict = {"id": f"xai_temp_{dataset.metadata['id']}"}
            if "index2label" in dataset.metadata:
                metadata_args["index2label"] = dataset.metadata["index2label"]
            self.metadata = DatasetMetadata(**metadata_args)
            self.items: list[tuple[Tensor, DetectionTarget, od.DatumMetadataType]] = self._construct_dataset(
                dataset, model, dets_limit
            )

        def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget, od.DatumMetadataType]:
            """Get `index`-th element from dataset."""
            return self.items[index]

        def __len__(self) -> int:
            """Return length of dataset."""
            return len(self.items)

        def _construct_dataset(
            self, dataset: od.Dataset, model: od.Model, dets_limit: int
        ) -> list[tuple[Tensor, DetectionTarget, od.DatumMetadataType]]:
            """Loads dataset in memory, replacing GT detection targets with predictions, limit 10 boxes per image."""
            items: list[tuple[Tensor, DetectionTarget, od.DatumMetadataType]] = []

            for datum in dataset:
                predictions = model([datum[0]])[0]

                # Sort predictions by scores in descending order and take the top `dets_limit`
                sorted_indices = np.argsort(np.asarray(predictions.scores))[::-1][:dets_limit].copy()
                top_boxes = np.asarray(predictions.boxes)[sorted_indices]
                top_labels = np.asarray(predictions.labels)[sorted_indices]
                top_scores = np.asarray(predictions.scores)[sorted_indices]

                detection_target = DetectionTarget(boxes=top_boxes, labels=top_labels, scores=top_scores)

                items.append((as_tensor(datum[0]), detection_target, datum[2]))

            return items
