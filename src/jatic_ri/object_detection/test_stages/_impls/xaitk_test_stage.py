"""XAITKTestStage implementation"""

# Python generic imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import maite.protocols.object_detection as od
import matplotlib.pyplot as plt
import numpy as np
from maite.protocols import DatasetMetadata
from matplotlib.patches import Rectangle
from PIL import Image
from pydantic import model_validator
from torch import Tensor, as_tensor
from xaitk_jatic.utils.sal_on_dets import sal_on_dets
from xaitk_saliency.interfaces.gen_object_detector_blackbox_sal import GenerateObjectDetectorBlackboxSaliency

from jatic_ri._common.test_stages.impls.xaitk_test_stage import XAITKTestStageBase
from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, OutputsBase, RunBase
from jatic_ri.object_detection.datasets import DetectionTarget
from jatic_ri.util._types import DeSerializablePlugfigurable
from jatic_ri.util.utils import save_figure_to_tempfile


class XAITKOutputDatumOD(OutputsBase):
    """XAITK outputs are stored as a list of per-datum outputs with these attributes"""

    img: np.ndarray
    img_id: int | str
    # These four properties correspond to 'N' detections each with a sal_map, box, label, and score.
    sal_maps: np.ndarray  # (H,W) pixel-wise saliency scores in [-1,1]
    boxes: np.ndarray  # [xmin,ymin,xmax,ymax]
    labels: np.ndarray
    scores: np.ndarray

    @model_validator(mode="after")
    def validate_lengths(self) -> XAITKOutputDatumOD:
        """Validate that sal_maps, boxes, labels, and scores all have the same length."""
        lengths = [len(self.sal_maps), len(self.boxes), len(self.labels), len(self.scores)]
        if len(set(lengths)) > 1:
            raise ValueError("Lengths of sal_maps, boxes, labels, and scores must be equal")
        return self


class XAITKOutputsOD(OutputsBase):
    """Each item in 'results' corresponds to one item in the input dataset"""

    results: list[XAITKOutputDatumOD]


class XAITKConfigOD(ConfigBase):
    """Config class for XAITKTestStage for OD"""

    name: str
    saliency_generator: DeSerializablePlugfigurable[GenerateObjectDetectorBlackboxSaliency]
    img_batch_size: int


class XAITKRunOD(RunBase):
    """Run class for XAITKTestStage for OD"""

    config: XAITKConfigOD
    outputs: XAITKOutputsOD


class XAITKTestStage(XAITKTestStageBase[XAITKOutputsOD, od.Dataset, od.Model, od.Metric]):
    """
    XAITKTestStage will generate saliency maps for every detections in all images from the dataset.

    Attributes:
        config (XAITKConfigOD):The config used to run XAITK saliency test stage for OD.
    """

    _task: str = "od"
    _RUN_TYPE = XAITKRunOD

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config: XAITKConfigOD = XAITKConfigOD.model_validate(config)

    def _create_config(self) -> XAITKConfigOD:
        return self.config

    def _run(
        self,
        models: list[od.Model],
        datasets: list[od.Dataset],
        metrics: list[od.Metric],  # noqa: ARG002
    ) -> XAITKOutputsOD:
        """Run the test stage, and store any outputs of the saliency generation in test stage"""

        model = models[0]
        dataset = datasets[0]

        prediction_dataset = self.XAITKDetectionBaselineDataset(dataset, model)
        if "index2label" not in model.metadata:
            raise (KeyError("'index2label' not found in model metadata but is required by XAITKTestStage"))
        all_dataset_sal_maps, _ = sal_on_dets(
            dataset=prediction_dataset,
            sal_generator=self.config.saliency_generator,
            detector=model,
            ids=sorted(model.metadata["index2label"].keys()),
            img_batch_size=self.config.img_batch_size,
        )

        return XAITKOutputsOD.model_validate(
            {
                "results": [
                    XAITKOutputDatumOD.model_validate(
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

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report
        generated in the run method or in the load_cached_results method.
        Each slide will have a short text description of the model used,
        the image id, the prediction, the ground truth, and the saliency
        map for target class."""

        if self._stored_run is None:
            raise RuntimeError("TestStage must be run before accessing outputs")
        outputs = self._stored_run.outputs

        model_id = self._stored_run.model_metadata[0]["id"]
        index2label = self._stored_run.model_metadata[0]["index2label"]  # pyright: ignore[reportTypedDictNotRequiredAccess]

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
                    "deck": "object_detection_dataset_evaluation",
                    "layout_name": "TwoItem",
                    "layout_arguments": {
                        "title": (f"XAITK Saliency Map -- " f"Image ID: {datum.img_id}, " f"Detection: {sal_idx}"),
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

    class XAITKDetectionBaselineDataset(od.Dataset):
        """
        A MAITE-compliant dataset wrapper class which is exclusive to the XAITK for OD use case. The underlying
        assumption is that due to the nature of XAITK and its OD saliency map algorithms, realistic use
        cases only involve small dataset and so re-creating it entirely in memory is feasible.

        Because `xaitk-saliency` evaluates the most salient locations with an image _for a given prediction_, the
        ground truth boxes and labels are not relevant.

        Since OD models may return many target boxes, this dataset will also save only the predictions where the model
        has the highest confidence.

        Parameters
        ----------
        dataset (od.Dataset):
            A MAITE-compliant Object Detection dataset
        model (od.Model):
            A MAITE-compliant Object Detection model
        dets_limit (int):
            The maximum number of detection targets that will be saved for an image, selected by highest confidence
            score.  Hardcoded to 10 for now, but could be made configurable in the future.

        Attributes
        ----------
        items (List[Tuple[Tensor, DetectionTarget, Dict[str, Any]]]):
            The whole dataset stored in memory.  The tuple structure is: image, boxes/labels/scores, datum metadata.

        Methods
        -------
        __getitem__(self, index: int) -> Tuple[Tensor, DetectionTarget, Dict[str, Any]]
            Provide mapping-style access to dataset elements. Returned tuple elements
            correspond to input type, target type, and datum-specific metadata,
            respectively.

        __len__(self) -> int
            Return the number of data elements in the dataset.
        """

        def __init__(self, dataset: od.Dataset, model: od.Model, dets_limit: int = 10) -> None:
            metadata_args: dict = {"id": f"xaitk_temp_{dataset.metadata['id']}"}
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
