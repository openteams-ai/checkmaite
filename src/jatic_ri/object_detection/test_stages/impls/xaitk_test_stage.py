"""XAITKTestStage implementation"""

# Python generic imports
from __future__ import annotations

import json
import logging
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import maite.protocols.object_detection as od

# 3rd Party Imports
import matplotlib.pyplot as plt
import numpy as np
from maite.protocols import DatasetMetadata
from matplotlib.patches import Rectangle
from PIL import Image

# SMQTK imports
from smqtk_core.configuration import from_config_dict
from torch import Tensor, as_tensor
from xaitk_jatic.utils.sal_on_dets import sal_on_dets

# XAITK imports
from xaitk_saliency.interfaces.gen_object_detector_blackbox_sal import GenerateObjectDetectorBlackboxSaliency

from jatic_ri._common.test_stages.impls.xaitk_test_stage import XAITKTestStageBase
from jatic_ri.object_detection.datasets import DetectionTarget
from jatic_ri.util.cache import TensorEncoder


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
        self.prediction_dataset = self.XAITKDetectionBaselineDataset(self.dataset, self.model)
        if "index2label" not in self.model.metadata:
            raise (KeyError("'index2label' not found in model metadata but is required by XAITKTestStage"))
        img_sal_maps, _ = sal_on_dets(
            dataset=self.prediction_dataset,
            sal_generator=self.sal_generator,
            detector=self.model,
            ids=sorted(self.model.metadata["index2label"].keys()),
            img_batch_size=self.img_batch_size,
        )

        return {
            "img" + str(i): {
                "sal_map": TensorEncoder().default(sal_map),
                "img": TensorEncoder().default(self.prediction_dataset[i][0].numpy()),
                "boxes": TensorEncoder().default(self.prediction_dataset[i][1].boxes),
                "labels": TensorEncoder().default(self.prediction_dataset[i][1].labels),
                "scores": TensorEncoder().default(self.prediction_dataset[i][1].scores),
            }
            for i, sal_map in enumerate(img_sal_maps)
        }

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Access the in-depth data needed by Gradient to produce a report
        generated in the run method or in the load_cached_results method.
        Each slide will have a short text description of the model used,
        the image id, the prediction, the ground truth, and the saliency
        map for target class."""

        gradient_slides = []
        output_values = list(self.outputs.values())

        for idx, datum in enumerate(output_values):
            datum = output_values[idx]
            sub_dir = os.path.join(os.path.splitext(self.cache_path)[0], f"img_{idx}")

            os.makedirs(sub_dir, exist_ok=True)
            labels = np.asarray(datum["labels"])
            bboxes = np.asarray(datum["boxes"])
            scores = np.asarray(datum["scores"])
            sal_maps = datum["sal_map"]
            if len(sal_maps) == 0:
                logging.warning(f"No detections found for image id {idx}")
                continue
            for sal_idx, bbox in enumerate(bboxes):
                sal_map = sal_maps[sal_idx]
                # PIL throws error unless cast as uint8 - TypeError: Cannot handle this data type: (1, 1, 3), <i8
                ref_img_np = np.asarray(datum["img"]).astype(np.uint8)
                if ref_img_np.shape[0] == 1:
                    gray_img = np.asarray(Image.fromarray(ref_img_np[0]).convert("L"))
                else:
                    gray_img = np.asarray(Image.fromarray(ref_img_np.transpose(1, 2, 0)).convert("L"))

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
                            f"Model: {self.model_id}\nImage: {idx}\n"
                            f"Pred: {self.model.metadata['index2label'][int(labels[sal_idx])]}\n"  # type: ignore
                            f"Score: {float(scores[sal_idx]):.2f}\n"
                        ),
                        "image_path": sal_map_path,
                    },
                }
                content["layout_arguments"]["text"] = content["layout_arguments"]["text"].replace("_", r"\_")
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
