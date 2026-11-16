from pathlib import Path
from typing import Any

import maite.protocols.image_classification as ic
import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import Field
from smqtk_core.configuration import from_config_dict
from torchvision.transforms.v2.functional import rgb_to_grayscale
from xaitk_saliency.interfaces.gen_image_classifier_blackbox_sal import GenerateImageClassifierBlackboxSaliency
from xaitk_saliency.interop.maite.image_classification import MAITEImageClassifier

from checkmaite.core._common.xaitk_explainable_capability import (
    SCORE_SUMMARY_HEADERS,
    XaitkExplainableBase,
    XaitkExplainableRecord,
    XaitkScoreSummary,
    saliency_entropy,
)
from checkmaite.core._utils import (
    CHECKMAITE_PLUGINS_UNSUPPORTED_INSTALL_HINT,
    deprecated,
    requires_optional_dependency,
)
from checkmaite.core.capability_core import CapabilityConfigBase, CapabilityOutputsBase, CapabilityRunBase
from checkmaite.core.report import InlineTextReport
from checkmaite.core.report._markdown import MarkdownOutput
from checkmaite.core.report._plotting_utils import save_figure_to_tempfile


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
    # Ground-truth label per image, or None when the target has no positive class.
    gt_labels: list[str | None]
    # Top-1 predicted label per image.
    pred_labels: list[str]
    # Label for each position on the class axis of a saliency array. The axis
    # follows the sorted keys of the model's index2label.
    class_labels: list[str]
    # Position on the class axis of the map that the scores describe: the
    # ground-truth class, or the predicted class when there is no ground truth.
    explained_class_indices: list[int]
    # Scores of the explained class's map, per image. The inner list has one
    # entry for RISE-style arrays and one per fill color for MC-RISE-style arrays.
    saliency_entropy: list[list[float | None]]
    normalized_saliency_entropy: list[list[float | None]]
    # Per-run summaries of the scores above, keyed by score name.
    score_summary: dict[str, XaitkScoreSummary]


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
    saliency_generator: GenerateImageClassifierBlackboxSaliency = Field(default_factory=_default_saliency_factory)
    img_batch_size: int = Field(
        default=1,
        description=(
            "Number of perturbed images per model call. xaitk-saliency 0.16.0's MAITEImageClassifier drops the "
            "last batch when it is smaller than this, so the saliency generator then fails with an IndexError. "
            "Use 1, or a value that divides the generator's mask count."
        ),
    )


def _explained_maps(sal_maps: np.ndarray, class_index: int) -> list[tuple[int | None, np.ndarray]]:
    """Return ``(fill_color_index, map)`` pairs for one class of one image.

    RISE-style arrays are ``(Cl, H, W)`` and give one pair with no fill color.
    MC-RISE-style arrays are ``(X, Cl, H, W)`` and give one pair per fill color.
    """
    arr = np.asarray(sal_maps)
    if arr.ndim == 4:
        return [(color_idx, arr[color_idx][class_index]) for color_idx in range(arr.shape[0])]
    if arr.ndim == 3:
        return [(None, arr[class_index])]
    raise ValueError(f"Unexpected saliency array shape {arr.shape}; expected (Cl, H, W) or (X, Cl, H, W)")


class XaitkExplainableRun(CapabilityRunBase[XaitkExplainableConfig, XaitkExplainableOutputs]):
    """Run class for xai capability"""

    config: XaitkExplainableConfig
    outputs: XaitkExplainableOutputs

    def extract(self) -> list[XaitkExplainableRecord]:
        """Extract per-image saliency statistics from this IC XaitkExplainable run.

        Emits one record per image, or one per (image, fill color) for
        MC-RISE-style ``(X, Cl, H, W)`` arrays. The statistics describe the
        map of the explained class: the ground-truth class, or the predicted
        class when the image has no ground truth.
        """
        # Single dataset/model (Number.ONE), no metrics (Number.ZERO)
        dataset_id = self.dataset_metadata[0]["id"]
        model_id = self.model_metadata[0]["id"]

        # Extract short generator class name, e.g. "RISEStack"
        conf_dict = self.config.model_dump()
        generator_type = conf_dict["saliency_generator"]["type"].split(".")[-1]

        outputs = self.outputs
        records: list[XaitkExplainableRecord] = []
        for image_index, sal_maps in enumerate(outputs.results):
            class_index = outputs.explained_class_indices[image_index]
            for map_idx, (fill_color_index, sal_map) in enumerate(_explained_maps(sal_maps, class_index)):
                flat = np.asarray(sal_map).ravel().astype(float)
                records.append(
                    XaitkExplainableRecord(
                        run_uid=self.run_uid,
                        dataset_id=dataset_id,
                        model_id=model_id,
                        saliency_generator_type=generator_type,
                        image_index=image_index,
                        gt_label=outputs.gt_labels[image_index],
                        explained_label=outputs.class_labels[class_index],
                        fill_color_index=fill_color_index,
                        predicted_label=outputs.pred_labels[image_index],
                        mean_saliency=float(np.mean(flat)),
                        max_saliency=float(np.max(flat)),
                        std_saliency=float(np.std(flat)),
                        positive_saliency_ratio=float(np.sum(flat > 0) / max(len(flat), 1)),
                        saliency_entropy=outputs.saliency_entropy[image_index][map_idx],
                        normalized_saliency_entropy=outputs.normalized_saliency_entropy[image_index][map_idx],
                    )
                )
        return records

    def _fill_color_names(self, n_colors: int) -> list[str]:
        """Names for the fill colors of an MC-RISE-style array."""
        conf_dict = self.config.model_dump()
        gen_type = conf_dict["saliency_generator"]["type"]
        fill_colors = conf_dict["saliency_generator"].get(gen_type, {}).get("fill_colors")
        if fill_colors is not None and len(fill_colors) == n_colors:
            return [str(c) for c in fill_colors]
        return [f"index {i}" for i in range(n_colors)]

    # The order is important
    @requires_optional_dependency("gradient", install_hint=CHECKMAITE_PLUGINS_UNSUPPORTED_INSTALL_HINT)
    @deprecated(replacement="collect_md_report")
    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:  # noqa: ARG002 # pragma: no cover
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
        class_labels = outputs.class_labels

        gradient_slides = []

        # TODO: Needs general refactoring once a determination is made on if/why the saliency maps are
        # being generated for every class in the model for every image.
        # REF: https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/345
        i = -1
        for sal_maps, gray_img, gt_label in zip(outputs.results, outputs.gray_imgs, outputs.gt_labels, strict=False):
            i += 1

            if np.asarray(sal_maps).ndim == 4:
                fill_colors = self._fill_color_names(len(sal_maps))
                for color_idx, color_value in enumerate(fill_colors):
                    for sal_idx, class_label in enumerate(class_labels):
                        fig = plt.figure()
                        plt.axis("off")
                        plt.imshow(gray_img, alpha=0.7, cmap="gray")
                        plt.xticks(())
                        plt.yticks(())
                        plt.imshow(np.asarray(sal_maps[color_idx][sal_idx]), cmap="seismic", alpha=0.3)
                        plt.colorbar()
                        fig.tight_layout()
                        plt.close(fig)

                        content = {
                            "deck": self.capability_id,
                            "layout_name": "OneImageText",
                            "layout_arguments": {
                                "title": f"**XAITK Saliency Map**: {class_label} \n",
                                "text": (
                                    f"Model: {model_id}\n"
                                    f"Image: {i}\n"
                                    f"Fill Color: {color_value}\n"
                                    f"GT: {gt_label}\n"
                                    f"Class explained: {class_label}"
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
                            "title": f"**XAITK Saliency Map**: {class_labels[sal_idx]} \n",
                            "text": (
                                f"Model: {model_id}\nImage: {i}\nGT: {gt_label}\n"
                                f"Class explained: {class_labels[sal_idx]}"
                            ),
                            "image_path": Path(save_figure_to_tempfile(fig)),
                        },
                    }
                    content["layout_arguments"]["text"] = content["layout_arguments"]["text"].replace("_", r"\_")
                    gradient_slides.append(content)

        return gradient_slides

    def collect_md_report(self, threshold: float) -> InlineTextReport:
        """Generate Markdown report for XAITK saliency analysis.

        Parameters
        ----------
        threshold : float
            Present for the shared report API; saliency maps have no threshold-based interpretation.

        Returns
        -------
        InlineTextReport
            Typed inline Markdown report.
        """
        _ = threshold  # TODO: Remove threshold as a keyword argument in a future MR.

        outputs = self.outputs
        model_id = self.model_metadata[0]["id"]
        class_labels = outputs.class_labels

        md = MarkdownOutput("XAITK Saliency Maps - Image Classification")

        md.add_text(f"**Model**: {model_id}")
        md.add_blank_line()

        md.add_section(heading="Saliency scores", level=2)
        md.add_text(
            "Scores describe the map of the explained class: the ground-truth class, or the predicted "
            "class when an image has no ground truth. Entropy is XAITK's `Entropy` metric in bits; "
            "the normalized value divides it by log2 of the pixel count, so it is in [0, 1]. "
            "Lower entropy means saliency is concentrated in fewer pixels."
        )
        md.add_table(
            headers=SCORE_SUMMARY_HEADERS,
            rows=[summary.as_row(name) for name, summary in outputs.score_summary.items()],
        )
        md.add_blank_line()

        for i, (sal_maps, gray_img, gt_label) in enumerate(
            zip(outputs.results, outputs.gray_imgs, outputs.gt_labels, strict=False)
        ):
            arr = np.asarray(sal_maps)
            if arr.ndim == 4:
                maps = [
                    (color_name, color_idx, class_idx, arr[color_idx][class_idx])
                    for color_idx, color_name in enumerate(self._fill_color_names(arr.shape[0]))
                    for class_idx in range(len(class_labels))
                ]
            else:
                maps = [(None, None, class_idx, arr[class_idx]) for class_idx in range(len(class_labels))]

            explained_idx = outputs.explained_class_indices[i]
            for color_name, color_idx, class_idx, sal_map in maps:
                class_label = class_labels[class_idx]
                fig = plt.figure()
                plt.axis("off")
                plt.imshow(gray_img, alpha=0.7, cmap="gray")
                plt.xticks(())
                plt.yticks(())
                plt.imshow(sal_map, cmap="seismic", alpha=0.3)
                plt.colorbar()
                fig.tight_layout()

                img_path = save_figure_to_tempfile(fig=fig)
                plt.close(fig)

                md.add_section(heading=f"Saliency Map: {class_label}", level=3)
                md.add_text(f"**Model**: {model_id}")
                md.add_text(f"**Image**: {i}")
                if color_name is not None:
                    md.add_text(f"**Fill Color**: {color_name}")
                md.add_text(f"**Ground Truth**: {gt_label}")
                md.add_text(f"**Prediction**: {outputs.pred_labels[i]}")
                md.add_text(f"**Class explained**: {class_label}")
                if class_idx == explained_idx:
                    map_idx = 0 if color_idx is None else color_idx
                    entropy = outputs.saliency_entropy[i][map_idx]
                    md.add_text(f"**Entropy**: {'n/a' if entropy is None else f'{entropy:.3f} bits'}")
                md.add_embedded_image(img_path, alt_text=f"Saliency Map {class_label}", remove_source=True)

        md.add_provenance(self)

        return InlineTextReport(
            media_type="text/markdown",
            content=md.render(),
            filename=f"{self.capability_id}.md",
        )


class XaitkExplainable(
    XaitkExplainableBase[XaitkExplainableOutputs, ic.Dataset, ic.Model, ic.Metric, XaitkExplainableConfig]
):
    """Xai will generate saliency maps for every detections in all images from the dataset."""

    task = "image_classification"

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

        index2label = model.metadata["index2label"]
        # MAITEImageClassifier sorts these ids, and the class axis of every
        # saliency array (and of model outputs and targets) follows that order.
        class_ids = sorted(index2label.keys())
        class_labels = [str(index2label[class_id]) for class_id in class_ids]

        classifier = MAITEImageClassifier(
            classifier=model,
            ids=class_ids,
            img_batch_size=config.img_batch_size,
        )
        img_sal_maps = []
        gray_imgs = []
        gt_labels: list[str | None] = []
        pred_labels: list[str] = []
        explained_class_indices: list[int] = []
        entropies: list[list[float | None]] = []
        normalized_entropies: list[list[float | None]] = []
        for ref_img, targets, _ in dataset:
            transposed_image = np.asarray(ref_img).transpose(1, 2, 0)
            if transposed_image.shape[2] == 1:
                transposed_image = np.concatenate((transposed_image,) * 3, axis=-1)
            # Checked before the saliency call, where xaitk-saliency would raise a bare IndexError.
            pred_vec = np.asarray(model([ref_img])[0]).ravel()
            if pred_vec.size > len(class_ids):
                raise ValueError(
                    f"Model output has {pred_vec.size} classes but the model's index2label has {len(class_ids)}"
                )
            pred_index = int(np.argmax(pred_vec))
            sal_maps = config.saliency_generator(ref_image=transposed_image, blackbox=classifier)
            img_sal_maps.append(sal_maps)

            gray_imgs.append(rgb_to_grayscale(torch.as_tensor(ref_img)).squeeze(0).numpy())

            pred_labels.append(class_labels[pred_index])

            # Targets are class-axis vectors, so argmax is a position, not an index2label key.
            target_vec = np.asarray(targets, dtype=float).ravel()
            if target_vec.size > len(class_ids):
                raise ValueError(
                    f"Target has {target_vec.size} classes but the model's index2label has {len(class_ids)}"
                )
            if target_vec.size > 0 and target_vec.max() > 0:
                gt_index = int(np.argmax(target_vec))
                gt_labels.append(class_labels[gt_index])
                explained_class_indices.append(gt_index)
            else:
                gt_labels.append(None)
                explained_class_indices.append(pred_index)

            scores = [saliency_entropy(m) for _, m in _explained_maps(sal_maps, explained_class_indices[-1])]
            entropies.append([raw for raw, _ in scores])
            normalized_entropies.append([norm for _, norm in scores])

        score_summary = {
            "saliency_entropy": XaitkScoreSummary.from_values(v for row in entropies for v in row),
            "normalized_saliency_entropy": XaitkScoreSummary.from_values(
                v for row in normalized_entropies for v in row
            ),
        }

        return XaitkExplainableOutputs.model_validate(
            {
                "results": img_sal_maps,
                "gray_imgs": gray_imgs,
                "gt_labels": gt_labels,
                "pred_labels": pred_labels,
                "class_labels": class_labels,
                "explained_class_indices": explained_class_indices,
                "saliency_entropy": entropies,
                "normalized_saliency_entropy": normalized_entropies,
                "score_summary": score_summary,
            }
        )
