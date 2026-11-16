import logging
import math
from pathlib import Path
from typing import Any

import maite.protocols.object_detection as od
import matplotlib.pyplot as plt
import numpy as np
import torch
from maite.protocols import DatasetMetadata
from matplotlib.patches import Rectangle
from PIL import Image
from pydantic import Field, model_validator
from smqtk_core.configuration import from_config_dict
from torchvision.ops import box_iou
from xaitk_saliency.interfaces.gen_object_detector_blackbox_sal import GenerateObjectDetectorBlackboxSaliency
from xaitk_saliency.interop.maite.utils import sal_on_dets
from xaitk_saliency.utils.sal_metrics import compute_saliency_coverage

from checkmaite.core._common.detection_target import DetectionTarget
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


class XaitkExplainableOutputDatum(CapabilityOutputsBase):
    """Xai outputs are stored as a list of per-datum outputs with these attributes"""

    img: np.ndarray
    img_id: int | str
    # These four properties correspond to 'N' detections each with a sal_map, box, label, and score.
    sal_maps: np.ndarray  # (H,W) pixel-wise saliency scores in [-1,1]
    boxes: np.ndarray  # [xmin,ymin,xmax,ymax]
    labels: np.ndarray
    scores: np.ndarray
    # Per-detection saliency scores. NaN means the score is undefined.
    saliency_entropy: np.ndarray
    normalized_saliency_entropy: np.ndarray
    # Ground-truth matching, per detection. -1 / NaN means no match.
    matched_gt_index: np.ndarray
    matched_gt_iou: np.ndarray
    gt_saliency_coverage: np.ndarray
    # 1.0 when the map's peak is inside the matched box, 0.0 when it is not, NaN when there is no score.
    gt_saliency_peak_in_box: np.ndarray
    # Distance in pixels from the map's peak to the matched box; 0 inside it, NaN when there is no score.
    gt_saliency_peak_distance: np.ndarray
    # Ground-truth boxes for this image, [xmin,ymin,xmax,ymax].
    gt_boxes: np.ndarray
    # Nominal factor the image was resized by before the run (``saliency_max_side``); 1.0 when it was not.
    # Each axis's actual ratio can differ from it by rounding, by under a pixel. Every array above is at the
    # resized size; the ground-truth boxes are scaled by each axis's actual ratio.
    image_scale: float = 1.0

    @model_validator(mode="after")
    def validate_lengths(self) -> "XaitkExplainableOutputDatum":
        """Validate that all per-detection arrays have the same length."""
        lengths = [
            len(self.sal_maps),
            len(self.boxes),
            len(self.labels),
            len(self.scores),
            len(self.saliency_entropy),
            len(self.normalized_saliency_entropy),
            len(self.matched_gt_index),
            len(self.matched_gt_iou),
            len(self.gt_saliency_coverage),
            len(self.gt_saliency_peak_in_box),
            len(self.gt_saliency_peak_distance),
        ]
        if len(set(lengths)) > 1:
            raise ValueError(
                "Per-detection arrays (sal_maps, boxes, labels, scores, score arrays) must have equal lengths"
            )
        return self


class XaitkExplainableOutputs(CapabilityOutputsBase):
    """Each item in 'results' corresponds to one item in the input dataset"""

    results: list[XaitkExplainableOutputDatum]
    # Per-run summaries of the per-detection scores, keyed by score name.
    score_summary: dict[str, XaitkScoreSummary]


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
    gt_match_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """A detection is matched to the ground-truth box it overlaps most, if that IoU is at least this value."""
    min_score: float | None = None
    """Only explain detections with at least this score. ``None`` explains the top detections whatever their score."""
    dets_limit: int = Field(default=10, ge=1)
    """The most detections explained per image, highest score first.

    The model runs once per perturbed image whatever this is, so it mostly sets how many maps are stored and reported.
    """
    saliency_max_side: int | None = Field(default=None, ge=1)
    """Longest image side, in pixels, to run at. ``None`` keeps every image at full size.

    A larger image is downscaled, keeping its aspect ratio, before the model sees it. Prediction,
    saliency, ground-truth matching and every output (image, boxes, ground truth, maps) are then at that
    size, and each result's ``image_scale`` records the nominal factor (rounding the size can move each axis's
    actual ratio by under a pixel; ground-truth boxes use the actual ratios). Saliency memory and the stored maps shrink
    with the square of the factor. Set it to the model's own input size (for example 640 for a YOLO model
    run at ``imgsz=640``): a model that resizes its input anyway loses little, while a 1700 px image at 640
    needs about a seventh of the memory.
    """


def _top_detections(
    predictions: od.TargetType, class_ids: list[int], min_score: float | None, dets_limit: int
) -> DetectionTarget:
    """The detections to explain for one image: highest score first, at most ``dets_limit``, at least ``min_score``."""
    scores = np.asarray(predictions.scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError(
            f"XaitkExplainable needs one score per detection, shape (N,), but got scores of shape {scores.shape}. "
            "Per-class scores of shape (N, n_classes) are not supported."
        )
    order = np.argsort(scores)[::-1]
    if min_score is not None:
        order = order[scores[order] >= min_score]
    order = order[:dets_limit]
    labels = np.asarray(predictions.labels)[order].astype(int)
    unknown = sorted({int(lbl) for lbl in labels} - set(class_ids))
    if unknown:
        raise ValueError(f"Model predicted label ids {unknown} that are not in its index2label")
    return DetectionTarget(
        boxes=np.asarray(predictions.boxes, dtype=float).reshape(-1, 4)[order],
        labels=labels,
        scores=scores[order],
    )


class _SaliencyTargetDataset(od.Dataset):
    """One already-loaded image, with its detections as the targets ``sal_on_dets`` explains.

    ``_run`` loads each image once and uses that copy for prediction, saliency
    and the outputs, so a dataset that returns something different on each
    read (random augmentation, for example) cannot mix samples.

    Labels are returned as positions in ``class_ids`` instead of ids. This
    works around a bug in xaitk-saliency 0.16.0
    (``xaitk_saliency.interop.maite.utils._sal_on_dets.compute_sal_maps``): it
    builds its reference score matrix with ``score_matrix[det][label]`` and
    ``len(ids)`` columns, while the detector adapter orders its columns by the
    sorted ids, so the two only agree when the ids are ``0..N-1``. Return the
    ids instead once ``compute_sal_maps`` maps labels to the position of their
    id in ``ids`` itself.

    The image must have at least one detection: the xaitk-saliency 0.16.0
    object detection generators raise ``ValueError`` for an image with none.
    """

    def __init__(
        self,
        image: np.ndarray,
        detections: DetectionTarget,
        datum_metadata: od.DatumMetadataType,
        class_ids: list[int],
        dataset_id: str,
    ) -> None:
        if len(detections.labels) == 0:
            raise ValueError("An image passed to sal_on_dets needs at least one detection")
        positions = {class_id: pos for pos, class_id in enumerate(class_ids)}
        self.metadata = DatasetMetadata(id=f"xai_temp_{dataset_id}")
        self._image = image
        self._target = DetectionTarget(
            boxes=detections.boxes,
            labels=np.asarray([positions[int(lbl)] for lbl in detections.labels], dtype=int),
            scores=detections.scores,
        )
        self._datum_metadata = datum_metadata

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[od.InputType, od.TargetType, od.DatumMetadataType]:
        if index != 0:
            raise IndexError(index)
        return self._image, self._target, self._datum_metadata


def _maps_or_zeros(maps: np.ndarray, dets: DetectionTarget, image: np.ndarray, image_index: int) -> np.ndarray:
    """Return one map per detection, or all-zero maps when the generator gave none.

    The xaitk-saliency 0.16.0 object detection generators return an empty
    array, not one map per detection, when the model detects nothing on any
    perturbed image. There is then no evidence for any pixel, so each
    detection gets an all-zero map, which has no entropy or coverage score.
    """
    maps = np.asarray(maps)
    if maps.ndim == 3 and len(maps) == len(dets.labels):
        return maps
    logging.warning(
        "The saliency generator returned no maps for image %s: the model detected nothing on any perturbed image. "
        "Its detections get all-zero maps.",
        image_index,
    )
    height, width = image.shape[-2:]
    return np.zeros((len(dets.labels), height, width))


def _resize_to_max_side(image: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float, float, float]:
    """Downscale a ``(C, H, W)`` image so its longest side is at most ``max_side``.

    Returns the image, in its original dtype, the nominal scale factor, and the
    actual ``(y, x)`` ratios ``new_h / H`` and ``new_w / W``. The size is
    rounded per axis, so the actual ratios can differ from the nominal factor.
    All three are 1.0 when the image is not resized.
    """
    height, width = image.shape[-2:]
    if max_side is None or max(height, width) <= max_side:
        return image, 1.0, 1.0, 1.0
    scale = max_side / max(height, width)
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    # torch cannot wrap arrays with negative strides, such as flipped views.
    resized = torch.nn.functional.interpolate(
        torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32)[None],
        size=size,
        mode="bilinear",
        antialias=True,
    )[0].numpy()
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        resized = np.clip(np.rint(resized), info.min, info.max)
    return resized.astype(image.dtype), scale, size[0] / height, size[1] / width


def _box_pixel_bounds(box: np.ndarray, height: int, width: int) -> tuple[int, int, int, int]:
    """Pixel bounds ``(x0, y0, x1, y1)``, end-exclusive, of ``[xmin, ymin, xmax, ymax]`` clipped to the image."""
    x0 = min(max(math.floor(float(box[0])), 0), width)
    y0 = min(max(math.floor(float(box[1])), 0), height)
    x1 = min(max(math.ceil(float(box[2])), 0), width)
    y1 = min(max(math.ceil(float(box[3])), 0), height)
    return x0, y0, x1, y1


def _box_mask(box: np.ndarray, height: int, width: int) -> np.ndarray:
    """Binary (H, W) mask of the pixels covered by ``[xmin, ymin, xmax, ymax]``."""
    x0, y0, x1, y1 = _box_pixel_bounds(box, height, width)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def _distance_to_box_mask(y: int, x: int, box: np.ndarray, height: int, width: int) -> float:
    """Euclidean distance in pixels from pixel ``(y, x)`` to the nearest pixel of ``_box_mask(box, ...)``.

    0 when the pixel is in the mask. A box that covers no pixel gives a positive
    distance to where it would be, so 0 still means "inside".
    """
    x0, y0, x1, y1 = _box_pixel_bounds(box, height, width)
    dx = max(x0 - x, 0, x - (x1 - 1))
    dy = max(y0 - y, 0, y - (y1 - 1))
    return float(math.hypot(dx, dy))


def _gt_scores(
    sal_maps: np.ndarray,
    det_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match each detection to a ground-truth box and score saliency inside it.

    Matching is class-agnostic and many-to-one: each detection takes the
    ground-truth box with the highest IoU, if that IoU is at least
    ``iou_threshold``.

    Coverage is XAITK's ``compute_saliency_coverage`` on the map's
    above-average saliency (the map minus its mean, clipped at 0), with the
    matched box as the mask: the share of above-average saliency that lies
    inside the box. The mean is removed because D-RISE maps average the
    detection score over the masks that leave each pixel visible, so every
    pixel sits near the average score and only the excess shows what the
    detection relied on. Without it, coverage mostly measures the box's area.

    Peak-in-box is the pointing game: 1.0 when the map's highest pixel is inside
    the matched box, 0.0 when it is not. Peak distance is the Euclidean distance
    in pixels from that pixel to the nearest pixel the box covers (the same
    pixel bounds as the mask), so it is 0 exactly when peak-in-box is 1.0.

    Returns ``(matched_gt_index, matched_gt_iou, gt_saliency_coverage,
    gt_saliency_peak_in_box, gt_saliency_peak_distance)``; -1 and NaN mark
    detections with no match or no above-average saliency.
    """
    n_dets = len(det_boxes)
    matched_idx = np.full(n_dets, -1, dtype=int)
    matched_iou = np.full(n_dets, np.nan, dtype=float)
    coverage = np.full(n_dets, np.nan, dtype=float)
    peak_in_box = np.full(n_dets, np.nan, dtype=float)
    peak_distance = np.full(n_dets, np.nan, dtype=float)
    if n_dets == 0 or len(gt_boxes) == 0:
        return matched_idx, matched_iou, coverage, peak_in_box, peak_distance

    ious = box_iou(
        torch.as_tensor(np.asarray(det_boxes, dtype=float)),
        torch.as_tensor(np.asarray(gt_boxes, dtype=float)),
    ).numpy()
    for det_idx in range(n_dets):
        best = int(np.argmax(ious[det_idx]))
        if ious[det_idx, best] < iou_threshold:
            continue
        matched_idx[det_idx] = best
        matched_iou[det_idx] = float(ious[det_idx, best])
        sal_map = np.asarray(sal_maps[det_idx], dtype=float)
        # D-RISE gives an all-NaN map for a zero-confidence detection; it is no evidence for any pixel.
        if not np.isfinite(sal_map).all():
            continue
        above_average = np.clip(sal_map - sal_map.mean(), 0.0, None)
        # Floating-point noise on a flat map is not evidence.
        above_average[above_average <= 1e-9 * np.abs(sal_map).max()] = 0.0
        if above_average.sum() <= 0:
            continue
        mask = _box_mask(np.asarray(gt_boxes[best]), *above_average.shape)
        coverage[det_idx] = float(
            compute_saliency_coverage(saliency_features=above_average, ground_truth_features=mask)
        )
        peak_y, peak_x = np.unravel_index(int(np.argmax(sal_map)), sal_map.shape)
        peak_in_box[det_idx] = float(mask[peak_y, peak_x])
        peak_distance[det_idx] = _distance_to_box_mask(
            int(peak_y), int(peak_x), np.asarray(gt_boxes[best]), *above_average.shape
        )
    return matched_idx, matched_iou, coverage, peak_in_box, peak_distance


def _none_if_nan(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


class XaitkExplainableRun(CapabilityRunBase[XaitkExplainableConfig, XaitkExplainableOutputs]):
    """Run class for Xai for OD"""

    config: XaitkExplainableConfig
    outputs: XaitkExplainableOutputs

    def extract(self) -> list[XaitkExplainableRecord]:
        """Extract per-detection saliency statistics from this OD XaitkExplainable run.

        Emits one record per detection. Each detection has a saliency map,
        bounding box, predicted label, and confidence score, plus its saliency
        entropy and, when it matches a ground-truth box, the IoU and saliency
        coverage of that box. Images with no detections are skipped.
        """
        # Single dataset/model (Number.ONE), no metrics (Number.ZERO)
        dataset_id = self.dataset_metadata[0]["id"]
        model_id = self.model_metadata[0]["id"]
        index2label = self.model_metadata[0]["index2label"]  # pyright: ignore[reportTypedDictNotRequiredAccess]

        # Extract short generator class name, e.g. "DRISEStack"
        conf_dict = self.config.model_dump()
        generator_type = conf_dict["saliency_generator"]["type"].split(".")[-1]

        records: list[XaitkExplainableRecord] = []
        for image_index, datum in enumerate(self.outputs.results):
            if len(datum.sal_maps) == 0:
                continue

            for det_idx in range(len(datum.sal_maps)):
                sal_map = np.asarray(datum.sal_maps[det_idx]).ravel().astype(float)

                records.append(
                    XaitkExplainableRecord(
                        run_uid=self.run_uid,
                        dataset_id=dataset_id,
                        model_id=model_id,
                        saliency_generator_type=generator_type,
                        image_index=image_index,
                        image_id=str(datum.img_id),
                        detection_index=det_idx,
                        predicted_label=index2label[int(datum.labels[det_idx])],
                        confidence=float(datum.scores[det_idx]),
                        mean_saliency=float(np.mean(sal_map)),
                        max_saliency=float(np.max(sal_map)),
                        std_saliency=float(np.std(sal_map)),
                        positive_saliency_ratio=float(np.sum(sal_map > 0) / max(len(sal_map), 1)),
                        saliency_entropy=_none_if_nan(datum.saliency_entropy[det_idx]),
                        normalized_saliency_entropy=_none_if_nan(datum.normalized_saliency_entropy[det_idx]),
                        matched_gt_index=(
                            int(datum.matched_gt_index[det_idx]) if datum.matched_gt_index[det_idx] >= 0 else None
                        ),
                        matched_gt_iou=_none_if_nan(datum.matched_gt_iou[det_idx]),
                        gt_saliency_coverage=_none_if_nan(datum.gt_saliency_coverage[det_idx]),
                        gt_saliency_peak_in_box=(
                            None
                            if math.isnan(datum.gt_saliency_peak_in_box[det_idx])
                            else bool(datum.gt_saliency_peak_in_box[det_idx])
                        ),
                        gt_saliency_peak_distance=_none_if_nan(datum.gt_saliency_peak_distance[det_idx]),
                    )
                )
        return records

    # The order is important
    @requires_optional_dependency("gradient", install_hint=CHECKMAITE_PLUGINS_UNSUPPORTED_INSTALL_HINT)
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
                        "title": (f"Xai Saliency Map -- Image ID: {datum.img_id}, Detection: {sal_idx}"),
                        "left_item": Path(save_figure_to_tempfile(fig)),
                        "right_item": (
                            f"**Model:** {model_id}\n"
                            f"**Image ID**: {datum.img_id}\n"
                            f"**Prediction:** {index2label[int(datum.labels[sal_idx])]}\n"
                            f"**Confidence:** {scores[sal_idx]:.2f}\n\n\n"
                            f"Note: The Confidence is the metric score that the given detection had in the original "
                            f"(un-occluded) image. Red marks the pixels that most supported the detection and blue "
                            f"the least; each map is scaled so its largest absolute value is 1."
                        ),
                    },
                }
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
        index2label = self.model_metadata[0]["index2label"]  # pyright: ignore[reportTypedDictNotRequiredAccess]

        md = MarkdownOutput("XAITK Saliency Maps - Object Detection")

        md.add_text(f"**Model**: {model_id}")
        md.add_text(
            "Saliency maps show which image regions most influenced each detection. "
            "Red marks the pixels that most supported the detection and blue the least; each map is scaled so "
            "its largest absolute value is 1. "
            "The confidence shown is the metric score from the original (un-occluded) image."
        )
        min_score = self.config.min_score
        md.add_text(
            f"Up to {self.config.dets_limit} detections are explained per image, highest score first"
            + ("." if min_score is None else f", if their score is at least {min_score}.")
        )
        if self.config.saliency_max_side is not None:
            md.add_text(
                f"Images were downscaled so their longest side is at most {self.config.saliency_max_side} px; "
                "boxes and maps are at that size."
            )
        md.add_blank_line()

        md.add_section(heading="Saliency scores", level=2)
        md.add_text(
            "Entropy is XAITK's `Entropy` metric in bits; the normalized value divides it by log2 of the "
            "pixel count, so it is in [0, 1]. Lower entropy means saliency is concentrated in fewer pixels. "
            "Each detection is matched to the ground-truth box it overlaps most, if the IoU is at least "
            f"{self.config.gt_match_iou_threshold} (class-agnostic). GT saliency coverage is the share of the "
            "map's above-average saliency (the map minus its mean) inside that box. GT saliency peak in box is 1 when "
            "the map's highest pixel is inside that box and 0 when it is not, so its mean is the run's hit rate. "
            "GT saliency peak distance is how many pixels outside that box the highest pixel lies (0 inside it). "
            "Count is the number of detections the score exists for."
        )
        md.add_table(
            headers=SCORE_SUMMARY_HEADERS,
            rows=[summary.as_row(name) for name, summary in outputs.score_summary.items()],
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
                gt_idx = int(datum.matched_gt_index[sal_idx])
                if gt_idx >= 0:
                    gt_box = datum.gt_boxes[gt_idx]
                    plt.gca().add_patch(
                        Rectangle(
                            (gt_box[0], gt_box[1]),
                            gt_box[2] - gt_box[0],
                            gt_box[3] - gt_box[1],
                            linewidth=2,
                            linestyle="--",
                            edgecolor="lime",
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
                entropy = _none_if_nan(datum.saliency_entropy[sal_idx])
                md.add_text(f"**Entropy**: {'n/a' if entropy is None else f'{entropy:.3f} bits'}")
                if gt_idx >= 0:
                    coverage = _none_if_nan(datum.gt_saliency_coverage[sal_idx])
                    peak = _none_if_nan(datum.gt_saliency_peak_in_box[sal_idx])
                    distance = _none_if_nan(datum.gt_saliency_peak_distance[sal_idx])
                    if peak is None or distance is None:
                        peak_text = "map peak n/a"
                    elif peak:
                        peak_text = "map peak inside the box"
                    else:
                        peak_text = f"map peak {distance:.1f} px outside the box"
                    md.add_text(
                        f"**Matched GT box** (green, dashed): {gt_idx}, IoU {datum.matched_gt_iou[sal_idx]:.2f}, "
                        f"saliency coverage {'n/a' if coverage is None else f'{coverage:.2f}'}, {peak_text}"
                    )
                else:
                    md.add_text("**Matched GT box**: none")
                md.add_blank_line()
                md.add_embedded_image(
                    img_path, alt_text=f"Saliency Map for Detection {sal_idx + 1}", remove_source=True
                )

        md.add_provenance(self)

        return InlineTextReport(
            media_type="text/markdown",
            content=md.render(),
            filename=f"{self.capability_id}.md",
        )


class XaitkExplainable(
    XaitkExplainableBase[XaitkExplainableOutputs, od.Dataset, od.Model, od.Metric, XaitkExplainableConfig]
):
    """
    Xai will generate saliency maps for every detections in all images from the dataset.

    Attributes:
        config :The config used to run Xai saliency capability for OD.
    """

    task = "object_detection"

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

        if "index2label" not in model.metadata:
            raise (KeyError("'index2label' not found in model metadata but is required by Xai"))
        class_ids = sorted(model.metadata["index2label"].keys())

        # Each image is read once and that copy is used for prediction,
        # saliency and the outputs.
        results = []
        n_without_detections = 0
        for i in range(len(dataset)):
            raw_image, target, metadata = dataset[i]
            image, scale, y_ratio, x_ratio = _resize_to_max_side(np.asarray(raw_image), config.saliency_max_side)
            model_input = raw_image if scale == 1.0 else image
            dets = _top_detections(model([model_input])[0], class_ids, config.min_score, config.dets_limit)
            if len(dets.labels) == 0:
                n_without_detections += 1
                datum_sal_maps = np.zeros((0, *image.shape[-2:]))
            else:
                (maps,), _ = sal_on_dets(
                    dataset=_SaliencyTargetDataset(image, dets, metadata, class_ids, dataset.metadata["id"]),
                    sal_generator=config.saliency_generator,
                    detector=model,
                    ids=class_ids,
                    img_batch_size=config.img_batch_size,
                )
                datum_sal_maps = _maps_or_zeros(maps, dets, image, image_index=i)
            det_boxes = dets.boxes
            gt_boxes = np.asarray(target.boxes, dtype=float).reshape(-1, 4) * np.array(
                [x_ratio, y_ratio, x_ratio, y_ratio]
            )
            entropy_pairs = [saliency_entropy(datum_sal_maps[k]) for k in range(len(datum_sal_maps))]
            matched_idx, matched_iou, coverage, peak_in_box, peak_distance = _gt_scores(
                datum_sal_maps, det_boxes, gt_boxes, config.gt_match_iou_threshold
            )
            results.append(
                XaitkExplainableOutputDatum.model_validate(
                    {
                        "sal_maps": datum_sal_maps,
                        "img": image,
                        "img_id": metadata["id"],
                        "boxes": det_boxes,
                        "labels": dets.labels,
                        "scores": dets.scores,
                        "saliency_entropy": np.asarray(
                            [np.nan if raw is None else raw for raw, _ in entropy_pairs], dtype=float
                        ),
                        "normalized_saliency_entropy": np.asarray(
                            [np.nan if norm is None else norm for _, norm in entropy_pairs], dtype=float
                        ),
                        "matched_gt_index": matched_idx,
                        "matched_gt_iou": matched_iou,
                        "gt_saliency_coverage": coverage,
                        "gt_saliency_peak_in_box": peak_in_box,
                        "gt_saliency_peak_distance": peak_distance,
                        "gt_boxes": gt_boxes,
                        "image_scale": scale,
                    }
                )
            )
        if n_without_detections:
            logging.info("%d of %d images have no detections to explain", n_without_detections, len(dataset))

        def summarise(field: str) -> XaitkScoreSummary:
            return XaitkScoreSummary.from_values(v for r in results for v in getattr(r, field))

        return XaitkExplainableOutputs.model_validate(
            {
                "results": results,
                "score_summary": {
                    "saliency_entropy": summarise("saliency_entropy"),
                    "normalized_saliency_entropy": summarise("normalized_saliency_entropy"),
                    "matched_gt_iou": summarise("matched_gt_iou"),
                    "gt_saliency_coverage": summarise("gt_saliency_coverage"),
                    "gt_saliency_peak_in_box": summarise("gt_saliency_peak_in_box"),
                    "gt_saliency_peak_distance": summarise("gt_saliency_peak_distance"),
                },
            },
        )
