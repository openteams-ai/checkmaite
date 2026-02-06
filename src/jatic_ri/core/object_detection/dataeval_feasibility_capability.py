import logging
from typing import Any

import cv2
import maite.protocols.object_detection as od
import numpy as np
import pandas as pd
import pydantic
import torch
from pydantic import Field
from torchvision.ops import box_iou

from jatic_ri.core._common._knn import compute_ber_and_confusion
from jatic_ri.core._common.feature_extractor import (
    FeatureExtractor,
    load_feature_extractor,
    pca_projector,
    to_unit_interval_01,
)
from jatic_ri.core._types import Device, ModelSpec, TorchvisionModelSpec
from jatic_ri.core._utils import deprecated, requires_optional_dependency, set_device
from jatic_ri.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
)
from jatic_ri.core.report import _gradient as gd
from jatic_ri.core.report._markdown import MarkdownOutput

logger = logging.getLogger(__name__)


class DataevalFeasibilityConfig(CapabilityConfigBase):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    device: Device = pydantic.Field(default_factory=lambda: set_device("cpu"))

    chunk_size: int = Field(
        default=50,
        ge=1,
        description=(
            "Number of images to process in each chunk. Smaller values use less memory "
            "but may have more overhead. Larger values are more efficient but use more memory."
        ),
    )

    embedding_batch_size: int = Field(default=32, ge=1, description="Batch size when computing embeddings on device.")
    knn_n_neighbors: int = pydantic.Field(default=7, description="Number of neighbors for kNN BER estimation.")
    feature_extractor_spec: ModelSpec = Field(
        default_factory=TorchvisionModelSpec,
        description=(
            "Spec for model used to extract embeddings from GT instance crops. "
            "Should be a pretrained image classification model."
        ),
    )
    target_embedding_dim: int = Field(
        default=256,
        ge=1,
        description=(
            "Target embedding dimension after optional PCA. "
            "If extractor outputs more dimensions, PCA reduces to this size."
        ),
    )

    crop_context_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of box size to add as context padding around crops.",
    )
    crop_resize_size: int = Field(
        default=224,
        ge=32,
        description="Size to resize crops to (square) before embedding.",
    )
    min_crop_size: int = Field(default=8, ge=1, description="Minimum crop size in pixels; smaller crops are skipped.")

    small_size_threshold: int = Field(
        default=32, ge=1, description="Boxes with max(w, h) below this are considered 'small'."
    )
    small_object_warn_ratio: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Warn if fraction of small objects exceeds this."
    )
    truncated_bbox_warn_ratio: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Warn if fraction of boundary-touching boxes exceeds this."
    )
    overlap_iou_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="IoU threshold to consider two boxes as highly overlapping."
    )
    overlap_image_warn_ratio: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Warn if fraction of images with high-IoU pairs exceeds this."
    )

    verbose: bool = Field(default=False, description="Log progress messages during execution.")

    @pydantic.model_validator(mode="after")
    def _check_chunk_size_gte_batch_size(self) -> "DataevalFeasibilityConfig":
        if self.chunk_size < self.embedding_batch_size:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be >= embedding_batch_size ({self.embedding_batch_size}). "
                "A smaller chunk_size means fewer crops per chunk than the batch allows, which is inefficient."
            )
        return self


class DatasetHealthStats(pydantic.BaseModel):
    """Health statistics computed from the dataset during the first pass."""

    num_classes: int
    small_object_ratio: float
    truncated_bbox_ratio: float
    overlap_image_ratio: float
    warnings: list[str]


class DataevalFeasibilityOutputs(CapabilityOutputsBase):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    ber_upper: float
    ber_lower: float
    num_instances: int
    num_classes: int
    class_confusion: np.ndarray  # shape (num_classes, num_classes), rows=true, cols=predicted
    confusion_labels: list[int]  # class IDs corresponding to rows/columns of class_confusion
    health_stats: DatasetHealthStats


class DataevalFeasibilityRun(CapabilityRunBase[DataevalFeasibilityConfig, DataevalFeasibilityOutputs]):
    config: DataevalFeasibilityConfig
    outputs: DataevalFeasibilityOutputs

    # The order is important
    @requires_optional_dependency("gradient", install_hint="pip install '.[unsupported]'")
    @deprecated(replacement="collect_md_report")
    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:
        """Create slides for Gradient report.

        Parameters
        ----------
        threshold
            Performance threshold for feasibility determination.

        Returns
        -------
            A list of dictionaries representing slides for the Gradient report.
        """
        results = self.outputs
        dataset_id = self.dataset_metadata[0]["id"]

        is_feasible = (1.0 - results.ber_upper) >= threshold

        semantic_df = pd.DataFrame(
            {
                "Metric": ["BER Upper Bound", "BER Lower Bound", "Num Instances", "Num Classes"],
                "Value": [
                    round(results.ber_upper, 3),
                    round(results.ber_lower, 3),
                    results.num_instances,
                    results.num_classes,
                ],
            }
        )

        semantic_title = f"Dataset: {dataset_id} | Feasibility"
        semantic_heading = "Instance Classification Difficulty"
        semantic_text = [
            gd.Text(t)
            for t in (
                [gd.SubText("Result:", bold=True)],
                f"{'Feasible' if is_feasible else 'Challenging'} " f"for threshold {threshold}",
                [gd.SubText("Interpretation:", bold=True)],
                " * Low BER -> classes are well-separated in embedding space",
                " * High BER -> class ambiguity, label noise, or insufficient visual evidence",
                [gd.SubText("Action:", bold=True)],
                f"* {'No action required' if is_feasible else 'Review class taxonomy and label quality'}",
            )
        ]

        semantic_slide = {
            "deck": self.capability_id,
            "layout_name": "SectionByItem",
            "layout_arguments": {
                gd.SectionByItem.ArgKeys.TITLE.value: semantic_title,
                gd.SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: semantic_heading,
                gd.SectionByItem.ArgKeys.LINE_SECTION_BODY.value: semantic_text,
                gd.SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: semantic_df,
            },
        }

        return [semantic_slide]

    def collect_md_report(self, threshold: float) -> str:  # noqa: C901
        """Create Markdown report for feasibility analysis.

        Parameters
        ----------
        threshold
            Performance threshold for feasibility determination.

        Returns
        -------
        str
            Markdown-formatted report content.
        """
        results = self.outputs
        dataset_id = self.dataset_metadata[0]["id"]

        is_feasible = (1.0 - results.ber_upper) >= threshold

        md = MarkdownOutput("Object Detection Dataset Feasibility Analysis")

        md.add_text(f"**Dataset**: {dataset_id}")
        md.add_text("**Category**: Feasibility")

        md.add_section(heading="kNN Bayes Error Rate")
        md.add_text(
            "Estimates dataset quality via kNN classification error on instance embeddings."
            "A high BER indicates potential mislabelled instances (crops with neighbors "
            "from wrong classes) or confusable class pairs (visually similar categories). "
            "The per-class breakdown identifies which classes warrant investigation."
        )
        md.add_blank_line()

        md.add_text(
            f"**Result:** Problem is likely "
            f"{'feasible' if is_feasible else 'NOT feasible'} "
            f"for threshold {threshold}."
        )
        md.add_blank_line()

        md.add_text("**Interpretation:**")
        md.add_bulleted_list(
            [
                "Low BER -> classes are well-separated in embedding space",
                "High BER -> class ambiguity, label noise, or insufficient visual evidence",
            ]
        )

        md.add_subsection(heading="Results")
        md.add_table(
            headers=["Metric", "Value"],
            rows=[
                ["BER Upper Bound", str(round(results.ber_upper, 3))],
                ["BER Lower Bound", str(round(results.ber_lower, 3))],
                ["Num Instances", str(results.num_instances)],
                ["Num Classes", str(results.num_classes)],
            ],
        )

        if results.class_confusion.size > 0 and results.confusion_labels:
            # Derive per-class BER from confusion matrix: fraction of each class misclassified
            per_class_ber = []
            for i, cls_id in enumerate(results.confusion_labels):
                row_total = results.class_confusion[i].sum()
                if row_total > 0:
                    error_rate = 1.0 - results.class_confusion[i, i] / row_total
                    per_class_ber.append((cls_id, error_rate))
            if per_class_ber:
                md.add_subsection(heading="Per-Class BER")
                md.add_text(
                    "Error rate per class based on kNN neighbor voting. "
                    "High values indicate classes that are frequently confused with others."
                )
                sorted_ber = sorted(per_class_ber, key=lambda x: -x[1])
                class_rows = [[str(cls_id), f"{ber_val:.3f}"] for cls_id, ber_val in sorted_ber]
                md.add_table(headers=["Class ID", "BER"], rows=class_rows)

        if results.class_confusion.size > 0:
            md.add_subsection(heading="Class Confusion Analysis")
            md.add_text(
                "Shows which classes are confused with each other. "
                "For each class, lists the most common predicted classes when kNN disagrees."
            )
            confusion_rows = []
            labels = results.confusion_labels
            for i, true_cls in enumerate(labels):
                row = results.class_confusion[i]
                total = row.sum()
                if total == 0:
                    continue
                errors = [(labels[j], int(row[j])) for j in range(len(labels)) if j != i and row[j] > 0]
                if errors:
                    top_confusions = sorted(errors, key=lambda x: -x[1])[:3]
                    confusion_str = ", ".join(f"{cls}({cnt}/{total})" for cls, cnt in top_confusions)
                    error_rate = (total - row[i]) / total
                    confusion_rows.append([str(true_cls), f"{error_rate:.1%}", confusion_str])
            if confusion_rows:
                md.add_table(
                    headers=["True Class", "Error Rate", "Top Confusions (class(count/total))"],
                    rows=confusion_rows,
                )

        # Dataset Health section
        hs = results.health_stats
        md.add_section(heading="Dataset Health Statistics")
        md.add_text(
            "The threshold value represents instance crop classification accuracy " "(1 − BER), not detection mAP."
        )
        md.add_table(
            headers=["Statistic", "Value"],
            rows=[
                ["Small Object Ratio", f"{hs.small_object_ratio:.1%}"],
                ["Truncated BBox Ratio", f"{hs.truncated_bbox_ratio:.1%}"],
                ["Overlap Image Ratio", f"{hs.overlap_image_ratio:.1%}"],
            ],
        )

        if hs.warnings:
            md.add_subsection(heading="Dataset Health Warnings")
            md.add_bulleted_list(hs.warnings)

        md.add_section(heading="Recommendations")

        if not is_feasible:
            md.add_text("**Issues Detected:**")
            md.add_bulleted_list(
                [
                    "Review class taxonomy for ambiguous or overlapping categories",
                    "Check for label noise or annotation inconsistencies",
                    "Consider representation/domain mismatch with embedding model",
                    "Limited gains expected from model improvements alone",
                ]
            )
        else:
            md.add_text("**Dataset appears feasible for the given threshold.**")
            md.add_bulleted_list(
                [
                    "Classes are sufficiently separable in embedding space",
                    "Focus on model architecture and training optimization",
                ]
            )

        return md.render()


def _letterbox_to_square(
    img: np.ndarray,
    size: int,
) -> np.ndarray:
    """
    Aspect-preserving resize to fit within (size,size), then pad to (size,size).
    Expects HWC (or HW for grayscale); returns HWC.
    """

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty crop passed to letterbox.")

    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_x = size - new_w
    pad_y = size - new_h
    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top

    return cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)


def _extract_instance_crops(
    image: np.ndarray,
    boxes: np.ndarray,
    context_fraction: float,
    resize_size: int,
    min_crop_size: int,
) -> list[np.ndarray]:
    """Extract and preprocess instance crops from an image.

    Parameters
    ----------
    image
        Image array in CHW format.
    boxes
        Bounding boxes in xyxy format, shape (N, 4).
    context_fraction
        Fraction of box size to add as padding context.
    resize_size
        Target size for resized crops (square).
    min_crop_size
        Minimum crop dimension; smaller crops are skipped.

    Returns
    -------
        List of preprocessed crop arrays in CHW format.
    """

    # _letterbox_to_square expects HWC
    image = np.transpose(image, (1, 2, 0))

    h, w = image.shape[:2]
    crops = []

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w < min_crop_size or box_h < min_crop_size:
            continue

        pad_w = box_w * context_fraction
        pad_h = box_h * context_fraction

        cx1 = max(0, int(x1 - pad_w))
        cy1 = max(0, int(y1 - pad_h))
        cx2 = min(w, int(x2 + pad_w))
        cy2 = min(h, int(y2 + pad_h))

        crop = image[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            continue

        crop_resized = _letterbox_to_square(crop, size=resize_size)

        crop_chw = np.transpose(crop_resized, (2, 0, 1))

        crops.append(crop_chw)

    return crops


def _first_pass_scan(
    dataset: od.Dataset,
    config: "DataevalFeasibilityConfig",
) -> tuple[int, list[int], DatasetHealthStats]:
    """Scan dataset to collect labels, count valid crops, and compute health stats.

    Returns
    -------
    tuple
        (total_crops, all_labels, health_stats)
    """
    num_images = len(dataset)
    verbose = config.verbose
    total_crops = 0
    all_labels: list[int] = []
    total_boxes = 0
    small_count = 0
    truncated_count = 0
    overlap_image_count = 0

    if verbose:
        logger.info("First pass: scanning dataset (%d images) for health stats...", num_images)

    for idx in range(num_images):
        image, target, _ = dataset[idx]
        image_np = np.array(image)
        _, img_h, img_w = image_np.shape
        boxes_np = np.array(target.boxes)
        labels_np = np.array(target.labels)

        if len(boxes_np) == 0:
            continue

        total_boxes += len(boxes_np)

        boxes_wh = boxes_np[:, [2, 3]] - boxes_np[:, [0, 1]]
        max_side = boxes_wh.max(axis=-1)
        small_count += int((max_side < config.small_size_threshold).sum())

        touches_boundary = (
            (boxes_np[:, 0] <= 0)
            | (boxes_np[:, 1] <= 0)
            | (boxes_np[:, 2] >= img_w - 1)
            | (boxes_np[:, 3] >= img_h - 1)
        )
        truncated_count += int(touches_boundary.sum())

        if _has_high_iou_pair(boxes_np, config.overlap_iou_threshold):
            overlap_image_count += 1

        valid_mask = (boxes_wh >= config.min_crop_size).all(axis=-1)
        selected_labels = labels_np[valid_mask]
        all_labels.extend(selected_labels.tolist())
        total_crops += len(selected_labels)

    if total_crops == 0:
        raise ValueError(
            "No valid instance crops could be extracted from the dataset. "
            "Check that the dataset contains bounding boxes larger than min_crop_size."
        )

    if verbose:
        logger.info("First pass complete: %d crops, %d classes", total_crops, len(set(all_labels)))

    health_stats = _build_health_stats(
        all_labels=all_labels,
        total_boxes=total_boxes,
        small_count=small_count,
        truncated_count=truncated_count,
        overlap_image_count=overlap_image_count,
        num_images=num_images,
        config=config,
    )

    return total_crops, all_labels, health_stats


def _build_health_stats(
    all_labels: list[int],
    total_boxes: int,
    small_count: int,
    truncated_count: int,
    overlap_image_count: int,
    num_images: int,
    config: "DataevalFeasibilityConfig",
) -> DatasetHealthStats:
    """Validate dataset health and return stats with any warnings."""
    num_unique_classes = len(set(all_labels))
    small_object_ratio = small_count / total_boxes if total_boxes > 0 else 0.0
    truncated_bbox_ratio = truncated_count / total_boxes if total_boxes > 0 else 0.0
    overlap_image_ratio = overlap_image_count / num_images if num_images > 0 else 0.0

    if num_unique_classes < 2:
        raise ValueError(f"Dataset contains only {num_unique_classes} class(es). kNN BER requires at least 2 classes.")

    health_warnings: list[str] = []
    if small_object_ratio > config.small_object_warn_ratio:
        msg = (
            f"High fraction of small objects: {small_object_ratio:.1%} "
            f"(threshold {config.small_object_warn_ratio:.1%})"
        )
        logger.warning(msg)
        health_warnings.append(msg)
    if truncated_bbox_ratio > config.truncated_bbox_warn_ratio:
        msg = (
            f"High fraction of truncated (boundary-touching) boxes: "
            f"{truncated_bbox_ratio:.1%} (threshold {config.truncated_bbox_warn_ratio:.1%})"
        )
        logger.warning(msg)
        health_warnings.append(msg)
    if overlap_image_ratio > config.overlap_image_warn_ratio:
        msg = (
            f"High fraction of images with overlapping boxes "
            f"(IoU>={config.overlap_iou_threshold}): {overlap_image_ratio:.1%} "
            f"(threshold {config.overlap_image_warn_ratio:.1%})"
        )
        logger.warning(msg)
        health_warnings.append(msg)

    return DatasetHealthStats(
        num_classes=num_unique_classes,
        small_object_ratio=small_object_ratio,
        truncated_bbox_ratio=truncated_bbox_ratio,
        overlap_image_ratio=overlap_image_ratio,
        warnings=health_warnings,
    )


def _has_high_iou_pair(boxes: np.ndarray, threshold: float) -> bool:
    """Return True if any pair of boxes in the array has IoU >= threshold."""
    n = len(boxes)
    if n < 2:
        return False
    t = torch.from_numpy(boxes).float()
    ious = box_iou(t, t)
    ious.fill_diagonal_(0.0)
    return bool((ious >= threshold).any().item())


class DataevalFeasibility(
    Capability[
        DataevalFeasibilityOutputs,
        od.Dataset,
        od.Model,
        od.Metric,
        DataevalFeasibilityConfig,
    ]
):
    """Object Detection Dataset Feasibility Capability.

    Estimates dataset difficulty via kNN Bayes Error Rate (BER) on embeddings
    of ground-truth instance crops. The BER measures **crop classification
    separability** — how well a kNN classifier can distinguish object classes
    from their visual appearance alone — *not* detection mAP.

    Assumptions
    -----------
    - The dataset contains at least 2 classes (hard requirement; raises
      ``ValueError`` otherwise).
    - Objects are reasonably sized — very small objects produce low-quality
      crops that degrade the embedding signal.
    - Objects are not heavily truncated — boundary-touching boxes yield
      partial crops that may mislead the classifier.
    - Ground-truth boxes do not heavily overlap — near-duplicate boxes
      inflate instance counts and bias the BER estimate.

    Dataset health checks
    ---------------------
    During the first pass the capability computes health statistics
    (small-object ratio, truncated-bbox ratio, overlap-image ratio).
    Violations are surfaced three ways:

    1. ``logger.warning()`` at runtime.
    2. ``outputs.health_stats.warnings`` list in the capability outputs.
    3. A dedicated "Dataset Health" section in ``collect_md_report()``.

    This is a standardized probe that is cheap, repeatable, and comparable
    across datasets when the embedding model is fixed.
    """

    _RUN_TYPE = DataevalFeasibilityRun

    @classmethod
    def _create_config(cls) -> DataevalFeasibilityConfig:
        return DataevalFeasibilityConfig()

    @property
    def supports_datasets(self) -> Number:
        return Number.ONE

    @property
    def supports_models(self) -> Number:
        return Number.ZERO

    @property
    def supports_metrics(self) -> Number:
        return Number.ZERO

    def _run(
        self,
        models: list[od.Model],  # noqa: ARG002
        datasets: list[od.Dataset],
        metrics: list[od.Metric],  # noqa: ARG002
        config: DataevalFeasibilityConfig,
        use_prediction_and_evaluation_cache: bool,  # noqa: ARG002
    ) -> DataevalFeasibilityOutputs:
        """Run the feasibility capability."""
        dataset = datasets[0]
        num_images = len(dataset)

        # First pass: count valid crops, collect labels, and compute health stats
        total_crops, all_labels, health_stats = _first_pass_scan(dataset, config)

        fe = load_feature_extractor(device=config.device, model_spec=config.feature_extractor_spec)

        # Pre-allocate arrays
        labels_array = np.array(all_labels, dtype=np.int64)
        embeddings_array = np.empty((total_crops, fe.out_dim), dtype=np.float32)
        current_idx = 0

        # Second pass: extract crops and compute embeddings, filling pre-allocated array
        num_chunks = (num_images + config.chunk_size - 1) // config.chunk_size
        for chunk_idx, chunk_start in enumerate(range(0, num_images, config.chunk_size)):
            chunk_end = min(chunk_start + config.chunk_size, num_images)
            chunk_indices = range(chunk_start, chunk_end)

            if config.verbose:
                logger.info(
                    "Second pass: embedding chunk %d/%d (%d images)...",
                    chunk_idx + 1,
                    num_chunks,
                    chunk_end - chunk_start,
                )

            # Extract crops sequentially (profiling shows this is <0.1% of total time,
            # so multiprocessing overhead likely exceeds any parallelism benefit)
            chunk_crops: list[np.ndarray] = []
            for idx in chunk_indices:
                image, target, _ = dataset[idx]

                image_np = np.array(image)
                boxes_np = np.array(target.boxes)

                crops = _extract_instance_crops(
                    image=image_np,
                    boxes=boxes_np,
                    context_fraction=config.crop_context_fraction,
                    resize_size=config.crop_resize_size,
                    min_crop_size=config.min_crop_size,
                )
                chunk_crops.extend(crops)

            if len(chunk_crops) == 0:
                continue

            chunk_embeddings = self._compute_embeddings_batched(
                crops=chunk_crops,
                feature_extractor=fe,
                batch_size=config.embedding_batch_size,
                device=config.device,
            )

            chunk_size = len(chunk_embeddings)
            embeddings_array[current_idx : current_idx + chunk_size] = chunk_embeddings
            current_idx += chunk_size

        n, d = embeddings_array.shape
        if config.target_embedding_dim < d:
            k_max = min(n, d)
            k = min(config.target_embedding_dim, k_max)

            if k != config.target_embedding_dim:
                logger.warning(
                    f"Requested target_embedding_dim={config.target_embedding_dim}, but PCA limited to "
                    f"min(N, D)={k_max}. Using {k} components."
                )

            if k < d:
                proj = pca_projector(embeddings_array, out_dim=k)
                embeddings_array = proj.transform(embeddings_array)

        embeddings_array = to_unit_interval_01(embeddings_array)

        if config.verbose:
            logger.info(
                "Computing kNN BER (%d instances, %d-d embeddings, k=%d)...",
                embeddings_array.shape[0],
                embeddings_array.shape[1],
                config.knn_n_neighbors,
            )

        ber_upper, ber_lower, class_confusion, confusion_labels = compute_ber_and_confusion(
            embeddings=embeddings_array,
            labels=labels_array,
            k=config.knn_n_neighbors,
        )

        return DataevalFeasibilityOutputs(
            ber_upper=ber_upper,
            ber_lower=ber_lower,
            num_instances=total_crops,
            num_classes=len(np.unique(labels_array)),
            class_confusion=class_confusion,
            confusion_labels=confusion_labels,
            health_stats=health_stats,
        )

    # Embedding computation is ~90% of total runtime when running on CPU.
    # When GPU/MPS is available, the GPU handles parallelism internally and
    # provides ~10x speedup.

    def _compute_embeddings_batched(
        self,
        crops: list[np.ndarray],
        feature_extractor: FeatureExtractor,
        batch_size: int,
        device: torch.device,
    ) -> np.ndarray:
        """Compute embeddings for crops in batches.

        Parameters
        ----------
        crops
            List of crop arrays in CHW format.
        feature_extractor
            The feature extractor to use.
        batch_size
            Number of crops to process at once.
        device
            Device to run inference on.

        Returns
        -------
            Embeddings array of shape (N, D).
        """
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                batch_crops = crops[i : i + batch_size]

                batch_tensors = []
                for crop in batch_crops:
                    crop_tensor = torch.from_numpy(crop)
                    crop_transformed = feature_extractor.transforms(crop_tensor)
                    batch_tensors.append(crop_transformed)

                batch = torch.stack(batch_tensors).to(device)
                embeddings = feature_extractor.model(batch)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
