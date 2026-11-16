import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from pydantic import BaseModel
from xaitk_saliency.impls.saliency_metric.entropy import Entropy

from checkmaite.core.analytics_store._schema import BaseRecord
from checkmaite.core.capability_core import (
    Capability,
    Number,
    TConfig,
    TDataset,
    TMetric,
    TModel,
    TOutputs,
)


class XaitkExplainableBase(Capability[TOutputs, TDataset, TModel, TMetric, TConfig]):
    """Xai capability that takes in the necessary arguements to demo saliency map generation."""

    cache_dependencies = ("xaitk-saliency",)

    @property
    def supports_datasets(self) -> Number:
        """Number of datasets this capability supports.
        Returns
        -------
        Number
            An enumeration value indicating dataset support.
        """
        return Number.ONE

    @property
    def supports_models(self) -> Number:
        """Number of models this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating model support.
        """
        return Number.ONE

    @property
    def supports_metrics(self) -> Number:
        """Number of metrics this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating metric support.
        """
        return Number.ZERO

    @property
    def name(self) -> str:
        return self.__class__.__name__


def saliency_entropy(sal_map: np.ndarray) -> tuple[float | None, float | None]:
    """Return the entropy of one saliency map, raw and normalized.

    Uses XAITK's ``Entropy`` saliency metric with its default settings: the map
    is min-max normalized to [0, 1] and the entropy is in bits. High entropy
    means saliency is spread over many pixels; low entropy means it is
    concentrated in a few.

    The raw value grows with the number of pixels, so the second value divides
    it by ``log2(H * W)``. That puts it in [0, 1] and makes images of different
    sizes comparable.

    Both values are None for a constant map, where entropy is undefined.
    """
    sal_map = np.asarray(sal_map, dtype=float)
    if sal_map.size < 2 or not np.isfinite(sal_map).all() or np.ptp(sal_map) == 0:
        return None, None
    value = Entropy()(sal_map)
    if not math.isfinite(value):
        return None, None
    return value, value / math.log2(sal_map.size)


class XaitkScoreSummary(BaseModel):
    """Summary of one saliency score over a whole run.

    Values that could not be computed (None) are left out. All statistics are
    None when no values remain.
    """

    count: int
    mean: float | None
    median: float | None
    std: float | None
    min: float | None
    max: float | None

    @classmethod
    def from_values(cls, values: Iterable[Any]) -> "XaitkScoreSummary":
        """Summarise the finite values in ``values``."""
        arr = np.asarray(
            [float(v) for v in values if v is not None and math.isfinite(float(v))],
            dtype=float,
        )
        if arr.size == 0:
            return cls(count=0, mean=None, median=None, std=None, min=None, max=None)
        return cls(
            count=int(arr.size),
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
        )

    def as_row(self, name: str) -> list[str]:
        """Format as a Markdown table row: name, count, mean, median, min, max."""

        def fmt(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.3f}"

        return [name, str(self.count), fmt(self.mean), fmt(self.median), fmt(self.min), fmt(self.max)]


SCORE_SUMMARY_HEADERS = ["Score", "Count", "Mean", "Median", "Min", "Max"]


class XaitkExplainableRecord(BaseRecord, table_name="xaitk_explainable"):
    """Record for XaitkExplainable capability results.

    For IC, one record is emitted per image, or per (image, fill color) for
    MC-RISE style generators. The statistics describe the saliency map of the
    explained class: the ground-truth class, or the predicted class when the
    image has no ground truth. For OD, one record is emitted per detection.
    OD-specific fields are None for IC runs, and IC-specific fields are None
    for OD runs.

    Attributes
    ----------
    dataset_id : str
        Dataset identifier (cross-capability JOIN key).
    model_id : str
        Model identifier.
    saliency_generator_type : str
        Short class name of the saliency generator (e.g. ``"RISEStack"``).
    image_index : int
        Position of the image in the dataset (0-based).
    gt_label : str | None
        Ground truth label (IC only). None when the image has no ground truth.
    explained_label : str | None
        Class whose saliency map the statistics describe (IC only).
    fill_color_index : int | None
        Fill color the map belongs to, for MC-RISE style generators (IC only).
    image_id : str | None
        Image ID from dataset metadata (OD only).
    detection_index : int | None
        Which detection within the image (OD only, 0-based).
    predicted_label : str | None
        Predicted class label (top-1 for IC, the detection's label for OD).
    confidence : float | None
        Detection confidence score (OD only).
    mean_saliency : float
        Mean pixel value across the saliency map.
    max_saliency : float
        Maximum pixel value in the saliency map.
    std_saliency : float
        Standard deviation of pixel values.
    positive_saliency_ratio : float
        Fraction of pixels with value > 0.
    saliency_entropy : float | None
        XAITK ``Entropy`` of the map in bits (min-max normalized first).
        None for a constant map.
    normalized_saliency_entropy : float | None
        ``saliency_entropy / log2(H * W)``, in [0, 1].
    matched_gt_index : int | None
        Index of the ground-truth box matched to the detection (OD only).
        None when no box overlaps it with IoU at or above the configured
        threshold.
    matched_gt_iou : float | None
        IoU between the detection box and the matched ground-truth box (OD only).
    gt_saliency_coverage : float | None
        Share of the map's above-average saliency (the map minus its mean,
        clipped at 0) that falls inside the matched ground-truth box, in
        [0, 1] (OD only). None when there is no match.
    gt_saliency_peak_in_box : bool | None
        Whether the map's highest pixel lies inside the matched ground-truth
        box (the pointing game; OD only). None when there is no match.
    gt_saliency_peak_distance : float | None
        Euclidean distance in pixels, at the working resolution, from the map's
        highest pixel to the nearest pixel of the matched ground-truth box; 0
        when the peak is inside the box (OD only). None when there is no match.
    """

    # Cross-capability JOIN key
    dataset_id: str

    # Entity identifiers
    model_id: str

    # Saliency configuration
    saliency_generator_type: str  # e.g. "RISEStack", "DRISEStack"

    # Image identification
    image_index: int  # position in dataset (0-based)

    # IC-specific fields (None for OD runs)
    gt_label: str | None = None
    explained_label: str | None = None
    fill_color_index: int | None = None

    # OD-specific fields (None for IC runs)
    image_id: str | None = None
    detection_index: int | None = None
    predicted_label: str | None = None
    confidence: float | None = None

    # Saliency map summary statistics
    mean_saliency: float
    max_saliency: float
    std_saliency: float
    positive_saliency_ratio: float  # fraction of pixels > 0

    # XAITK saliency scores
    saliency_entropy: float | None = None
    normalized_saliency_entropy: float | None = None

    # OD ground-truth coverage (None for IC runs and unmatched detections)
    matched_gt_index: int | None = None
    matched_gt_iou: float | None = None
    gt_saliency_coverage: float | None = None
    gt_saliency_peak_in_box: bool | None = None
    gt_saliency_peak_distance: float | None = None
