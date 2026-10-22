"""Metrics for multi-object tracking (MOT) evaluation.

CLEAR-MOT (MOTA/MOTP) and identity (IDF1) metrics are the community-standard
evaluation figures for multi-object tracking, and what the program's FMV
evaluation path (datamaite MOT datasets, tracking-by-detection models) is
measured with (#723). They are computed here with `motmetrics
<https://github.com/cheind/py-motmetrics>`_.

Scope: this is **generic class-agnostic CLEAR-MOT evaluation**, not a
reimplementation of the MOTChallenge devkit. Every loaded ground-truth box is
an ordinary target: labels take no part in matching, distractor classes are not
filtered out, and predictions overlapping ignored regions are not suppressed.
Numbers from a real MOTChallenge benchmark root will therefore differ from
published leaderboard figures. Devkit-faithful evaluation — class filtering plus
ignored-region handling — is tracked in checkmaite#742.
"""

import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import motmetrics as mm
import numpy as np
from maite.protocols import MetricMetadata
from maite.protocols import multiobject_tracking as mot

from checkmaite.core._utils import id_hash

__all__: list[str] = [
    "DEFAULT_MOT_METRIC_NAMES",
    "MotMetricsTrackingMetric",
    "clear_mot_metrics_factory",
]


class MetricInputDataError(Exception):
    pass


#: Metrics computed by default: MOTA and IDF1 are the headline CLEAR-MOT and
#: identity figures; MOTP and identity switches are their standard companions.
DEFAULT_MOT_METRIC_NAMES: tuple[str, ...] = ("mota", "motp", "idf1", "num_switches")


def _frame_arrays(
    frame: mot.SingleFrameObjectTrackingTarget,
    *,
    batch_name: str,
    datum_index: int,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one frame target and return its ``(boxes_xyxy, track_ids)`` arrays."""
    where = f"{batch_name}[{datum_index}].frame_tracks[{frame_index}]"
    boxes = _boxes_array(frame.boxes, where=where)
    track_ids = _track_ids_array(frame.track_ids, expected_length=boxes.shape[0], where=where)
    return boxes, track_ids


def _boxes_array(raw: Any, *, where: str) -> np.ndarray:
    """Validate one frame's boxes and return them as ``(N, 4)`` float64 xyxy.

    The shape is validated as given, never reshaped into agreement: a bare
    ``(8,)`` or a ``(2, 8)`` array would otherwise be reinterpreted as four
    detections, and a ``(0, 5)`` array would be normalized into a valid empty
    frame while telling us the producer disagrees about what a box is. An empty
    1-D array is the one accepted exception: ``[]`` is how a frame with no
    detections arrives.
    """
    boxes = _numeric_array(raw, where=where, field="boxes")
    if boxes.ndim == 1 and boxes.shape[0] == 0:
        return np.empty((0, 4))
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise MetricInputDataError(
            f"Invalid {where}.boxes shape: got {tuple(boxes.shape)}, expected (N_DETECTIONS, 4)."
        )

    boxes = boxes.astype(np.float64)
    if not np.isfinite(boxes).all():
        raise MetricInputDataError(f"Invalid {where}.boxes: contains non-finite values (NaN or infinity).")
    # An inverted box has negative width or height, which makes every IoU with it
    # NaN; motmetrics then silently treats the detection as unmatchable, so the
    # run reports plausible numbers computed from fewer boxes than were supplied.
    if (boxes[:, 2] < boxes[:, 0]).any() or (boxes[:, 3] < boxes[:, 1]).any():
        raise MetricInputDataError(
            f"Invalid {where}.boxes: expected xyxy corners with x1 >= x0 and y1 >= y0, got inverted boxes."
        )
    return boxes


def _numeric_array(raw: Any, *, where: str, field: str) -> np.ndarray:
    """Convert one frame field to a numeric array, reporting every failure in the same terms.

    ``np.asarray`` raises a bare ``ValueError`` for ragged or non-numeric input
    and torch raises ``RuntimeError`` for a tensor that still requires grad;
    those all describe one user-facing problem. String and object dtypes are
    rejected rather than coerced, because ``np.asarray([["0", "0", "1", "1"]],
    dtype=float)`` succeeds — numeric-looking strings would otherwise be
    accepted as boxes.
    """
    try:
        array = np.asarray(raw)
    except (TypeError, ValueError, RuntimeError) as error:
        raise MetricInputDataError(
            f"Invalid {where}.{field}: expected a numeric array, got {type(raw).__name__} "
            f"({type(error).__name__}: {error})."
        ) from error
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise MetricInputDataError(f"Invalid {where}.{field}: expected a numeric array, got dtype {array.dtype}.")
    return array


def _track_ids_array(raw: Any, *, expected_length: int, where: str) -> np.ndarray:
    """Validate one frame's track ids and return them as int64.

    Track ids are identities, not measurements, so an id is either representable
    exactly or rejected. Integer arrays stay integers: routing them through
    ``float64`` collapses ``2**53 + 1`` onto ``2**53`` and merges two distinct
    tracks. Floating-point input is accepted only when every value is a whole
    number its own dtype holds exactly — ``1.9`` is rejected rather than
    truncated to track ``1``, and ``float32`` cannot tell ``16777217`` from
    ``16777216``.
    """
    track_ids = _exact_integer_ids(raw, where=where)
    if track_ids.shape != (expected_length,):
        raise MetricInputDataError(
            f"Invalid {where}.track_ids shape: got {tuple(track_ids.shape)}, "
            f"expected ({expected_length},) to match boxes."
        )
    # A track holds at most one box per frame, so a repeated id is either a
    # labelling error or two ids that aliased onto one value in a narrow float
    # dtype. Unassigned predictions (-1) are exempt: a frame may hold many.
    identities = track_ids[track_ids >= 0]
    if np.unique(identities).size != identities.size:
        raise MetricInputDataError(
            f"Invalid {where}.track_ids: the same identity appears more than once in one frame "
            f"({track_ids.tolist()})."
        )
    return track_ids


def _exact_integer_ids(raw: Any, *, where: str) -> np.ndarray:
    """Return ``raw`` as exact int64 ids, or raise :class:`MetricInputDataError`."""
    try:
        ids = np.asarray(raw)
    except (TypeError, ValueError, RuntimeError) as error:
        raise MetricInputDataError(
            f"Invalid {where}.track_ids: expected integer ids, got {type(raw).__name__} "
            f"({type(error).__name__}: {error})."
        ) from error

    if ids.size == 0:
        return np.empty((0,), dtype=np.int64)
    if ids.dtype == np.bool_:
        raise MetricInputDataError(f"Invalid {where}.track_ids: expected integer ids, got dtype {ids.dtype}.")
    if ids.dtype == object:
        return _object_dtype_ids(ids, where=where)
    if np.issubdtype(ids.dtype, np.signedinteger):
        return ids.astype(np.int64)
    if np.issubdtype(ids.dtype, np.unsignedinteger):
        return _within_int64([int(value) for value in ids.reshape(-1).tolist()], shape=ids.shape, where=where)
    if np.issubdtype(ids.dtype, np.floating):
        return _float_dtype_ids(ids, where=where)
    raise MetricInputDataError(f"Invalid {where}.track_ids: expected integer ids, got dtype {ids.dtype}.")


def _object_dtype_ids(ids: np.ndarray, *, where: str) -> np.ndarray:
    """Convert object-dtype ids, where a Python int wider than int64 also lands.

    Integrality and range are checked on the objects themselves so the error says
    which of the two is actually wrong.
    """
    values = ids.reshape(-1).tolist()
    if not all(isinstance(value, numbers.Integral) for value in values):
        raise MetricInputDataError(f"Invalid {where}.track_ids: expected integer ids, got {values}.")
    return _within_int64([int(value) for value in values], shape=ids.shape, where=where)


def _float_dtype_ids(ids: np.ndarray, *, where: str) -> np.ndarray:
    """Convert floating-point ids, accepting only whole numbers the dtype holds exactly."""
    if not np.isfinite(ids).all():
        raise MetricInputDataError(f"Invalid {where}.track_ids: contains non-finite values (NaN or infinity).")
    if not (ids == np.rint(ids)).all():
        raise MetricInputDataError(f"Invalid {where}.track_ids: expected integer-valued ids, got {ids.tolist()}.")
    # Largest integer this float dtype represents exactly: at and beyond it,
    # adjacent ids alias onto one value (float16 at 2**11, float32 at 2**24,
    # float64 at 2**53), so a merge may already have happened before we saw them.
    exact_limit = 2.0 ** (np.finfo(ids.dtype).nmant + 1)
    if np.abs(ids).max() >= exact_limit:
        raise MetricInputDataError(
            f"Invalid {where}.track_ids: {ids.dtype} ids at or beyond {int(exact_limit)} are not represented "
            f"exactly, so distinct tracks may already have merged; pass integer ids."
        )
    return ids.astype(np.int64)


def _within_int64(values: list[int], *, shape: tuple[int, ...], where: str) -> np.ndarray:
    """Convert already-integral ids to int64, rejecting anything the type cannot hold."""
    if max(values) > np.iinfo(np.int64).max:
        raise MetricInputDataError(f"Invalid {where}.track_ids: ids exceed the int64 range ({np.iinfo(np.int64).max}).")
    if min(values) < np.iinfo(np.int64).min:
        raise MetricInputDataError(
            f"Invalid {where}.track_ids: ids are below the int64 range ({np.iinfo(np.int64).min})."
        )
    return np.asarray(values, dtype=np.int64).reshape(shape)


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert ``(x0, y0, x1, y1)`` MAITE boxes to the ``(x, y, w, h)`` motmetrics expects."""
    converted = boxes.copy()
    converted[:, 2] = boxes[:, 2] - boxes[:, 0]
    converted[:, 3] = boxes[:, 3] - boxes[:, 1]
    return converted


def _iou_distance_matrix(gt_boxes: np.ndarray, hyp_boxes: np.ndarray, *, max_distance: float) -> np.ndarray:
    """Return ``motmetrics.distances.iou_matrix`` for float64 ``(x, y, w, h)`` boxes.

    motmetrics 1.4.0, its latest release, converts its inputs with ``np.asfarray``,
    which NumPy 2 removed, and CheckMAITE resolves NumPy 2 on Python 3.13. Inputs
    here are already float64, so this keeps the rest of ``iou_matrix`` unchanged:
    IoU distance (1 - IoU), with pairs beyond ``max_distance`` set to NaN
    (do not pair), and an empty matrix when either side has no boxes.
    """
    if gt_boxes.size == 0 or hyp_boxes.size == 0:
        return np.empty((0, 0))
    distances = 1 - mm.distances.boxiou(gt_boxes[:, None], hyp_boxes[None, :])
    return np.where(distances > max_distance, np.nan, distances)


class MotMetricsTrackingMetric(mot.Metric):
    """A MAITE-compliant wrapper for multi-object-tracking metrics from the motmetrics package.

    Each MAITE MOT datum is one video: predictions and targets are
    per-video sequences of per-frame ``boxes``/``labels``/``scores``/``track_ids``
    frame targets. ``update()`` accumulates one ``motmetrics`` accumulator per
    video (keyed by datum-metadata ``id``); ``compute()`` reports metrics over
    all accumulated videos combined (motmetrics ``OVERALL``).

    Predictions and ground truth are associated per frame by IoU, with pairs
    below ``iou_threshold`` ineligible. Association is **class-agnostic**:
    ``labels`` do not participate in matching, mirroring the class-agnostic
    association of tracking-by-detection models such as ByteTrack. Predicted
    detections with ``track_id == -1`` (returned but not assigned to any track)
    are excluded: identity metrics are only defined for identity-bearing
    predictions.

    Because matching is class-agnostic, every ground-truth box counts as a
    target regardless of its class, and nothing is treated as a distractor or
    an ignored region. This is generic CLEAR-MOT evaluation rather than the
    MOTChallenge devkit protocol, which filters classes and suppresses
    predictions inside ignored regions (checkmaite#742); benchmark-comparable
    numbers need that filtering.

    Note that motmetrics ``motp`` is the mean IoU *distance* (1 - IoU) over
    matches — lower is better — not the MOTChallenge-percentage variant.

    A value that is undefined for the evaluation, such as ``motp`` when nothing
    matched, is reported as ``None`` rather than NaN or infinity. ``compute()``
    raises :class:`MetricInputDataError` if ``return_key`` itself is undefined,
    such as ``mota`` when there are no ground-truth objects.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU for a prediction/ground-truth pair to be matchable,
        by default 0.5 (the customary tracking-evaluation threshold).
    metric_names : Sequence[str]
        motmetrics metric names to compute, by default
        :data:`DEFAULT_MOT_METRIC_NAMES`.
    return_key : str
        Key in the ``compute()`` mapping treated as the overall metric by the
        evaluation capability, by default ``"mota"``. Must be in ``metric_names``.
    metric_id : str
        The ID for the metric metadata.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.5,
        metric_names: Sequence[str] = DEFAULT_MOT_METRIC_NAMES,
        return_key: str = "mota",
        metric_id: str,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}.")
        if return_key not in metric_names:
            raise ValueError(f"return_key {return_key!r} is not among metric_names {tuple(metric_names)}.")
        self._iou_threshold = iou_threshold
        self._metric_names = tuple(metric_names)
        self.return_key: str = return_key
        self.metadata = MetricMetadata(id=metric_id)
        self._accumulators: dict[str, mm.MOTAccumulator] = {}

    def update(
        self,
        preds: Sequence[mot.TargetType],
        targets: Sequence[mot.TargetType],
        metadatas: Sequence[mot.DatumMetadataType] = [],
    ) -> None:
        if len(preds) != len(targets):
            raise MetricInputDataError(
                f"Prediction and target batches differ in length: {len(preds)} != {len(targets)}."
            )
        # CheckMAITE's evaluation always passes datum metadata, so videos are keyed
        # by their metadata id and a repeated id is rejected. Only a direct
        # update() call without metadata keys videos by arrival order.
        if metadatas and len(metadatas) != len(preds):
            raise MetricInputDataError(
                f"Metadata batch length {len(metadatas)} does not match batch length {len(preds)}; "
                "MOT accumulation is keyed by the datum-metadata video id when metadata is provided."
            )

        for datum_index, (pred, target) in enumerate(zip(preds, targets, strict=False)):
            if metadatas:
                video_id = str(metadatas[datum_index]["id"])
                if video_id in self._accumulators:
                    raise MetricInputDataError(
                        f"Video id {video_id!r} was already accumulated; each MOT datum must be one whole video."
                    )
            else:
                video_id = f"__anonymous_video_{len(self._accumulators)}"

            pred_frames = list(pred.frame_tracks)
            target_frames = list(target.frame_tracks)
            if len(pred_frames) != len(target_frames):
                raise MetricInputDataError(
                    f"preds[{datum_index}] has {len(pred_frames)} frame targets but targets[{datum_index}] "
                    f"has {len(target_frames)}; frames cannot be aligned for evaluation."
                )

            accumulator = mm.MOTAccumulator(auto_id=False)
            for frame_index, (pred_frame, target_frame) in enumerate(zip(pred_frames, target_frames, strict=False)):
                hyp_boxes, hyp_ids = _frame_arrays(
                    pred_frame, batch_name="preds", datum_index=datum_index, frame_index=frame_index
                )
                gt_boxes, gt_ids = _frame_arrays(
                    target_frame, batch_name="targets", datum_index=datum_index, frame_index=frame_index
                )
                if (gt_ids < 0).any():
                    raise MetricInputDataError(
                        f"targets[{datum_index}].frame_tracks[{frame_index}] contains negative track_ids; "
                        "ground truth for tracking evaluation must assign every detection to a track."
                    )
                # Unconfirmed predictions carry no identity to evaluate.
                confirmed = hyp_ids >= 0
                hyp_boxes, hyp_ids = hyp_boxes[confirmed], hyp_ids[confirmed]

                distances = _iou_distance_matrix(
                    _xyxy_to_xywh(gt_boxes),
                    _xyxy_to_xywh(hyp_boxes),
                    # A maximum IoU *distance* (1 - IoU), as motmetrics' max_iou.
                    max_distance=1.0 - self._iou_threshold + 1e-12,
                )
                accumulator.update(gt_ids.tolist(), hyp_ids.tolist(), distances, frameid=frame_index)

            self._accumulators[video_id] = accumulator

    def compute(self) -> Mapping[str, Any]:
        if not self._accumulators:
            raise MetricInputDataError("compute() called before any videos were accumulated via update().")
        metrics_host = mm.metrics.create()
        summary = metrics_host.compute_many(
            list(self._accumulators.values()),
            names=list(self._accumulators.keys()),
            metrics=list(dict.fromkeys([*self._metric_names, "num_detections"])),
            generate_overall=True,
        )
        # generate_overall appends the OVERALL row last; select by position so a
        # video whose id is "OVERALL" cannot shadow it.
        per_video, overall = summary.iloc[:-1], summary.iloc[-1]
        results = {name: float(overall[name]) for name in self._metric_names}
        if "motp" in results:
            # motmetrics combines MOTP as sum(motp * num_detections) over videos, so a
            # video without matches (motp NaN, num_detections 0) makes the overall
            # NaN. Weight only the videos that have matches.
            matched = per_video[per_video["num_detections"] > 0]
            num_detections = float(matched["num_detections"].sum())
            results["motp"] = (
                float((matched["motp"] * matched["num_detections"]).sum()) / num_detections
                if num_detections
                else float("nan")
            )
        # A score that is a ratio with a zero denominator is undefined (motmetrics
        # reports NaN or infinity). CheckMAITE reports undefined values as None,
        # which JSON caches and run files keep, and refuses an undefined headline.
        undefined = {name for name, value in results.items() if not math.isfinite(value)}
        if self.return_key in undefined:
            raise MetricInputDataError(
                f"{self.return_key!r} is undefined for this evaluation (not a finite number); "
                "for example, MOTA is undefined when the dataset has no ground-truth objects."
            )
        return {name: None if name in undefined else value for name, value in results.items()}

    def reset(self) -> None:
        self._accumulators = {}


def clear_mot_metrics_factory(
    iou_threshold: float = 0.5,
    metric_names: Sequence[str] = DEFAULT_MOT_METRIC_NAMES,
) -> MotMetricsTrackingMetric:
    """Create the standard class-agnostic CLEAR-MOT metric (MOTA overall).

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU for a prediction/ground-truth match, by default 0.5.
    metric_names : Sequence[str]
        motmetrics metric names to compute, by default
        :data:`DEFAULT_MOT_METRIC_NAMES`.

    Returns
    -------
    A metric whose ``compute()`` reports the requested metrics over all
    accumulated videos, with ``mota`` as the overall figure. Matching is
    class-agnostic with no distractor or ignored-region handling, so this is
    not MOTChallenge-devkit-equivalent (checkmaite#742).
    """
    return MotMetricsTrackingMetric(
        iou_threshold=iou_threshold,
        metric_names=metric_names,
        metric_id=f"motmetrics_clear_{id_hash(iou_threshold=iou_threshold, metric_names=tuple(metric_names))}",
    )
