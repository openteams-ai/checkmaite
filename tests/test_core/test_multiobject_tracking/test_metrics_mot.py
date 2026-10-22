import numpy as np
import pytest

from checkmaite.core.multiobject_tracking.metrics import (
    DEFAULT_MOT_METRIC_NAMES,
    MetricInputDataError,
    MotMetricsTrackingMetric,
    clear_mot_metrics_factory,
)
from tests.test_core.test_multiobject_tracking.conftest import (
    DEFAULT_MOT_DATASET_TARGETS,
    DEFAULT_MOT_DATUM_METADATAS,
    DEFAULT_MOT_MODEL_PREDICTIONS,
    EXPECTED_MOT_IDF1,
    EXPECTED_MOT_MOTA,
    EXPECTED_MOT_MOTP,
    EXPECTED_MOT_NUM_SWITCHES,
    FakeMOTTarget,
    frame_tracks,
)


def make_metric(**kwargs) -> MotMetricsTrackingMetric:
    kwargs.setdefault("metric_id", "test_mot_metric")
    return MotMetricsTrackingMetric(**kwargs)


def update_with_defaults(metric: MotMetricsTrackingMetric) -> None:
    metric.update(DEFAULT_MOT_MODEL_PREDICTIONS, DEFAULT_MOT_DATASET_TARGETS, DEFAULT_MOT_DATUM_METADATAS)


def test_clear_mot_metrics_hand_computed_values():
    """The canned two-video scenario reproduces its hand-computed CLEAR-MOT/identity figures."""
    metric = make_metric()
    update_with_defaults(metric)
    result = metric.compute()

    assert result["mota"] == pytest.approx(EXPECTED_MOT_MOTA)
    assert result["motp"] == pytest.approx(EXPECTED_MOT_MOTP, abs=1e-9)
    assert result["idf1"] == pytest.approx(EXPECTED_MOT_IDF1)
    assert result["num_switches"] == EXPECTED_MOT_NUM_SWITCHES


def test_update_accepts_videos_across_multiple_batches():
    """Accumulating the two videos in separate update() calls matches single-batch results."""
    metric = make_metric()
    for i in range(2):
        metric.update(
            [DEFAULT_MOT_MODEL_PREDICTIONS[i]], [DEFAULT_MOT_DATASET_TARGETS[i]], [DEFAULT_MOT_DATUM_METADATAS[i]]
        )
    assert metric.compute()["mota"] == pytest.approx(EXPECTED_MOT_MOTA)


def test_unconfirmed_predictions_are_excluded():
    """track_id == -1 predictions are dropped; giving them an id makes them false positives."""
    confirmed = [
        FakeMOTTarget(
            frame_tracks=[
                frame_tracks(
                    [ft.boxes[j].tolist() for j in range(len(ft.track_ids))],
                    [105 if tid == -1 else int(tid) for tid in ft.track_ids],
                )
                for ft in DEFAULT_MOT_MODEL_PREDICTIONS[0].frame_tracks
            ]
        ),
        DEFAULT_MOT_MODEL_PREDICTIONS[1],
    ]
    metric = make_metric()
    metric.update(confirmed, DEFAULT_MOT_DATASET_TARGETS, DEFAULT_MOT_DATUM_METADATAS)

    # The formerly-unconfirmed detection now counts as a second false positive:
    # MOTA drops from 1 - 3/8 to 1 - 4/8.
    assert metric.compute()["mota"] == pytest.approx(0.5)


def test_iou_threshold_controls_matching():
    """A pred box with IoU 1/3 is a miss+FP at the 0.5 threshold and a match at 1/3."""
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    # Shifted box: intersection 50, union 150 -> IoU = 1/3.
    pred = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 5.0, 10.0, 15.0]], [11])])]
    metadatas = [{"id": "v"}]

    strict = make_metric(iou_threshold=0.5)
    strict.update(pred, gt, metadatas)
    assert strict.compute()["mota"] == pytest.approx(-1.0)  # 1 miss + 1 FP over 1 gt box

    loose = make_metric(iou_threshold=1.0 / 3.0)
    loose.update(pred, gt, metadatas)
    assert loose.compute()["mota"] == pytest.approx(1.0)


def test_reset_clears_accumulated_videos():
    metric = make_metric()
    update_with_defaults(metric)
    metric.reset()
    with pytest.raises(MetricInputDataError, match="before any videos"):
        metric.compute()
    # The same video ids can be accumulated again after reset.
    update_with_defaults(metric)
    assert metric.compute()["mota"] == pytest.approx(EXPECTED_MOT_MOTA)


def test_compute_before_update_raises():
    with pytest.raises(MetricInputDataError, match="before any videos"):
        make_metric().compute()


def test_duplicate_video_id_raises():
    metric = make_metric()
    update_with_defaults(metric)
    with pytest.raises(MetricInputDataError, match="already accumulated"):
        update_with_defaults(metric)


def test_batch_length_mismatch_raises():
    with pytest.raises(MetricInputDataError, match="differ in length"):
        make_metric().update(
            DEFAULT_MOT_MODEL_PREDICTIONS[:1], DEFAULT_MOT_DATASET_TARGETS, DEFAULT_MOT_DATUM_METADATAS
        )


def test_metadata_length_mismatch_raises():
    with pytest.raises(MetricInputDataError, match="[Mm]etadata"):
        make_metric().update(
            DEFAULT_MOT_MODEL_PREDICTIONS, DEFAULT_MOT_DATASET_TARGETS, DEFAULT_MOT_DATUM_METADATAS[:1]
        )


def test_frame_count_mismatch_raises():
    truncated = [
        FakeMOTTarget(frame_tracks=DEFAULT_MOT_MODEL_PREDICTIONS[0].frame_tracks[:2]),
        DEFAULT_MOT_MODEL_PREDICTIONS[1],
    ]
    with pytest.raises(MetricInputDataError, match="frame targets"):
        make_metric().update(truncated, DEFAULT_MOT_DATASET_TARGETS, DEFAULT_MOT_DATUM_METADATAS)


def test_negative_ground_truth_track_id_raises():
    bad_gt = [
        FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [-1])]),
    ]
    pred = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="negative track_ids"):
        make_metric().update(pred, bad_gt, [{"id": "v"}])


def test_non_finite_boxes_raise():
    bad_pred = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, np.nan, 10.0]], [1])])]
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="non-finite"):
        make_metric().update(bad_pred, gt, [{"id": "v"}])


def test_track_id_shape_mismatch_raises():
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])
    bad.frame_tracks[0].track_ids = np.asarray([1, 2], dtype=np.int64)
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="track_ids shape"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_misshapen_boxes_raise_instead_of_being_reshaped():
    """A flat or wrongly-widthed box array must not be silently reinterpreted as N detections."""
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])
    bad.frame_tracks[0].boxes = np.asarray([0.0, 0.0, 10.0, 10.0, 5.0, 5.0, 15.0, 15.0])
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="boxes shape"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_boxes_with_wrong_column_count_raise():
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])
    bad.frame_tracks[0].boxes = np.asarray([[0.0, 0.0, 10.0, 10.0, 1.0, 2.0, 3.0, 4.0]])
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="boxes shape"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_non_integer_track_ids_raise():
    """Track ids are identities: truncating 1.9 to track 1 would silently merge two distinct tracks."""
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])
    bad.frame_tracks[0].track_ids = np.asarray([1.9])
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="integer-valued ids"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_integer_valued_float_track_ids_are_accepted():
    """Float arrays holding whole numbers are a normal model output and must still evaluate."""
    pred = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])
    pred.frame_tracks[0].track_ids = np.asarray([1.0])
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    metric = make_metric()
    metric.update([pred], gt, [{"id": "v"}])
    assert metric.compute()["mota"] == pytest.approx(1.0)


def test_empty_frames_are_valid():
    """Frames with no ground truth and no predictions contribute nothing but are legal."""
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([], []), frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    pred = [FakeMOTTarget(frame_tracks=[frame_tracks([], []), frame_tracks([[0.0, 0.0, 10.0, 10.0]], [9])])]
    metric = make_metric()
    metric.update(pred, gt, [{"id": "v"}])
    assert metric.compute()["mota"] == pytest.approx(1.0)


def test_invalid_constructor_arguments_raise():
    with pytest.raises(ValueError, match="iou_threshold"):
        make_metric(iou_threshold=0.0)
    with pytest.raises(ValueError, match="return_key"):
        make_metric(return_key="hota")


def test_factory_defaults_and_id_stability():
    metric = clear_mot_metrics_factory()
    assert metric.return_key == "mota"
    assert metric.metadata["id"] == clear_mot_metrics_factory().metadata["id"]
    assert metric.metadata["id"] != clear_mot_metrics_factory(iou_threshold=0.75).metadata["id"]


def test_factory_metric_computes_default_names():
    metric = clear_mot_metrics_factory()
    metric.update(DEFAULT_MOT_MODEL_PREDICTIONS, DEFAULT_MOT_DATASET_TARGETS, DEFAULT_MOT_DATUM_METADATAS)
    assert set(metric.compute().keys()) == set(DEFAULT_MOT_METRIC_NAMES)


def test_empty_boxes_with_wrong_column_count_raise():
    """An empty array still declares what a box is: (0, 5) disagrees and must not be normalized to (0, 4)."""
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([], [])])
    bad.frame_tracks[0].boxes = np.empty((0, 5))
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([], [])])]
    with pytest.raises(MetricInputDataError, match="boxes shape"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_large_integer_track_ids_stay_distinct():
    """Track ids routed through float64 collapse above 2**53, silently merging two identities into one."""
    box = [[0.0, 0.0, 10.0, 10.0]]
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks(box, [1]), frame_tracks(box, [1])])]
    # One ground-truth track, predicted under two different ids: an identity switch.
    pred = [FakeMOTTarget(frame_tracks=[frame_tracks(box, [2**53]), frame_tracks(box, [2**53 + 1])])]

    metric = make_metric()
    metric.update(pred, gt, [{"id": "v"}])

    assert (
        metric.compute()["num_switches"] == 1
    ), "2**53 and 2**53 + 1 were treated as the same track, so track ids are losing precision"


def test_non_numeric_track_ids_raise():
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])
    bad.frame_tracks[0].track_ids = np.asarray(["1"])
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="expected integer ids"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_unrepresentable_float_track_ids_raise():
    """A float id beyond 2**53 cannot round trip, so accepting it would silently change the identity."""
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])
    bad.frame_tracks[0].track_ids = np.asarray([float(2**54)])
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="not represented\n?.*exactly|are not represented"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_uint64_track_ids_beyond_int64_raise():
    """Unsigned ids above int64 cannot be represented, so they are rejected rather than wrapped."""
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])
    bad.frame_tracks[0].track_ids = np.asarray([2**63 + 5], dtype=np.uint64)
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="exceed the int64 range"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_track_ids_below_int64_raise():
    """A Python int below int64 is rejected as invalid input, not by an OverflowError from NumPy."""
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])
    bad.frame_tracks[0].track_ids = np.asarray([-(2**63) - 1], dtype=object)
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="below the int64 range"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_large_unsigned_track_ids_within_int64_are_accepted():
    pred = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])
    pred.frame_tracks[0].track_ids = np.asarray([2**63 - 1], dtype=np.uint64)
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    metric = make_metric()
    metric.update([pred], gt, [{"id": "v"}])
    assert metric.compute()["mota"] == pytest.approx(1.0)


def test_narrow_float_track_ids_are_rejected_before_they_can_merge():
    """float32 cannot tell 16777217 from 16777216, so ids that far out are refused rather than merged."""
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]] * 2, [1, 2])])
    bad.frame_tracks[0].track_ids = np.asarray([16777216, 16777217], dtype=np.float32)
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]] * 2, [1, 2])])]
    with pytest.raises(MetricInputDataError, match="not represented"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_repeated_identity_in_one_frame_raises():
    """A track holds at most one box per frame; a repeat is a labelling error or an aliased id."""
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], [5, 5])])
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="more than once in one frame"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_inverted_boxes_raise_instead_of_becoming_unmatchable():
    """Negative width or height makes every IoU NaN, which motmetrics scores as simply unmatched."""
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[10.0, 10.0, 0.0, 0.0]], [1])])
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="inverted boxes"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_numeric_looking_string_boxes_raise():
    """``np.asarray(..., dtype=float)`` would accept these, so the dtype is checked before conversion."""
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])
    bad.frame_tracks[0].boxes = np.asarray([["0", "0", "10", "10"]])
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="expected a numeric array"):
        make_metric().update([bad], gt, [{"id": "v"}])


def test_ragged_boxes_raise_as_metric_input_errors():
    """A ragged list raises inside numpy; it is reported in the same terms as every other bad input."""
    bad = FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])
    bad.frame_tracks[0].boxes = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0]]
    gt = [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])]
    with pytest.raises(MetricInputDataError, match="expected a numeric array"):
        make_metric().update([bad], gt, [{"id": "v"}])


@pytest.mark.skipif(not hasattr(np, "asfarray"), reason="motmetrics.distances.iou_matrix needs np.asfarray (NumPy < 2)")
def test_iou_distance_matrix_matches_motmetrics_iou_matrix():
    """The NumPy 2 replacement returns exactly what motmetrics' own iou_matrix does."""
    import motmetrics as mm

    from checkmaite.core.multiobject_tracking.metrics import _iou_distance_matrix

    rng = np.random.default_rng(0)
    gt = np.hstack([rng.uniform(0, 50, (6, 2)), rng.uniform(1, 20, (6, 2))])
    hyp = np.vstack([gt[:4] + rng.normal(0, 2, (4, 4)), rng.uniform(0, 50, (3, 4))])
    for max_distance in (0.5, 0.3, 1.0):
        np.testing.assert_array_equal(
            _iou_distance_matrix(gt, hyp, max_distance=max_distance),
            mm.distances.iou_matrix(gt, hyp, max_iou=max_distance),
        )
    for empty_gt, empty_hyp in ((gt[:0], hyp), (gt, hyp[:0])):
        np.testing.assert_array_equal(
            _iou_distance_matrix(empty_gt, empty_hyp, max_distance=0.5),
            mm.distances.iou_matrix(empty_gt, empty_hyp, max_iou=0.5),
        )


def test_overall_motp_ignores_videos_without_matches():
    """motmetrics' own OVERALL MOTP is NaN as soon as one video has no matches."""
    gt_box = [0.0, 0.0, 10.0, 10.0]
    # IoU distances: shifting by 1 gives 1 - 90/110 = 2/11; by 2 gives 1 - 80/120 = 1/3.
    gt = [
        FakeMOTTarget(frame_tracks=[frame_tracks([gt_box], [1])] * 2),
        FakeMOTTarget(frame_tracks=[frame_tracks([gt_box], [1])]),
        FakeMOTTarget(frame_tracks=[frame_tracks([gt_box], [1])]),
    ]
    preds = [
        FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 1.0, 10.0, 11.0]], [5])] * 2),
        FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 2.0, 10.0, 12.0]], [6])]),
        FakeMOTTarget(frame_tracks=[frame_tracks([[50.0, 50.0, 60.0, 60.0]], [7])]),  # matches nothing
    ]
    metric = make_metric()
    metric.update(preds, gt, [{"id": f"video_{i}"} for i in range(3)])

    assert metric.compute()["motp"] == pytest.approx((2 * 2 / 11 + 1 / 3) / 3)


def test_undefined_secondary_value_is_none():
    """MOTP is undefined when nothing matched; it is reported as None, never NaN."""
    metric = make_metric()
    metric.update(
        [FakeMOTTarget(frame_tracks=[frame_tracks([[50.0, 50.0, 60.0, 60.0]], [7])])],
        [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])])],
        [{"id": "v"}],
    )
    result = metric.compute()

    assert result["motp"] is None
    assert result["mota"] == pytest.approx(-1.0)


def test_undefined_headline_value_raises():
    """MOTA is undefined (-inf) with no ground-truth objects, so compute() refuses it."""
    metric = make_metric()
    metric.update(
        [FakeMOTTarget(frame_tracks=[frame_tracks([[0.0, 0.0, 10.0, 10.0]], [5])])],
        [FakeMOTTarget(frame_tracks=[frame_tracks([], [])])],
        [{"id": "v"}],
    )
    with pytest.raises(MetricInputDataError, match="'mota' is undefined for this evaluation"):
        metric.compute()


def test_compute_never_returns_non_finite_values():
    """Every value is a finite float or None, across all degenerate cases that have ground truth."""
    names = ["mota", "motp", "idf1", "num_switches", "precision", "recall"]
    cases = [
        ([frame_tracks([], [])], [frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])]),  # predicted nothing
        ([frame_tracks([[50.0, 50.0, 60.0, 60.0]], [7])], [frame_tracks([[0.0, 0.0, 10.0, 10.0]], [1])]),  # no match
    ]
    for pred_frames, gt_frames in cases:
        metric = make_metric(metric_names=names)
        metric.update([FakeMOTTarget(frame_tracks=pred_frames)], [FakeMOTTarget(frame_tracks=gt_frames)], [{"id": "v"}])
        for value in metric.compute().values():
            assert value is None or np.isfinite(value)
