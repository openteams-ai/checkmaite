import math

import numpy as np
import pytest
from maite.protocols import ModelMetadata
from modelmaite.object_detection import DetectionTarget

from checkmaite.core.object_detection.xaitk_explainable_capability import XaitkExplainable, XaitkExplainableConfig
from checkmaite.core.report._gradient import HAS_GRADIENT
from tests.conftest import DEFAULT_OD_DATASET_METADATA, DEFAULT_OD_MODEL_PREDICTIONS, FakeODDataset, FakeODModel
from tests.report_assertions import assert_inline_markdown_report

ARGS = {
    "name": "XaitkExplainable Example",
    "saliency_generator": {
        "type": "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack",
        "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack": {
            "n": 10,
            "s": 8,
            "p1": 0.5,
            "seed": 0,
            "threads": 4,
        },
    },
    "img_batch_size": 1,
}


@pytest.fixture
def test_config():
    return XaitkExplainableConfig(name=ARGS["name"], saliency_generator=ARGS["saliency_generator"])


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_run_and_collect(fake_od_model_default, fake_od_dataset_default, test_config):
    capability = XaitkExplainable()

    run_result = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        datasets=[fake_od_dataset_default],
        config=test_config,
    )

    assert run_result.model_dump()  # smoke test
    return run_result


@pytest.mark.optional
@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_run_and_collect_consumables(fake_od_model_default, fake_od_dataset_default, test_config):
    capability = XaitkExplainable()

    run_result = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        datasets=[fake_od_dataset_default],
        config=test_config,
    )

    slides = run_result.collect_report_consumables(threshold=0.5)

    assert len(slides) == len(fake_od_dataset_default) * len(fake_od_dataset_default[0][1].scores)


def test_run_and_collect_md(fake_od_model_default, fake_od_dataset_default, test_config):
    capability = XaitkExplainable()

    run_result = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        datasets=[fake_od_dataset_default],
        config=test_config,
    )

    report = run_result.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=run_result.capability_id)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_extract_returns_records(fake_od_model_default, fake_od_dataset_default, test_config) -> None:
    """extract() returns one XaitkExplainableRecord per detection."""
    from checkmaite.core._common.xaitk_explainable_capability import XaitkExplainableRecord

    capability = XaitkExplainable()
    run_result = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        datasets=[fake_od_dataset_default],
        config=test_config,
    )
    records = run_result.extract()
    assert len(records) > 0
    assert all(isinstance(r, XaitkExplainableRecord) for r in records)
    # OD-specific fields should be populated
    assert all(r.detection_index is not None for r in records)
    assert all(r.confidence is not None for r in records)
    assert all(r.image_id is not None for r in records)
    assert all(r.predicted_label is not None for r in records)
    # IC-specific field should be None
    assert all(r.gt_label is None for r in records)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_extract_saliency_stats_valid(fake_od_model_default, fake_od_dataset_default, test_config) -> None:
    """extract() produces valid saliency statistics."""
    capability = XaitkExplainable()
    run_result = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        datasets=[fake_od_dataset_default],
        config=test_config,
    )
    records = run_result.extract()
    for r in records:
        assert r.mean_saliency == r.mean_saliency  # not NaN
        assert 0.0 <= r.positive_saliency_ratio <= 1.0
        assert r.std_saliency >= 0.0
        assert r.confidence >= 0.0


RANDOM_GRID_ARGS = {
    "type": "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.RandomGridStack",
    "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.RandomGridStack": {
        "n": 5,
        "s": [40, 40],
        "p1": 0.5,
        "seed": 0,
        "threads": 0,
    },
}

OCCLUSION_ARGS = {
    "type": "xaitk_saliency.impls.gen_object_detector_blackbox_sal.occlusion_based.PerturbationOcclusion",
    "xaitk_saliency.impls.gen_object_detector_blackbox_sal.occlusion_based.PerturbationOcclusion": {
        "perturber": {
            "type": "xaitk_saliency.impls.perturb_image.sliding_window.SlidingWindow",
            "xaitk_saliency.impls.perturb_image.sliding_window.SlidingWindow": {
                "window_size": [100, 100],
                "stride": [100, 100],
            },
        },
        "generator": {
            "type": "xaitk_saliency.impls.gen_detector_prop_sal.drise_scoring.DRISEScoring",
            "xaitk_saliency.impls.gen_detector_prop_sal.drise_scoring.DRISEScoring": {},
        },
        "fill": [95, 96, 93],
        "threads": 0,
    },
}


def _small_od_dataset(dataset, n: int = 2):
    return type(dataset)(
        images=dataset.images[:n],
        targets=dataset.targets[:n],
        datum_metadata=dataset.datum_metadata[:n],
        dataset_metadata=dataset.metadata,
    )


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.parametrize(
    ("generator", "generator_type"),
    [(RANDOM_GRID_ARGS, "RandomGridStack"), (OCCLUSION_ARGS, "PerturbationOcclusion")],
)
def test_run_other_generators(fake_od_model_default, fake_od_dataset_default, generator, generator_type) -> None:
    """RandomGridStack and PerturbationOcclusion are reachable through config."""
    dataset = _small_od_dataset(fake_od_dataset_default)
    config = XaitkExplainableConfig(name=generator_type, saliency_generator=generator)
    run = XaitkExplainable().run(use_cache=False, models=[fake_od_model_default], datasets=[dataset], config=config)
    for datum, (image, _, _) in zip(run.outputs.results, dataset, strict=True):
        assert datum.sal_maps.shape == (len(datum.boxes), *np.asarray(image).shape[1:])
    records = run.extract()
    assert len(records) > 0
    assert all(r.saliency_generator_type == generator_type for r in records)
    report = run.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=run.capability_id)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_label_ids_not_zero_based(fake_od_model_default, fake_od_dataset_default, test_config) -> None:
    """Label ids that are not 0..N-1 give the same maps as the equivalent 0-based model."""
    offset = 10
    shifted_model = FakeODModel(
        model_metadata=ModelMetadata(
            id="fake_od_model_shifted",
            index2label={k + offset: v for k, v in fake_od_model_default.metadata["index2label"].items()},
        ),
        prediction_data=[
            DetectionTarget(boxes=p.boxes, labels=p.labels + offset, scores=p.scores)
            for p in DEFAULT_OD_MODEL_PREDICTIONS
        ],
    )
    dataset = _small_od_dataset(fake_od_dataset_default)

    base = XaitkExplainable().run(
        use_cache=False, models=[fake_od_model_default], datasets=[dataset], config=test_config
    )
    shifted = XaitkExplainable().run(use_cache=False, models=[shifted_model], datasets=[dataset], config=test_config)

    for base_datum, shifted_datum in zip(base.outputs.results, shifted.outputs.results, strict=True):
        np.testing.assert_allclose(shifted_datum.sal_maps, base_datum.sal_maps)
        np.testing.assert_array_equal(np.asarray(shifted_datum.labels), np.asarray(base_datum.labels) + offset)
    assert [r.predicted_label for r in shifted.extract()] == [r.predicted_label for r in base.extract()]


def test_unknown_predicted_label_raises(fake_od_model_default, fake_od_dataset_default, test_config) -> None:
    model = FakeODModel(
        model_metadata=ModelMetadata(id="fake_od_model_missing", index2label={0: "a", 1: "b"}),
    )
    with pytest.raises(ValueError, match="not in its index2label"):
        XaitkExplainable().run(
            use_cache=False,
            models=[model],
            datasets=[_small_od_dataset(fake_od_dataset_default, n=1)],
            config=test_config,
        )


def test_top_detections_orders_limits_and_filters() -> None:
    from checkmaite.core.object_detection.xaitk_explainable_capability import _top_detections

    predictions = DetectionTarget(
        boxes=np.arange(16, dtype=float).reshape(4, 4),
        labels=np.array([10, 11, 12, 10]),
        scores=np.array([0.2, 0.9, 0.5, 0.05]),
    )
    class_ids = [10, 11, 12]

    top = _top_detections(predictions, class_ids, min_score=None, dets_limit=3)
    np.testing.assert_array_equal(top.scores, [0.9, 0.5, 0.2])
    np.testing.assert_array_equal(top.labels, [11, 12, 10])
    np.testing.assert_array_equal(top.boxes, np.arange(16, dtype=float).reshape(4, 4)[[1, 2, 0]])

    confident = _top_detections(predictions, class_ids, min_score=0.5, dets_limit=10)
    np.testing.assert_array_equal(confident.scores, [0.9, 0.5])

    none_left = _top_detections(predictions, class_ids, min_score=0.95, dets_limit=10)
    assert len(none_left.labels) == 0
    assert np.asarray(none_left.boxes).shape == (0, 4)

    with pytest.raises(ValueError, match="not in its index2label"):
        _top_detections(predictions, [10, 11], min_score=None, dets_limit=10)

    per_class = DetectionTarget(
        boxes=np.zeros((2, 4)), labels=np.array([10, 11]), scores=np.array([[0.9, 0.1], [0.2, 0.8]])
    )
    with pytest.raises(ValueError, match="one score per detection"):
        _top_detections(per_class, class_ids, min_score=None, dets_limit=10)


def test_saliency_target_dataset_returns_positions(fake_od_dataset_default) -> None:
    from checkmaite.core.object_detection.xaitk_explainable_capability import _SaliencyTargetDataset

    class_ids = [10, 11, 12]
    image = np.asarray(fake_od_dataset_default[0][0])
    metadata = fake_od_dataset_default[0][2]
    dets = DetectionTarget(boxes=np.ones((2, 4)), labels=np.array([12, 10]), scores=np.array([0.9, 0.4]))

    sal_dataset = _SaliencyTargetDataset(image, dets, metadata, class_ids, "ds")
    assert len(sal_dataset) == 1
    out_image, target, out_metadata = sal_dataset[0]
    assert out_image is image
    assert out_metadata == metadata
    np.testing.assert_array_equal(target.labels, [2, 0])
    np.testing.assert_array_equal(target.boxes, dets.boxes)
    with pytest.raises(IndexError):
        sal_dataset[1]

    empty = DetectionTarget(boxes=np.zeros((0, 4)), labels=np.zeros(0, dtype=int), scores=np.zeros(0))
    with pytest.raises(ValueError, match="at least one detection"):
        _SaliencyTargetDataset(image, empty, metadata, class_ids, "ds")


class _ChangesOnEveryRead:
    """Wrap a dataset so every read returns a different image, like random augmentation."""

    def __init__(self, dataset) -> None:
        self._dataset = dataset
        self.metadata = dataset.metadata
        self.reads: list[int] = []

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        image, target, metadata = self._dataset[index]
        self.reads.append(index)
        shifted = (np.asarray(image).astype(int) + len(self.reads)) % 256
        return shifted.astype(np.uint8), target, metadata


class _RecordsUnperturbedInputs(FakeODModel):
    """The default fake model, recording the first image of each call."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[np.ndarray] = []

    def __call__(self, input_batch):
        self.inputs.append(np.asarray(input_batch[0]))
        return super().__call__(input_batch)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_each_image_is_read_once(fake_od_dataset_default, test_config) -> None:
    dataset = _ChangesOnEveryRead(_small_od_dataset(fake_od_dataset_default))
    model = _RecordsUnperturbedInputs()

    run = XaitkExplainable().run(use_cache=False, models=[model], datasets=[dataset], config=test_config)

    assert sorted(dataset.reads) == [0, 1]
    predicted_on = [img for img in model.inputs if any(np.array_equal(img, r.img) for r in run.outputs.results)]
    assert len(predicted_on) == 2
    for stored, seen in zip(run.outputs.results, predicted_on, strict=True):
        np.testing.assert_array_equal(stored.img, seen)


class _NoDetectionsOnImage(FakeODModel):
    """The default fake model, except that it detects nothing on one image."""

    def __init__(self, image) -> None:
        super().__init__()
        self._blank_image = np.asarray(image)

    def __call__(self, input_batch):
        predictions = super().__call__(input_batch)
        empty = DetectionTarget(boxes=np.zeros((0, 4)), labels=np.zeros(0, dtype=int), scores=np.zeros(0))
        return [
            empty if np.array_equal(np.asarray(img), self._blank_image) else pred
            for img, pred in zip(input_batch, predictions, strict=True)
        ]


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_image_without_detections_gets_no_maps(fake_od_dataset_default, test_config) -> None:
    small = _small_od_dataset(fake_od_dataset_default)
    # The fake images are identical, so make the blank one distinguishable.
    blank_image = np.asarray(small.images[1]) // 2
    dataset = type(small)(
        images=[small.images[0], blank_image],
        targets=small.targets,
        datum_metadata=small.datum_metadata,
        dataset_metadata=small.metadata,
    )
    model = _NoDetectionsOnImage(blank_image)

    run = XaitkExplainable().run(models=[model], datasets=[dataset], config=test_config)

    with_dets, without = run.outputs.results
    assert len(with_dets.sal_maps) > 0
    assert without.sal_maps.shape == (0, *np.asarray(dataset[1][0]).shape[-2:])
    assert len(without.boxes) == len(without.labels) == len(without.gt_saliency_coverage) == 0
    assert {r.image_index for r in run.extract()} == {0}
    assert "Image" in run.collect_md_report(threshold=0.5).content

    cached = XaitkExplainable().run(models=[model], datasets=[dataset], config=test_config)
    assert cached.outputs.results[1].sal_maps.shape == without.sal_maps.shape


class _DetectsOnlyUnperturbed(FakeODModel):
    """The default fake model on the dataset's own images; no detections on perturbed copies."""

    def __init__(self, images) -> None:
        super().__init__()
        self._images = [np.asarray(img) for img in images]

    def __call__(self, input_batch):
        predictions = super().__call__(input_batch)
        empty = DetectionTarget(boxes=np.zeros((0, 4)), labels=np.zeros(0, dtype=int), scores=np.zeros(0))
        return [
            pred if any(np.array_equal(np.asarray(img), known) for known in self._images) else empty
            for img, pred in zip(input_batch, predictions, strict=True)
        ]


def test_no_detections_on_perturbed_images_gives_zero_maps(fake_od_dataset_default, test_config, caplog) -> None:
    dataset = _small_od_dataset(fake_od_dataset_default, n=1)
    model = _DetectsOnlyUnperturbed(dataset.images)

    run = XaitkExplainable().run(use_cache=False, models=[model], datasets=[dataset], config=test_config)

    (datum,) = run.outputs.results
    assert len(datum.sal_maps) == len(datum.labels) > 0
    assert not np.any(datum.sal_maps)
    assert np.all(np.isnan(datum.saliency_entropy))
    assert np.all(np.isnan(datum.gt_saliency_coverage))
    assert "returned no maps" in caplog.text


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_min_score_and_dets_limit(fake_od_model_default, fake_od_dataset_default) -> None:
    dataset = _small_od_dataset(fake_od_dataset_default)
    all_scores = np.concatenate([np.asarray(p.scores) for p in DEFAULT_OD_MODEL_PREDICTIONS[:2]])
    min_score = float(np.median(all_scores))

    config = XaitkExplainableConfig(**ARGS, min_score=min_score, dets_limit=2)
    run = XaitkExplainable().run(use_cache=False, models=[fake_od_model_default], datasets=[dataset], config=config)

    for datum in run.outputs.results:
        assert len(datum.scores) <= 2
        assert np.all(np.asarray(datum.scores) >= min_score)
        assert len(datum.sal_maps) == len(datum.scores)
    assert f"if their score is at least {min_score}" in run.collect_md_report(threshold=0.5).content


class _RecordsInputShapes(FakeODModel):
    """The default fake model, recording the shape of every image it is called on."""

    def __init__(self) -> None:
        super().__init__()
        self.shapes: set[tuple[int, ...]] = set()

    def __call__(self, input_batch):
        self.shapes.update(np.asarray(img).shape for img in input_batch)
        return super().__call__(input_batch)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_saliency_max_side_runs_at_the_smaller_size(fake_od_dataset_default) -> None:
    dataset = _small_od_dataset(fake_od_dataset_default, n=1)
    full_image, target, _ = dataset[0]
    height, width = np.asarray(full_image).shape[-2:]  # 230 x 352
    model = _RecordsInputShapes()

    config = XaitkExplainableConfig(**ARGS, saliency_max_side=176)
    run = XaitkExplainable().run(use_cache=False, models=[model], datasets=[dataset], config=config)

    (datum,) = run.outputs.results
    assert datum.image_scale == pytest.approx(176 / width)
    assert datum.img.shape == (3, 115, 176)
    assert datum.img.dtype == np.asarray(full_image).dtype
    assert datum.sal_maps.shape[1:] == (115, 176)
    np.testing.assert_allclose(datum.gt_boxes, np.asarray(target.boxes) * 176 / width)
    assert model.shapes == {(3, 115, 176)}
    assert "longest side is at most 176 px" in run.collect_md_report(threshold=0.5).content

    unchanged = XaitkExplainable().run(
        use_cache=False,
        models=[FakeODModel()],
        datasets=[dataset],
        config=XaitkExplainableConfig(**ARGS, saliency_max_side=max(height, width)),
    )
    assert unchanged.outputs.results[0].image_scale == 1.0
    assert unchanged.outputs.results[0].img.shape == (3, height, width)


def test_resize_to_max_side_keeps_dtype_and_range() -> None:
    from checkmaite.core.object_detection.xaitk_explainable_capability import _resize_to_max_side

    image = np.full((3, 40, 20), 255, dtype=np.uint8)
    resized, scale, y_ratio, x_ratio = _resize_to_max_side(image, 10)
    assert scale == pytest.approx(0.25)
    assert (y_ratio, x_ratio) == (pytest.approx(0.25), pytest.approx(0.25))
    assert resized.shape == (3, 10, 5)
    assert resized.dtype == np.uint8
    assert np.all(resized == 255)

    floats = np.random.default_rng(0).random((1, 8, 8)).astype(np.float32)
    same, scale, y_ratio, x_ratio = _resize_to_max_side(floats, None)
    assert same is floats
    assert (scale, y_ratio, x_ratio) == (1.0, 1.0, 1.0)


def test_no_image_has_detections(fake_od_model_default, fake_od_dataset_default) -> None:
    config = XaitkExplainableConfig(**ARGS, min_score=2.0)
    run = XaitkExplainable().run(
        use_cache=False,
        models=[fake_od_model_default],
        datasets=[_small_od_dataset(fake_od_dataset_default)],
        config=config,
    )

    assert all(len(datum.sal_maps) == 0 for datum in run.outputs.results)
    assert run.extract() == []
    assert all(summary.count == 0 for summary in run.outputs.score_summary.values())
    run.collect_md_report(threshold=0.5)


def test_gt_scores_matching_and_coverage() -> None:
    from checkmaite.core.object_detection.xaitk_explainable_capability import _gt_scores

    # 20x20 maps; GT box 0 is the top-left quarter (a quarter of the image).
    sal_maps = np.zeros((5, 20, 20))
    sal_maps[0, 0:10, 0:10] = 1.0  # all above-average saliency inside GT box 0
    sal_maps[1, 0:10, 0:20] = 1.0  # half of it inside GT box 0
    sal_maps[1, 15:20, :] = -1.0  # below-average saliency is ignored
    sal_maps[2, :, :] = 1.0
    sal_maps[3, :, :] = 0.6  # a D-RISE-like floor everywhere...
    sal_maps[3, 0:10, 0:10] = 1.0  # ...with the excess only on the object
    sal_maps[4, :, :] = 0.8  # flat map: nothing above average
    det_boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 10, 10], [15, 15, 20, 20], [0, 0, 10, 10], [0, 0, 10, 10]], dtype=float
    )
    gt_boxes = np.array([[0, 0, 10, 10], [10, 0, 20, 10]], dtype=float)

    matched_idx, matched_iou, coverage, peak, distance = _gt_scores(sal_maps, det_boxes, gt_boxes, iou_threshold=0.5)

    np.testing.assert_array_equal(matched_idx, [0, 0, -1, 0, 0])
    assert matched_iou[0] == pytest.approx(1.0)
    assert matched_iou[1] == pytest.approx(81 / 100)
    assert np.isnan(matched_iou[2])
    assert coverage[0] == pytest.approx(1.0)
    assert peak[0] == 1.0
    assert coverage[1] == pytest.approx(0.5)
    assert peak[1] == 1.0  # the first maximum, at (0, 0), is inside box 0
    assert np.isnan(coverage[2])
    assert np.isnan(peak[2])
    # The floor does not count: without removing the mean, coverage would be 0.4 / (0.4 + 0.6 * 0.75) = 0.47.
    assert coverage[3] == pytest.approx(1.0)
    assert peak[3] == 1.0
    assert np.isnan(coverage[4])
    assert np.isnan(peak[4])
    # Hits are 0 px from the box; missing scores are NaN in the same places as peak-in-box.
    np.testing.assert_array_equal(distance[[0, 1, 3]], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(np.isnan(distance), np.isnan(peak))

    # A miss: the detection matches GT box 1, but the map peaks in the bottom-left corner.
    miss_map = np.zeros((1, 20, 20))
    miss_map[0, 18, 1] = 1.0
    _, _, miss_coverage, miss_peak, miss_distance = _gt_scores(
        miss_map, np.array([[10, 0, 20, 10]], dtype=float), gt_boxes, iou_threshold=0.5
    )
    assert miss_coverage[0] == pytest.approx(0.0)
    assert miss_peak[0] == 0.0
    # Box 1 covers pixels x 10..19, y 0..9, so the nearest one to (y=18, x=1) is (y=9, x=10): 9 px each way.
    assert miss_distance[0] == pytest.approx(math.hypot(9, 9))

    # Raising the threshold drops the weaker match.
    matched_idx, _, coverage, peak, distance = _gt_scores(sal_maps, det_boxes, gt_boxes, iou_threshold=0.9)
    np.testing.assert_array_equal(matched_idx, [0, -1, -1, 0, 0])
    assert np.isnan(coverage[1])
    assert np.isnan(peak[1])
    assert np.isnan(distance[1])

    # No ground truth: nothing matches.
    matched_idx, _, _, _, distance = _gt_scores(sal_maps, det_boxes, np.zeros((0, 4)), iou_threshold=0.5)
    np.testing.assert_array_equal(matched_idx, [-1, -1, -1, -1, -1])
    assert np.all(np.isnan(distance))


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_scores_in_outputs_and_records(fake_od_model_default, fake_od_dataset_default, test_config) -> None:
    from xaitk_saliency.impls.saliency_metric.entropy import Entropy

    dataset = _small_od_dataset(fake_od_dataset_default)
    run = XaitkExplainable().run(
        use_cache=False, models=[fake_od_model_default], datasets=[dataset], config=test_config
    )

    datum = run.outputs.results[0]
    for det_idx, sal_map in enumerate(datum.sal_maps):
        assert datum.saliency_entropy[det_idx] == pytest.approx(Entropy()(sal_map))
    # Detections are sorted by score. Detection 0 (0.9) overlaps GT box 0 ("date"), detection 1 (0.7) is
    # exactly GT box 2 with the wrong label, and detection 2 (0.2) overlaps nothing.
    np.testing.assert_array_equal(datum.matched_gt_index, [0, 2, -1])
    assert datum.matched_gt_iou[1] == pytest.approx(1.0)
    # The fake model ignores the masks, so every map is flat: nothing is above average, and even
    # matched detections have no coverage or peak-in-box score.
    assert np.all(np.isnan(datum.gt_saliency_coverage))
    assert np.all(np.isnan(datum.gt_saliency_peak_in_box))
    assert np.all(np.isnan(datum.gt_saliency_peak_distance))

    records = run.extract()
    by_det = {r.detection_index: r for r in records if r.image_index == 0}
    assert by_det[0].matched_gt_index == 0
    assert by_det[1].matched_gt_iou == pytest.approx(1.0)
    assert by_det[2].matched_gt_index is None
    assert by_det[2].gt_saliency_coverage is None
    assert by_det[2].gt_saliency_peak_in_box is None
    assert by_det[2].gt_saliency_peak_distance is None
    assert all(r.saliency_entropy is not None for r in records)

    summary = run.outputs.score_summary
    assert set(summary) == {
        "saliency_entropy",
        "normalized_saliency_entropy",
        "matched_gt_iou",
        "gt_saliency_coverage",
        "gt_saliency_peak_in_box",
        "gt_saliency_peak_distance",
    }
    assert summary["gt_saliency_peak_in_box"].count == summary["gt_saliency_coverage"].count
    assert summary["gt_saliency_peak_distance"].count == summary["gt_saliency_coverage"].count
    assert summary["saliency_entropy"].count == len(records)
    assert summary["gt_saliency_coverage"].count == sum(r.gt_saliency_coverage is not None for r in records)
    coverages = [r.gt_saliency_coverage for r in records if r.gt_saliency_coverage is not None]
    assert coverages == []  # flat maps, see above
    assert summary["gt_saliency_coverage"].count == 0
    assert summary["gt_saliency_coverage"].mean is None

    report = run.collect_md_report(threshold=0.5)
    assert "Saliency scores" in report.content
    assert "Matched GT box" in report.content


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_saliency_max_side_scales_gt_per_axis() -> None:
    """GT boxes are scaled by each axis's actual resize ratio, which rounding can move away from the nominal one."""
    # 101 x 200 at saliency_max_side=100 becomes 50 x 100: x scales by 0.5 but y by 50/101, not 0.5.
    gt_box = np.array([[96.0, 97.0, 100.0, 101.0]])  # a small box at the bottom edge
    dataset = FakeODDataset(
        images=[np.full((3, 101, 200), 128, dtype=np.uint8)],
        targets=[DetectionTarget(boxes=gt_box, labels=np.array([1]), scores=np.array([1.0]))],
        datum_metadata=[{"id": "edge"}],
        dataset_metadata=DEFAULT_OD_DATASET_METADATA,
    )
    resized_box = gt_box * np.array([100 / 200, 50 / 101, 100 / 200, 50 / 101])
    model = FakeODModel(
        model_metadata=ModelMetadata(id="edge_model", index2label={0: "a", 1: "b"}),
        prediction_data=[DetectionTarget(boxes=resized_box, labels=np.array([1]), scores=np.array([0.9]))],
    )

    config = XaitkExplainableConfig(**ARGS, saliency_max_side=100)
    run = XaitkExplainable().run(use_cache=False, models=[model], datasets=[dataset], config=config)

    (datum,) = run.outputs.results
    assert datum.img.shape == (3, 50, 100)
    assert datum.image_scale == pytest.approx(0.5)
    assert datum.matched_gt_index[0] == 0
    assert datum.matched_gt_iou[0] == pytest.approx(1.0)
    np.testing.assert_allclose(datum.gt_boxes, resized_box)


def test_gt_scores_ignore_non_finite_maps() -> None:
    """A NaN map, as D-RISE gives a zero-confidence detection, gets no coverage or pointing-game score."""
    from checkmaite.core._common.xaitk_explainable_capability import XaitkScoreSummary, saliency_entropy
    from checkmaite.core.object_detection.xaitk_explainable_capability import _gt_scores

    sal_maps = np.full((1, 20, 20), np.nan)
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=float)

    matched_idx, matched_iou, coverage, peak, distance = _gt_scores(sal_maps, gt_boxes, gt_boxes, iou_threshold=0.5)

    # The box match still stands; only the saliency scores are undefined.
    assert matched_idx[0] == 0
    assert matched_iou[0] == pytest.approx(1.0)
    assert np.isnan(coverage[0])
    assert np.isnan(peak[0])
    assert np.isnan(distance[0])
    assert XaitkScoreSummary.from_values(peak).count == 0
    assert saliency_entropy(sal_maps[0]) == (None, None)


def test_resize_to_max_side_accepts_flipped_views() -> None:
    from checkmaite.core.object_detection.xaitk_explainable_capability import _resize_to_max_side

    image = np.arange(3 * 40 * 20, dtype=np.uint8).reshape(3, 40, 20)
    flipped = image[:, :, ::-1]  # negative strides
    resized = _resize_to_max_side(flipped, 10)[0]
    assert resized.shape == (3, 10, 5)
    assert resized.dtype == np.uint8
