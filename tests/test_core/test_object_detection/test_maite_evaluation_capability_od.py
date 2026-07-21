import numpy as np
import pytest
import torch
from pydantic import ValidationError

from checkmaite.core._common.maite_evaluation_capability import MaiteEvaluationConfig as BaseMaiteEvaluationConfig
from checkmaite.core.object_detection.dataset_loaders import DetectionTarget
from checkmaite.core.object_detection.maite_evaluation_capability import (
    MaiteEvaluation,
    MaiteEvaluationConfig,
    MaiteEvaluationRun,
)
from checkmaite.core.object_detection.metrics import multiclass_map50_torch_metric_factory
from checkmaite.core.report._gradient import HAS_GRADIENT


class CountingODModel:
    """Protocol-compatible OD model wrapper that records inference batches."""

    def __init__(self, model, *, model_id: str):
        self._model = model
        self.metadata = {**model.metadata, "id": model_id}
        self.batch_sizes = []

    def __call__(self, inputs):
        self.batch_sizes.append(len(inputs))
        return self._model(inputs)


@pytest.fixture
def test_run(fake_od_model_default, fake_od_dataset_default, fake_od_metric_default):
    capability = MaiteEvaluation()

    output = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
    )

    assert output.model_dump()  # smoke test
    return output


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_collect_report_consumables(test_run):
    assert test_run.collect_report_consumables(threshold=0.5)  # smoke test


def test_collect_md_report(test_run):
    assert test_run.collect_md_report(threshold=0.5)  # smoke test


def test_batch_size_controls_inference_batches(fake_od_model_default, fake_od_dataset_default, fake_od_metric_default):
    model = CountingODModel(fake_od_model_default, model_id="batch-size-capability-model")
    MaiteEvaluation().run(
        use_cache=False,
        models=[model],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
        config=MaiteEvaluationConfig(batch_size=4),
    )

    assert model.batch_sizes == [4, 2]


def _postprocess_one(prediction, config):
    return MaiteEvaluation()._cpu_postprocess_predictions([[prediction]], config)[0][0]


def test_default_config_skips_postprocessing():
    postprocessor, postprocessor_id = MaiteEvaluation()._cpu_prediction_postprocessor(MaiteEvaluationConfig())

    assert postprocessor is None
    assert postprocessor_id is None


def test_class_agnostic_nms_requires_nms_threshold():
    with pytest.raises(ValidationError, match="class_agnostic_nms=True requires nms_iou_threshold"):
        MaiteEvaluationConfig(class_agnostic_nms=True)


def test_default_evaluation_passes_model_predictions_unchanged(fake_od_model_default, fake_od_dataset_default):
    class CapturingMetric:
        metadata = {"id": "default-no-postprocessing-metric"}
        return_key = "count"

        def reset(self):
            self.predictions = []

        def update(self, preds, targets, metadatas):
            self.predictions.extend(preds)

        def compute(self):
            return {"count": len(self.predictions)}

    expected_prediction = fake_od_model_default([fake_od_dataset_default[0][0]])[0]
    metric = CapturingMetric()
    MaiteEvaluation().run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[metric],
        datasets=[fake_od_dataset_default],
        config=MaiteEvaluationConfig(),
    )

    assert metric.predictions
    assert all(prediction is expected_prediction for prediction in metric.predictions)


def test_object_detection_postprocessing_confidence_only():
    prediction = DetectionTarget(
        boxes=torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]),
        labels=torch.tensor([1, 2, 3]),
        scores=torch.tensor([0.9, 0.4, 0.8]),
    )

    processed = _postprocess_one(prediction, MaiteEvaluationConfig(confidence_threshold=0.5))

    assert len(processed.scores) == 2
    np.testing.assert_allclose(processed.scores, np.asarray([0.9, 0.8]))


def test_confidence_filter_preserves_model_order_without_detection_limit():
    prediction = DetectionTarget(
        boxes=torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]]),
        labels=torch.tensor([1, 2]),
        scores=torch.tensor([0.5, 0.9]),
    )

    processed = _postprocess_one(prediction, MaiteEvaluationConfig(confidence_threshold=0.4))

    np.testing.assert_allclose(processed.scores, np.asarray([0.5, 0.9]))


def test_object_detection_postprocessing_nms_only():
    prediction = DetectionTarget(
        boxes=torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]]),
        labels=torch.tensor([1, 1, 2]),
        scores=torch.tensor([0.9, 0.8, 0.7]),
    )

    processed = _postprocess_one(prediction, MaiteEvaluationConfig(nms_iou_threshold=0.5))

    assert len(processed.scores) == 2
    np.testing.assert_array_equal(processed.labels, np.asarray([1, 2]))


def test_object_detection_postprocessing_class_agnostic_nms():
    prediction = DetectionTarget(
        boxes=torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]]),
        labels=torch.tensor([1, 2, 3]),
        scores=torch.tensor([0.9, 0.8, 0.7]),
    )

    processed = _postprocess_one(
        prediction,
        MaiteEvaluationConfig(nms_iou_threshold=0.5, class_agnostic_nms=True),
    )

    assert len(processed.scores) == 1
    np.testing.assert_array_equal(processed.labels, np.asarray([1]))


def test_object_detection_postprocessing_max_detections_only():
    prediction = DetectionTarget(
        boxes=torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]),
        labels=torch.tensor([1, 2, 3]),
        scores=torch.tensor([0.1, 0.9, 0.5]),
    )

    processed = _postprocess_one(prediction, MaiteEvaluationConfig(max_detections=2))

    assert len(processed.scores) == 2
    np.testing.assert_allclose(processed.scores, np.asarray([0.9, 0.5]))


def test_cpu_postprocessing_does_not_mutate_raw_predictions():
    prediction = DetectionTarget(
        boxes=np.asarray([[0, 0, 10, 10], [20, 20, 30, 30]]),
        labels=np.asarray([1, 2]),
        scores=np.asarray([0.9, 0.4]),
    )
    original_boxes = prediction.boxes.copy()
    original_labels = prediction.labels.copy()
    original_scores = prediction.scores.copy()

    _postprocess_one(prediction, MaiteEvaluationConfig(confidence_threshold=0.5))

    np.testing.assert_array_equal(prediction.boxes, original_boxes)
    np.testing.assert_array_equal(prediction.labels, original_labels)
    np.testing.assert_array_equal(prediction.scores, original_scores)


def test_object_detection_postprocessing_supports_mixed_cpu_arraylike_fields():
    prediction = DetectionTarget(
        boxes=torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]]),
        labels=np.asarray([1, 2]),
        scores=[0.9, 0.4],
    )

    processed = _postprocess_one(prediction, MaiteEvaluationConfig(confidence_threshold=0.5))

    np.testing.assert_array_equal(processed.boxes, np.asarray([[0, 0, 10, 10]]))
    np.testing.assert_array_equal(processed.labels, np.asarray([1]))
    np.testing.assert_allclose(processed.scores, np.asarray([0.9]))


def test_object_detection_postprocessing_supports_numpy_targets():
    prediction = DetectionTarget(
        boxes=np.asarray([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]]),
        labels=np.asarray([1, 1, 2]),
        scores=np.asarray([0.9, 0.8, 0.7]),
    )

    processed = _postprocess_one(prediction, MaiteEvaluationConfig(nms_iou_threshold=0.5))

    np.testing.assert_array_equal(processed.labels, np.asarray([1, 2]))
    np.testing.assert_allclose(processed.scores, np.asarray([0.9, 0.7]))


@pytest.mark.parametrize("array_backend", ["torch", "numpy"])
def test_object_detection_postprocessing_supports_class_score_matrix(array_backend):
    boxes = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
    labels = [0, 1, 1]
    scores = [[0.9, 0.1], [0.2, 0.8], [0.3, 0.4]]
    array = torch.tensor if array_backend == "torch" else np.asarray
    prediction = DetectionTarget(boxes=array(boxes), labels=array(labels), scores=array(scores))

    processed = _postprocess_one(prediction, MaiteEvaluationConfig(confidence_threshold=0.5))

    expected_scores = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    np.testing.assert_allclose(processed.scores, expected_scores)


def test_class_score_matrix_uses_max_score_through_full_postprocessing_chain():
    prediction = DetectionTarget(
        boxes=np.asarray(
            [
                [40, 40, 50, 50],
                [0, 0, 10, 10],
                [1, 1, 11, 11],
                [20, 20, 30, 30],
            ]
        ),
        labels=np.asarray([1, 1, 1, 2]),
        scores=np.asarray([[0.3, 0.4], [0.9, 0.1], [0.2, 0.8], [0.7, 0.1]]),
    )

    processed = _postprocess_one(
        prediction,
        MaiteEvaluationConfig(confidence_threshold=0.5, nms_iou_threshold=0.5, max_detections=1),
    )

    np.testing.assert_array_equal(processed.boxes, np.asarray([[0, 0, 10, 10]]))
    np.testing.assert_array_equal(processed.labels, np.asarray([1]))
    np.testing.assert_allclose(processed.scores, np.asarray([[0.9, 0.1]]))


def test_object_detection_postprocessing_accepts_nonstandard_constructor_target():
    class NonstandardConstructorTarget:
        def __init__(self, token):
            self.token = token
            self.boxes = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]])
            self.labels = torch.tensor([1, 2])
            self.scores = torch.tensor([0.9, 0.4])

    processed = _postprocess_one(
        NonstandardConstructorTarget("sentinel"), MaiteEvaluationConfig(confidence_threshold=0.5)
    )

    assert isinstance(processed, DetectionTarget)
    assert not hasattr(processed, "token")
    assert all(isinstance(value, np.ndarray) for value in (processed.boxes, processed.labels, processed.scores))
    np.testing.assert_allclose(processed.scores, np.asarray([0.9]))


def test_evaluate_return_preds_returns_cpu_postprocessed_predictions(
    fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
):
    from checkmaite import cached_tasks

    capability = MaiteEvaluation()
    postprocessor, postprocessor_id = capability._cpu_prediction_postprocessor(
        MaiteEvaluationConfig(confidence_threshold=0.8)
    )
    _, predictions, _ = cached_tasks.evaluate(
        model=fake_od_model_default,
        dataset=fake_od_dataset_default,
        metric=fake_od_metric_default,
        return_preds=True,
        cpu_prediction_postprocessor=postprocessor,
        cpu_prediction_postprocessor_id=postprocessor_id,
        use_cache=False,
    )

    assert predictions
    assert all(len(target.scores) == 1 for batch in predictions for target in batch)
    assert all(isinstance(target.scores, np.ndarray) for batch in predictions for target in batch)


def test_cpu_prediction_postprocessor_id_is_stable():
    capability = MaiteEvaluation()
    config = MaiteEvaluationConfig(confidence_threshold=0.5, nms_iou_threshold=0.4, max_detections=10)

    _, first_id = capability._cpu_prediction_postprocessor(config)
    _, second_id = capability._cpu_prediction_postprocessor(config.model_copy())

    assert first_id == second_id


def test_cpu_prediction_postprocessor_id_ignores_batch_size():
    capability = MaiteEvaluation()

    _, batch_one_id = capability._cpu_prediction_postprocessor(
        MaiteEvaluationConfig(batch_size=1, confidence_threshold=0.5)
    )
    _, batch_four_id = capability._cpu_prediction_postprocessor(
        MaiteEvaluationConfig(batch_size=4, confidence_threshold=0.5)
    )

    assert batch_one_id == batch_four_id


@pytest.mark.parametrize(
    "changed_config",
    [
        MaiteEvaluationConfig(confidence_threshold=0.6, nms_iou_threshold=0.4, max_detections=10),
        MaiteEvaluationConfig(confidence_threshold=0.5, nms_iou_threshold=0.6, max_detections=10),
        MaiteEvaluationConfig(
            confidence_threshold=0.5,
            nms_iou_threshold=0.4,
            class_agnostic_nms=True,
            max_detections=10,
        ),
        MaiteEvaluationConfig(confidence_threshold=0.5, nms_iou_threshold=0.4, max_detections=5),
    ],
)
def test_each_postprocessing_setting_changes_postprocessor_id(changed_config):
    capability = MaiteEvaluation()
    baseline = MaiteEvaluationConfig(confidence_threshold=0.5, nms_iou_threshold=0.4, max_detections=10)

    _, baseline_id = capability._cpu_prediction_postprocessor(baseline)
    _, changed_id = capability._cpu_prediction_postprocessor(changed_config)

    assert changed_id != baseline_id


def test_cpu_prediction_postprocessor_id_includes_future_od_config_fields():
    class FutureMaiteEvaluationConfig(MaiteEvaluationConfig):
        future_postprocessing_setting: bool = False

    capability = MaiteEvaluation()
    _, disabled_id = capability._cpu_prediction_postprocessor(
        FutureMaiteEvaluationConfig(confidence_threshold=0.5, future_postprocessing_setting=False)
    )
    _, enabled_id = capability._cpu_prediction_postprocessor(
        FutureMaiteEvaluationConfig(confidence_threshold=0.5, future_postprocessing_setting=True)
    )

    assert disabled_id != enabled_id


@pytest.mark.parametrize(
    "changed_config",
    [
        MaiteEvaluationConfig(batch_size=2),
        MaiteEvaluationConfig(confidence_threshold=0.5),
        MaiteEvaluationConfig(nms_iou_threshold=0.5),
        MaiteEvaluationConfig(max_detections=1),
    ],
)
def test_inference_settings_change_run_uid(
    changed_config, fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
):
    capability = MaiteEvaluation()
    default_run = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
        config=MaiteEvaluationConfig(),
    )
    changed_run = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
        config=changed_config,
    )

    assert changed_run.run_uid != default_run.run_uid


def test_class_agnostic_nms_changes_run_uid(fake_od_model_default, fake_od_dataset_default, fake_od_metric_default):
    capability = MaiteEvaluation()
    class_aware_run = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
        config=MaiteEvaluationConfig(nms_iou_threshold=0.5),
    )
    class_agnostic_run = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
        config=MaiteEvaluationConfig(nms_iou_threshold=0.5, class_agnostic_nms=True),
    )

    assert class_agnostic_run.run_uid != class_aware_run.run_uid


def test_object_detection_rejects_base_config(fake_od_model_default, fake_od_dataset_default, fake_od_metric_default):
    with pytest.raises(TypeError, match="requires an object-detection MaiteEvaluationConfig"):
        MaiteEvaluation().run(
            use_cache=False,
            models=[fake_od_model_default],
            metrics=[fake_od_metric_default],
            datasets=[fake_od_dataset_default],
            config=BaseMaiteEvaluationConfig(),
        )


def test_object_detection_run_cache_round_trip_preserves_config_type_fields_and_uid(
    fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
):
    config = MaiteEvaluationConfig(
        batch_size=2,
        confidence_threshold=0.5,
        nms_iou_threshold=0.4,
        class_agnostic_nms=True,
        max_detections=1,
    )
    capability = MaiteEvaluation()

    first_run = capability.run(
        use_cache=True,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
        config=config,
    )
    cached_run = capability.run(
        use_cache=True,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
        config=config,
    )

    assert cached_run is not first_run
    assert isinstance(cached_run, MaiteEvaluationRun)
    assert isinstance(cached_run.config, MaiteEvaluationConfig)
    assert cached_run.config.model_dump() == config.model_dump()
    assert cached_run.run_uid == first_run.run_uid


def test_inference_settings_are_validated():
    assert "multi_label_nms" not in MaiteEvaluationConfig.model_fields

    with pytest.raises(ValidationError):
        MaiteEvaluationConfig(batch_size=0)
    with pytest.raises(ValidationError):
        MaiteEvaluationConfig(confidence_threshold=1.1)
    with pytest.raises(ValidationError):
        MaiteEvaluationConfig(max_detections=0)


def test_multiclass(fake_od_model_default, fake_od_dataset_default):
    capability = MaiteEvaluation()

    metric = multiclass_map50_torch_metric_factory()

    output = capability.run(
        use_cache=False, models=[fake_od_model_default], metrics=[metric], datasets=[fake_od_dataset_default]
    )

    assert output.outputs.class_metrics is not None
