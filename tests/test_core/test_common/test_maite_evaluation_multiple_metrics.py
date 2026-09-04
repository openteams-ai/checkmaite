import pickle
from types import MappingProxyType

import pytest

from checkmaite.core._common.metric_fanout import MaiteEvaluationMetricError, _MetricFanout
from checkmaite.core.capability_core import Number
from checkmaite.core.image_classification import MaiteEvaluationMetricError as ExportedMaiteEvaluationMetricError
from checkmaite.core.image_classification.maite_evaluation_capability import MaiteEvaluation, MaiteEvaluationConfig


class CountingModel:
    def __init__(self, model, *, model_id="counting-multi-metric-model"):
        self._model = model
        self.metadata = {**model.metadata, "id": model_id}
        self.calls = 0

    def __call__(self, inputs):
        self.calls += 1
        return self._model(inputs)


class RecordingMetric:
    def __init__(self, metric_id, result, *, return_key=None, fail_stage=None):
        self.metadata = {"id": metric_id}
        self.result = result
        self.fail_stage = fail_stage
        self.reset_calls = 0
        self.update_calls = 0
        self.compute_calls = 0
        self.prediction_batch_ids = []
        self.target_batch_ids = []
        self.metadata_batch_ids = []
        if return_key is not None:
            self.return_key = return_key

    def _fail(self, stage):
        if self.fail_stage == stage:
            raise ValueError(f"intentional {stage} failure")

    def reset(self):
        self.reset_calls += 1
        self._fail("reset")

    def update(self, predictions, targets, metadata):
        self.update_calls += 1
        self.prediction_batch_ids.append(id(predictions))
        self.target_batch_ids.append(id(targets))
        self.metadata_batch_ids.append(id(metadata))
        self._fail("update")

    def compute(self):
        self.compute_calls += 1
        self._fail("compute")
        return self.result


def test_maite_evaluation_supports_one_or_more_metrics(fake_ic_model_default, fake_ic_dataset_default):
    capability = MaiteEvaluation()

    assert capability.supports_metrics is Number.MANY
    assert ExportedMaiteEvaluationMetricError is MaiteEvaluationMetricError
    with pytest.raises(TypeError, match="requires at least 1 metric"):
        capability.run(
            use_cache=False,
            models=[fake_ic_model_default],
            datasets=[fake_ic_dataset_default],
            metrics=[],
        )


def test_multiple_metrics_share_one_uncached_inference_pass(
    fake_ic_model_default,
    fake_ic_dataset_default,
):
    model = CountingModel(fake_ic_model_default)
    metric_b = RecordingMetric("b-metric", {"score_b": 0.2}, return_key="score_b")
    metric_a = RecordingMetric("a-metric", {"score_a": 0.1}, return_key="score_a")

    run = MaiteEvaluation().run(
        use_cache=False,
        models=[model],
        datasets=[fake_ic_dataset_default],
        metrics=[metric_b, metric_a],
        config=MaiteEvaluationConfig(batch_size=4),
    )

    assert model.calls == 5
    assert metric_a.reset_calls == metric_b.reset_calls == 1
    assert metric_a.update_calls == metric_b.update_calls == 5
    assert metric_a.compute_calls == metric_b.compute_calls == 1
    assert metric_a.prediction_batch_ids == metric_b.prediction_batch_ids
    assert metric_a.target_batch_ids == metric_b.target_batch_ids
    assert metric_a.metadata_batch_ids == metric_b.metadata_batch_ids
    assert list(run.outputs.metrics) == ["a-metric", "b-metric"]
    assert run.outputs.metrics["a-metric"].overall_metric_value == 0.1
    assert run.outputs.metrics["b-metric"].overall_metric_value == 0.2


def test_fanout_preserves_member_mapping_type():
    member_result = MappingProxyType({"score": 0.1})
    metric = RecordingMetric("custom-mapping", member_result)

    result = _MetricFanout([metric]).compute()

    assert result["custom-mapping"] is member_result


def test_metric_order_does_not_change_fanout_or_run_identity(fake_ic_model_default, fake_ic_dataset_default):
    metric_a = RecordingMetric("a-metric", {"a": 0.1}, return_key="a")
    metric_b = RecordingMetric("b-metric", {"b": 0.2}, return_key="b")

    forward_fanout = _MetricFanout([metric_a, metric_b])
    reverse_fanout = _MetricFanout([metric_b, metric_a])
    capability = MaiteEvaluation()
    forward_run = capability.run(
        use_cache=False,
        models=[fake_ic_model_default],
        datasets=[fake_ic_dataset_default],
        metrics=[metric_a, metric_b],
    )
    reverse_run = capability.run(
        use_cache=False,
        models=[fake_ic_model_default],
        datasets=[fake_ic_dataset_default],
        metrics=[metric_b, metric_a],
    )

    assert forward_fanout.metadata["id"] == reverse_fanout.metadata["id"]
    assert forward_run.run_uid == reverse_run.run_uid
    assert forward_run.outputs == reverse_run.outputs
    assert forward_run.metric_metadata == [{"id": "a-metric"}, {"id": "b-metric"}]


def test_changed_metric_collection_reuses_prediction_cache(fake_ic_model_default, fake_ic_dataset_default):
    model = CountingModel(fake_ic_model_default, model_id="changed-collection-cache-model")
    metric_a = RecordingMetric("a-prediction-cache-metric", {"a": 0.1}, return_key="a")
    metric_b = RecordingMetric("b-prediction-cache-metric", {"b": 0.2}, return_key="b")
    capability = MaiteEvaluation()

    capability.run(
        use_cache=True,
        models=[model],
        datasets=[fake_ic_dataset_default],
        metrics=[metric_a],
        config=MaiteEvaluationConfig(batch_size=5),
    )
    first_model_calls = model.calls
    run = capability.run(
        use_cache=True,
        models=[model],
        datasets=[fake_ic_dataset_default],
        metrics=[metric_a, metric_b],
        config=MaiteEvaluationConfig(batch_size=5),
    )

    assert first_model_calls == 4
    assert model.calls == first_model_calls
    assert list(run.outputs.metrics) == ["a-prediction-cache-metric", "b-prediction-cache-metric"]


def test_reordered_metrics_share_capability_cache(fake_ic_model_default, fake_ic_dataset_default):
    model = CountingModel(fake_ic_model_default, model_id="reordered-cache-model")
    metric_a = RecordingMetric("a-cache-metric", {"a": 0.1}, return_key="a")
    metric_b = RecordingMetric("b-cache-metric", {"b": 0.2}, return_key="b")
    capability = MaiteEvaluation()

    first = capability.run(
        use_cache=True,
        models=[model],
        datasets=[fake_ic_dataset_default],
        metrics=[metric_b, metric_a],
        config=MaiteEvaluationConfig(batch_size=5),
    )
    model_calls = model.calls
    second = capability.run(
        use_cache=True,
        models=[model],
        datasets=[fake_ic_dataset_default],
        metrics=[metric_a, metric_b],
        config=MaiteEvaluationConfig(batch_size=5),
    )

    assert model_calls == 4
    assert model.calls == model_calls
    assert second.run_uid == first.run_uid
    assert second.outputs == first.outputs


@pytest.mark.parametrize("invalid_id", [None, "", " ", 1])
def test_invalid_metric_id_fails_before_inference(fake_ic_model_default, fake_ic_dataset_default, invalid_id):
    model = CountingModel(fake_ic_model_default, model_id=f"invalid-id-model-{invalid_id!r}")
    metric = RecordingMetric("temporary", {"score": 1.0})
    metric.metadata = {"id": invalid_id}

    with pytest.raises(ValueError, match="non-empty string"):
        MaiteEvaluation().run(models=[model], datasets=[fake_ic_dataset_default], metrics=[metric], use_cache=False)

    assert model.calls == 0


def test_missing_metric_id_fails_before_inference(fake_ic_model_default, fake_ic_dataset_default):
    model = CountingModel(fake_ic_model_default, model_id="missing-id-model")
    metric = RecordingMetric("temporary", {"score": 1.0})
    metric.metadata = {}

    with pytest.raises(ValueError, match="must define metadata"):
        MaiteEvaluation().run(models=[model], datasets=[fake_ic_dataset_default], metrics=[metric], use_cache=False)

    assert model.calls == 0


def test_duplicate_metric_ids_fail_before_inference(fake_ic_model_default, fake_ic_dataset_default):
    model = CountingModel(fake_ic_model_default, model_id="duplicate-id-model")
    metrics = [
        RecordingMetric("duplicate", {"a": 0.1}),
        RecordingMetric("duplicate", {"b": 0.2}),
    ]

    with pytest.raises(ValueError, match="must be unique"):
        MaiteEvaluation().run(models=[model], datasets=[fake_ic_dataset_default], metrics=metrics, use_cache=False)

    assert model.calls == 0


@pytest.mark.parametrize("stage", ["reset", "update", "compute"])
def test_metric_failure_aborts_with_attributed_pickleable_error(
    fake_ic_model_default,
    fake_ic_dataset_default,
    stage,
):
    model = CountingModel(fake_ic_model_default, model_id=f"failure-{stage}-model")
    failing = RecordingMetric("a-failing-metric", {"score": 0.1}, fail_stage=stage)
    later = RecordingMetric("z-later-metric", {"score": 0.2})

    with pytest.raises(MaiteEvaluationMetricError) as error_info:
        MaiteEvaluation().run(
            use_cache=False,
            models=[model],
            datasets=[fake_ic_dataset_default],
            metrics=[later, failing],
            config=MaiteEvaluationConfig(batch_size=4),
        )

    error = error_info.value
    assert error.metric_id == "a-failing-metric"
    assert error.stage == stage
    assert error.error_type == "ValueError"
    assert error.message == f"intentional {stage} failure"
    assert isinstance(error.__cause__, ValueError)
    restored = pickle.loads(pickle.dumps(error))  # noqa: S301 - round-trip a locally constructed exception
    assert (restored.metric_id, restored.stage, restored.error_type, restored.message) == (
        error.metric_id,
        error.stage,
        error.error_type,
        error.message,
    )

    if stage == "reset":
        assert model.calls == 0
        assert later.reset_calls == 0
    elif stage == "update":
        assert model.calls == 1
        assert later.update_calls == 0
    else:
        assert model.calls == 5
        assert later.compute_calls == 0


def test_normalization_failure_is_attributed(fake_ic_model_default, fake_ic_dataset_default):
    metric = RecordingMetric("bad-return-key", {"different": 0.5}, return_key="missing")

    with pytest.raises(MaiteEvaluationMetricError) as error_info:
        MaiteEvaluation().run(
            use_cache=False,
            models=[fake_ic_model_default],
            datasets=[fake_ic_dataset_default],
            metrics=[metric],
        )

    assert error_info.value.metric_id == "bad-return-key"
    assert error_info.value.stage == "normalize"
    assert error_info.value.error_type == "ValueError"


def test_generic_structured_metric_without_return_key_is_supported(fake_ic_model_default, fake_ic_dataset_default):
    metric = RecordingMetric(
        "structured-metric",
        {"score": 0.5, "enabled": True, "label": "0.8", "details": {"count": 2}},
    )

    run = MaiteEvaluation().run(
        use_cache=False,
        models=[fake_ic_model_default],
        datasets=[fake_ic_dataset_default],
        metrics=[metric],
    )
    result = run.outputs.metrics["structured-metric"]

    assert result.overall_metric_name is None
    assert result.overall_metric_value is None
    assert result.scalar_values == {"score": 0.5}
    assert result.result["enabled"] is True
    assert result.result["label"] == "0.8"
    assert result.result["details"] == {"count": 2}


def test_report_and_analytics_include_every_metric(fake_ic_model_default, fake_ic_dataset_default):
    metrics = [
        RecordingMetric("b-report-metric", {"b": 0.2}, return_key="b"),
        RecordingMetric("a-report-metric", {"a": 0.1}, return_key="a"),
    ]
    run = MaiteEvaluation().run(
        use_cache=False,
        models=[fake_ic_model_default],
        datasets=[fake_ic_dataset_default],
        metrics=metrics,
    )

    report = run.collect_md_report(threshold=0.5)
    records = run.extract()

    assert report.content.index("a-report-metric") < report.content.index("b-report-metric")
    assert {record.metric_id for record in records} == {"a-report-metric", "b-report-metric"}
    assert {record.run_uid for record in records} == {run.run_uid}


def test_duplicate_display_labels_do_not_fail(fake_ic_model_default, fake_ic_dataset_default):
    class ModelWithDuplicateLabels(CountingModel):
        def __init__(self, model):
            super().__init__(model, model_id="duplicate-display-label-model")
            self.metadata = {**self.metadata, "index2label": {0: "N/A", 1: "N/A"}}

    metric = RecordingMetric(
        "per-class-duplicates",
        {"per_class_flag": 1, "score": 0.5, "0": 0.1, "1": 0.2},
        return_key="score",
    )

    run = MaiteEvaluation().run(
        use_cache=False,
        models=[ModelWithDuplicateLabels(fake_ic_model_default)],
        datasets=[fake_ic_dataset_default],
        metrics=[metric],
    )

    assert run.outputs.metrics["per-class-duplicates"].class_metrics == {"N/A": 0.2}
