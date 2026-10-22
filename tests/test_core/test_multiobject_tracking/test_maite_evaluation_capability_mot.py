import pytest
from maite.protocols import DatasetMetadata

from checkmaite.core.multiobject_tracking import MaiteEvaluationConfig, MaiteEvaluationMetricError
from checkmaite.core.multiobject_tracking.maite_evaluation_capability import MaiteEvaluation
from checkmaite.core.multiobject_tracking.metrics import DEFAULT_MOT_METRIC_NAMES, clear_mot_metrics_factory
from tests.report_assertions import assert_inline_markdown_report
from tests.test_core.test_multiobject_tracking.conftest import (
    EXPECTED_MOT_MOTA,
    EXPECTED_MOT_NUM_SWITCHES,
    FakeMOTDataset,
    FakeMOTModel,
    FakeMOTTarget,
    frame_tracks,
)


def _only_metric(run):
    (metric_result,) = run.outputs.metrics.values()
    return metric_result


@pytest.fixture
def test_run_mot(fake_mot_model_default, fake_mot_dataset_default):
    capability = MaiteEvaluation()
    output = capability.run(
        datasets=[fake_mot_dataset_default],
        models=[fake_mot_model_default],
        metrics=[clear_mot_metrics_factory()],
    )
    assert output.model_dump()  # smoke test

    return output


def test_maite_evaluation_mot_overall_metric(test_run_mot):
    metric_result = _only_metric(test_run_mot)
    assert metric_result.overall_metric_name == "mota"
    assert metric_result.overall_metric_value == pytest.approx(EXPECTED_MOT_MOTA)
    assert metric_result.scalar_values["num_switches"] == EXPECTED_MOT_NUM_SWITCHES
    assert metric_result.class_metrics is None


def test_maite_evaluation_mot_run_cache_round_trip(fake_mot_model_default, fake_mot_dataset_default):
    """The second run is served from cache, exercising the MOT target serialization round trip."""
    metric = clear_mot_metrics_factory()

    first = MaiteEvaluation().run(
        datasets=[fake_mot_dataset_default], models=[fake_mot_model_default], metrics=[metric]
    )
    second = MaiteEvaluation().run(
        datasets=[fake_mot_dataset_default], models=[fake_mot_model_default], metrics=[metric]
    )

    assert second.run_uid == first.run_uid
    assert second is not first
    assert _only_metric(second).overall_metric_value == pytest.approx(EXPECTED_MOT_MOTA)


class CountingMOTModel:
    """Protocol-compatible MOT model wrapper that records inference batch sizes."""

    def __init__(self, model, *, model_id: str):
        self._model = model
        self.metadata = {**model.metadata, "id": model_id}
        self.batch_sizes = []

    def __call__(self, inputs):
        self.batch_sizes.append(len(inputs))
        return self._model(inputs)


def test_batch_size_controls_inference_batches_and_preserves_metrics(fake_mot_model_default, fake_mot_dataset_default):
    model = CountingMOTModel(fake_mot_model_default, model_id="mot-batch-size-model")
    output = MaiteEvaluation().run(
        use_cache=False,
        datasets=[fake_mot_dataset_default],
        models=[model],
        metrics=[clear_mot_metrics_factory()],
        config=MaiteEvaluationConfig(batch_size=2),
    )

    assert model.batch_sizes == [2]
    assert _only_metric(output).overall_metric_value == pytest.approx(EXPECTED_MOT_MOTA)


def test_collect_md_report_mot(test_run_mot):
    report = test_run_mot.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=test_run_mot.capability_id)


def test_maite_evaluation_mot_extract_records(test_run_mot):
    records = test_run_mot.extract()

    # One flat record per compute() output key.
    assert len(records) == len(DEFAULT_MOT_METRIC_NAMES)
    by_key = {record.output_key: record for record in records}
    assert by_key["mota"].output_value == pytest.approx(EXPECTED_MOT_MOTA)
    assert all(record.scope == "overall" for record in records)


def test_class_agnostic_tracking_evaluates_despite_a_differing_taxonomy(fake_mot_model_default):
    """A tracker carries its detector's taxonomy (commonly COCO); MOT datasets rarely share it.

    CLEAR-MOT/identity matching here is class-agnostic, so a differing taxonomy must still evaluate.
    Nothing on main compares taxonomies today, so this guards against a model/dataset index2label
    check (checkmaite#743) rejecting tracking evaluations once it is enforced.
    """
    coco_flavoured = FakeMOTDataset(
        metadata=DatasetMetadata(id="mot_dataset_other_taxonomy", index2label={0: "person", 2: "car"})
    )

    output = MaiteEvaluation().run(
        datasets=[coco_flavoured],
        models=[fake_mot_model_default],
        metrics=[clear_mot_metrics_factory()],
    )

    assert _only_metric(output).overall_metric_value == pytest.approx(EXPECTED_MOT_MOTA)


def _degenerate_run_inputs(case):
    """One-video inputs whose tracking scores are undefined (division by zero)."""
    box = [0.0, 0.0, 10.0, 10.0]
    if case == "no_ground_truth":  # MOTA is -inf
        targets = [FakeMOTTarget([frame_tracks([], [])] * 2)]
        predictions = [FakeMOTTarget([frame_tracks([box], [5])] * 2)]
    else:  # all ground truth missed: MOTP is NaN
        targets = [FakeMOTTarget([frame_tracks([box], [1])] * 2)]
        predictions = [FakeMOTTarget([frame_tracks([], [])] * 2)]
    model = FakeMOTModel(predictions=predictions, metadata={"id": f"mot-model-{case}"})
    dataset = FakeMOTDataset(targets, DatasetMetadata(id=f"mot-dataset-{case}"))
    return model, dataset


def test_undefined_headline_metric_fails_the_same_way_on_a_cached_rerun():
    model, dataset = _degenerate_run_inputs("no_ground_truth")

    for _ in range(2):  # the second attempt reads the cached evaluation
        with pytest.raises(MaiteEvaluationMetricError, match="'mota' is undefined for this evaluation"):
            MaiteEvaluation().run(models=[model], datasets=[dataset], metrics=[clear_mot_metrics_factory()])


def test_undefined_secondary_metric_is_omitted_and_survives_the_run_cache(recwarn):
    model, dataset = _degenerate_run_inputs("all_missed")

    first = MaiteEvaluation().run(models=[model], datasets=[dataset], metrics=[clear_mot_metrics_factory()])
    calls_after_first = model.n_calls
    second = MaiteEvaluation().run(models=[model], datasets=[dataset], metrics=[clear_mot_metrics_factory()])

    metric_result = _only_metric(first)
    assert "motp" not in metric_result.scalar_values
    assert metric_result.result["motp"] is None
    assert metric_result.overall_metric_value == pytest.approx(0.0)
    assert model.n_calls == calls_after_first  # served from the run cache
    assert second.outputs.model_dump() == first.outputs.model_dump()
    assert not [w for w in recwarn if "invalid cache entry" in str(w.message)]
