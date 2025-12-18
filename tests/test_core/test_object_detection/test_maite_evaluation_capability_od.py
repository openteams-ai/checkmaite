from jatic_ri.core.object_detection.maite_evaluation_capability import MaiteEvaluation
from jatic_ri.core.object_detection.metrics import multiclass_map50_torch_metric_factory


def test_run_and_collect(fake_od_model_default, fake_od_dataset_default, fake_od_metric_default):
    capability = MaiteEvaluation()

    output = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
    )

    assert output.model_dump()  # smoke test

    assert output.collect_report_consumables(threshold=0.5)  # smoke test

    assert output.collect_md_report(threshold=0.5)  # smoke test


def test_multiclass(fake_od_model_default, fake_od_dataset_default):
    capability = MaiteEvaluation()

    metric = multiclass_map50_torch_metric_factory()

    output = capability.run(
        use_cache=False, models=[fake_od_model_default], metrics=[metric], datasets=[fake_od_dataset_default]
    )

    assert output.outputs.class_metrics is not None
