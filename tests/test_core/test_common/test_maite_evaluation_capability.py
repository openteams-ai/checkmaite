from jatic_ri.core.image_classification.maite_evaluation_capability import MaiteEvaluation


def test_run_and_collect(fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default):
    capability = MaiteEvaluation()
    output = capability.run(
        datasets=[fake_ic_dataset_default], metrics=[fake_ic_metric_default], models=[fake_ic_model_default]
    )

    assert output.model_dump()  # smoke test

    assert output.collect_report_consumables(threshold=0.5)  # smoke test

    assert output.collect_md_report(threshold=0.5)  # smoke test
