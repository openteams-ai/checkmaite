"""Test baseline evalutation"""

from unittest.mock import MagicMock

from jatic_ri.image_classification.test_stages import BaselineEvaluation


def test_baseline_evaluation_dummy_ic(fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default) -> None:
    """Test BaselineEvaluation implementation using dummy setup"""

    test = BaselineEvaluation()
    run = test.run(datasets=[fake_ic_dataset_default], metrics=[fake_ic_metric_default], models=[fake_ic_model_default])
    run.collect_report_consumables(threshold=0.5)


def test_baseline_evaluation_dummy_ic_with_cache(
    fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default
) -> None:
    """Test BaselineEvaluation implementation using cache"""
    test1 = BaselineEvaluation()
    run1 = test1.run(
        use_stage_cache=True,
        models=[fake_ic_model_default],
        metrics=[fake_ic_metric_default],
        datasets=[fake_ic_dataset_default],
    )
    output1 = run1.collect_report_consumables(threshold=0.5)

    test2 = BaselineEvaluation()
    test2._run = MagicMock()  # mock out _run to ensure cache hit
    run2 = test2.run(
        use_stage_cache=True,
        models=[fake_ic_model_default],
        metrics=[fake_ic_metric_default],
        datasets=[fake_ic_dataset_default],
    )
    output2 = run2.collect_report_consumables(threshold=0.5)

    assert test2._run.call_count == 0
    assert len(output1) == len(output2)
    assert all(len(output1[i]) == len(output2[i]) for i in range(len(output1)))
    assert all(output1[i].keys() == output2[i].keys() for i in range(len(output1)))
