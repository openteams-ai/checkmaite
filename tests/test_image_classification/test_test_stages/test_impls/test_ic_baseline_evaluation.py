"""Test baseline evalutation"""

from unittest.mock import MagicMock

from jatic_ri.image_classification.test_stages import BaselineEvaluation


def test_baseline_evaluation_dummy_ic(fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default) -> None:
    """Test BaselineEvaluation implementation using dummy setup"""

    test = BaselineEvaluation()
    test.load_model(model=fake_ic_model_default, model_id=fake_ic_model_default.metadata["id"])
    test.load_metric(metric=fake_ic_metric_default, metric_id=fake_ic_metric_default.metadata["id"])
    test.load_threshold(threshold=0.5)
    test.load_dataset(dataset=fake_ic_dataset_default, dataset_id=fake_ic_dataset_default.metadata["id"])
    test.run()
    test.collect_report_consumables()


def test_baseline_evaluation_dummy_ic_with_cache(
    fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default
) -> None:
    """Test BaselineEvaluation implementation using cache"""
    test1 = BaselineEvaluation()
    test1.load_model(model=fake_ic_model_default, model_id=fake_ic_model_default.metadata["id"])
    test1.load_metric(metric=fake_ic_metric_default, metric_id=fake_ic_metric_default.metadata["id"])
    test1.load_threshold(threshold=0.5)
    test1.load_dataset(dataset=fake_ic_dataset_default, dataset_id=fake_ic_dataset_default.metadata["id"])
    test1.run(use_stage_cache=True)
    output1 = test1.collect_report_consumables()

    test2 = BaselineEvaluation()
    test2.load_model(model=fake_ic_model_default, model_id=fake_ic_model_default.metadata["id"])
    test2.load_metric(metric=fake_ic_metric_default, metric_id=fake_ic_metric_default.metadata["id"])
    test2.load_threshold(threshold=0.5)
    test2.load_dataset(dataset=fake_ic_dataset_default, dataset_id=fake_ic_dataset_default.metadata["id"])
    test2._run = MagicMock()  # mock out _run to ensure cache hit
    test2.run(use_stage_cache=True)
    output2 = test2.collect_report_consumables()

    assert test2._run.call_count == 0
    assert len(output1) == len(output2)
    assert all(len(output1[i]) == len(output2[i]) for i in range(len(output1)))
    assert all(output1[i].keys() == output2[i].keys() for i in range(len(output1)))
