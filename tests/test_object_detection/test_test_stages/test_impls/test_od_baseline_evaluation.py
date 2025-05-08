"""Test baseline evalutation"""

from unittest.mock import MagicMock

from jatic_ri.object_detection.metrics import multiclass_map50_torch_metric_factory
from jatic_ri.object_detection.test_stages.impls.baseline_evaluation import (
    BaselineEvaluation,
)


def test_baseline_evaluation_dummy_od(
    fake_od_model_default, fake_od_dataset_default, fake_od_metric_default, default_eval_tool_no_cache
) -> None:
    """Test BaselineEvaluation implementation using dummy setup"""

    test = BaselineEvaluation()
    test.load_model(model=fake_od_model_default, model_id=fake_od_model_default.metadata["id"])
    test.load_metric(metric=fake_od_metric_default, metric_id=fake_od_metric_default.metadata["id"])
    test.load_threshold(threshold=0.5)
    test.load_dataset(dataset=fake_od_dataset_default, dataset_id=fake_od_dataset_default.metadata["id"])
    test.load_eval_tool(eval_tool=default_eval_tool_no_cache)

    run = test.run(use_stage_cache=False)

    assert run.outputs.class_metrics is None

    test.collect_report_consumables()


def test_baseline_evaluation_multiclass(
    fake_od_model_default, fake_od_dataset_default, default_eval_tool_no_cache
) -> None:
    """Test BaselineEvaluation with multiclass metrics that include per_class_flag"""

    test = BaselineEvaluation()
    test.load_model(model=fake_od_model_default, model_id=fake_od_model_default.metadata["id"])
    metric = multiclass_map50_torch_metric_factory()
    test.load_metric(metric=metric, metric_id=metric.metadata["id"])
    test.load_threshold(threshold=0.5)
    test.load_dataset(dataset=fake_od_dataset_default, dataset_id=fake_od_dataset_default.metadata["id"])
    test.load_eval_tool(eval_tool=default_eval_tool_no_cache)

    run = test.run(use_stage_cache=False)

    assert run.outputs.class_metrics is not None

    results = test.collect_report_consumables()

    # Assert classes not found in dummy dataset are added to output slide text
    for text in ("ignored regions", "apple", "eggplant"):
        assert text in results[0]["layout_arguments"]["text"]


def test_baseline_evaluation_dummy_od_with_cache(
    fake_od_model_default, fake_od_dataset_default, fake_od_metric_default, default_eval_tool_no_cache
) -> None:
    """Test BaselineEvaluation implementation using cache"""
    test1 = BaselineEvaluation()
    test1.load_model(model=fake_od_model_default, model_id=fake_od_model_default.metadata["id"])
    test1.load_metric(metric=fake_od_metric_default, metric_id=fake_od_metric_default.metadata["id"])
    test1.load_threshold(threshold=0.5)
    test1.load_dataset(dataset=fake_od_dataset_default, dataset_id=fake_od_dataset_default.metadata["id"])
    test1.load_eval_tool(eval_tool=default_eval_tool_no_cache)
    test1.run(use_stage_cache=True)
    output1 = test1.collect_report_consumables()

    test2 = BaselineEvaluation()
    test2.load_model(model=fake_od_model_default, model_id=fake_od_model_default.metadata["id"])
    test2.load_metric(metric=fake_od_metric_default, metric_id=fake_od_metric_default.metadata["id"])
    test2.load_threshold(threshold=0.5)
    test2.load_dataset(dataset=fake_od_dataset_default, dataset_id=fake_od_dataset_default.metadata["id"])
    test2.load_eval_tool(eval_tool=default_eval_tool_no_cache)
    test2._run = MagicMock()  # mock out _run to ensure cache hit
    test2.run(use_stage_cache=True)
    output2 = test2.collect_report_consumables()

    assert test2._run.call_count == 0
    assert len(output1) == len(output2)
    assert all(len(output1[i]) == len(output2[i]) for i in range(len(output1)))
    assert all(output1[i].keys() == output2[i].keys() for i in range(len(output1)))
