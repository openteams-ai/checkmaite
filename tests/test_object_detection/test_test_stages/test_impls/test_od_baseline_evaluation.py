"""Test baseline evalutation"""

from unittest.mock import MagicMock

from jatic_ri.object_detection.metrics import multiclass_map50_torch_metric_factory
from jatic_ri.object_detection.test_stages import (
    BaselineEvaluation,
)


def test_baseline_evaluation_dummy_od(fake_od_model_default, fake_od_dataset_default, fake_od_metric_default) -> None:
    """Test BaselineEvaluation implementation using dummy setup"""

    test = BaselineEvaluation()
    test.load_threshold(threshold=0.5)

    run = test.run(
        use_stage_cache=False,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
    )

    assert run.outputs.class_metrics is None

    test.collect_report_consumables()


def test_baseline_evaluation_multiclass(fake_od_model_default, fake_od_dataset_default) -> None:
    """Test BaselineEvaluation with multiclass metrics that include per_class_flag"""

    test = BaselineEvaluation()
    metric = multiclass_map50_torch_metric_factory()
    test.load_threshold(threshold=0.5)

    run = test.run(
        use_stage_cache=False, models=[fake_od_model_default], metrics=[metric], datasets=[fake_od_dataset_default]
    )

    assert run.outputs.class_metrics is not None

    results = test.collect_report_consumables()

    # Assert classes not found in dummy dataset are added to output slide text
    for text in ("ignored regions", "apple", "eggplant"):
        assert text in results[0]["layout_arguments"]["text"]


def test_baseline_evaluation_dummy_od_with_cache(
    fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
) -> None:
    """Test BaselineEvaluation implementation using cache"""
    test1 = BaselineEvaluation()
    test1.load_threshold(threshold=0.5)
    test1.run(
        use_stage_cache=True,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
    )
    output1 = test1.collect_report_consumables()

    test2 = BaselineEvaluation()
    test2.load_threshold(threshold=0.5)

    test2._run = MagicMock()  # mock out _run to ensure cache hit
    test2.run(
        use_stage_cache=True,
        datasets=[fake_od_dataset_default],
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
    )
    output2 = test2.collect_report_consumables()

    assert test2._run.call_count == 0
    assert len(output1) == len(output2)
    assert all(len(output1[i]) == len(output2[i]) for i in range(len(output1)))
    assert all(output1[i].keys() == output2[i].keys() for i in range(len(output1)))
