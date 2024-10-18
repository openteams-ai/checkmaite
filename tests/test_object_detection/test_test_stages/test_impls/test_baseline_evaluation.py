"""Test baseline evalutation"""

from jatic_ri.object_detection.test_stages.impls.baseline_evaluation import (
    BaselineEvaluation,
)


def test_baseline_evaluation(dummy_model_od, dummy_dataset_od, dummy_metric_od) -> None:
    """Test BaselineEvaluation implementation"""

    test = BaselineEvaluation()
    # load the maite compliant model
    test.load_model(model=dummy_model_od, model_id="model_1")
    test.load_metric(metric=dummy_metric_od, metric_id="metric_1")
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dummy_dataset_od, dataset_id="dataset_1")
    test.run()
    test.collect_report_consumables()
