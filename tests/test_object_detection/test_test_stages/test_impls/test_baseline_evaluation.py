"""Test baseline evalutation"""

from jatic_ri.object_detection.test_stages.impls.baseline_evaluation import (
    BaselineEvaluation,
)


def test_baseline_evaluation(dummy_model, dummy_dataset, dummy_metric) -> None:
    """Test BaselineEvaluation implementation"""

    test = BaselineEvaluation()
    # load the maite compliant model
    test.load_model(model=dummy_model, model_id="model_1")
    test.load_metric(metric=dummy_metric, metric_id="metric_1")
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dummy_dataset, dataset_id="dataset_1")
    test.run()
    test.collect_report_consumables()
