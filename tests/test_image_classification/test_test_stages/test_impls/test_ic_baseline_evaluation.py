"""Test baseline evalutation"""

import os
from pathlib import Path

import maite.protocols.image_classification as ic

from jatic_ri.image_classification.test_stages.impls.baseline_evaluation import (
    BaselineEvaluation,
)


def test_baseline_evaluation_dummy_ic(dummy_model_ic, dummy_dataset_ic, dummy_metric_ic) -> None:
    """Test BaselineEvaluation implementation using dummy setup"""

    test = BaselineEvaluation()
    test.load_model(model=dummy_model_ic, model_id="model_1")
    test.load_metric(metric=dummy_metric_ic, metric_id="metric_1")
    test.load_threshold(threshold=0.5)
    test.load_dataset(dataset=dummy_dataset_ic, dataset_id="dataset_1")
    test.run()
    test.collect_report_consumables()
