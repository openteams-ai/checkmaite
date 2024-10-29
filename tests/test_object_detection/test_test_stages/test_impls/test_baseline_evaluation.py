"""Test baseline evalutation"""

import os
from pathlib import Path

import maite.protocols.object_detection as od

from jatic_ri.object_detection.test_stages.impls.baseline_evaluation import (
    BaselineEvaluation,
)


def test_baseline_evaluation_dummy(dummy_model_od, dummy_dataset_od, dummy_metric_od) -> None:
    """Test BaselineEvaluation implementation using dummy setup"""

    test = BaselineEvaluation()
    test.load_model(model=dummy_model_od, model_id="model_1")
    test.load_metric(metric=dummy_metric_od, metric_id="metric_1")
    test.load_threshold(threshold=0.5)
    test.load_dataset(dataset=dummy_dataset_od, dataset_id="dataset_1")
    test.run()
    test.collect_report_consumables()


def test_baseline_evaluation_real(model_od_yolov5, dataset_od_fwow, metric_od_map) -> None:
    """Test BaselineEvaluation implementation using real data"""

    test = BaselineEvaluation()
    test.load_model(model=model_od_yolov5, model_id="model_1")
    test.load_metric(metric=metric_od_map, metric_id="metric_1")
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dataset_od_fwow, dataset_id="dataset_1")
    test.run()
    test.collect_report_consumables()
