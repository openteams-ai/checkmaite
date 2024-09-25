"""Test baseline evalutation"""

import os
from jatic_ri.object_detection.test_stages.impls.dataeval_drift_test_stage import (
    DatasetDriftTestStage,
)


def test_drift(dummy_dataset) -> None:
    """Test DataEval implementation"""
    dev_dataset = dummy_dataset
    op_dataset = dummy_dataset
    op_dataset.images *= 0.5

    stage = DatasetDriftTestStage()
    stage.load_datasets(dataset_1=dev_dataset, dataset_2=op_dataset, dataset_1_id="dev", dataset_2_id="op")
    stage.run(use_cache=False)
    report = stage.collect_report_consumables()

    assert report
    assert len(report) == 1
    assert len(report[0]["Method"]) == 3
    assert len(report[0]["Has drifted?"]) == 3
    assert len(report[0]["Test statistic"]) == 3
    assert len(report[0]["P-value"]) == 3


def test_drift_cache(dummy_dataset, tmp_path) -> None:
    stage = DatasetDriftTestStage()
    stage.cache_base_path = tmp_path
    stage.load_datasets(dataset_1=dummy_dataset, dataset_2=dummy_dataset, dataset_1_id="dev", dataset_2_id="op")
    stage.run()

    assert os.path.exists(stage.cache_path)


def test_drift_no_outputs() -> None:
    assert DatasetDriftTestStage().collect_report_consumables() == []
