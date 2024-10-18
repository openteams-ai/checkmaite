"""Test baseline evalutation"""

import copy
import os

from jatic_ri.object_detection.test_stages.impls.dataeval_ood_test_stage import (
    DatasetOODTestStage,
)


def test_ood(dummy_dataset_od, tmp_path) -> None:
    """Test OOD Detection implementation"""
    dev_dataset = dummy_dataset_od
    op_dataset = copy.deepcopy(dev_dataset)
    op_dataset.images *= 0.5

    test = DatasetOODTestStage()
    test.cache_base_path = tmp_path
    test.load_datasets(dataset_1=dev_dataset, dataset_2=op_dataset, dataset_1_id="dev", dataset_2_id="op")
    test.run()
    report = test.collect_report_consumables()

    assert report
    assert len(report) == 1
    assert "Method" in report[0]
    assert "Test statistic" in report[0]


def test_ood_with_cache(dummy_dataset_od, tmp_path) -> None:
    test = DatasetOODTestStage()
    test.cache_base_path = tmp_path

    dev_dataset = dummy_dataset_od
    op_dataset = copy.deepcopy(dev_dataset)
    op_dataset.images *= 0.5

    test.load_datasets(dataset_1=dev_dataset, dataset_2=op_dataset, dataset_1_id="dev", dataset_2_id="op")
    test.run()

    assert os.path.exists(test.cache_path)


def test_ood_with_cached_values(dummy_dataset_od, tmp_path) -> None:
    dev_dataset = dummy_dataset_od
    op_dataset = copy.deepcopy(dev_dataset)
    op_dataset.images *= 0.5

    test = DatasetOODTestStage()
    test.cache_base_path = tmp_path
    test.load_datasets(dataset_1=dev_dataset, dataset_2=op_dataset, dataset_1_id="dev", dataset_2_id="op")
    test.run()
    output1 = test.collect_report_consumables()

    test2 = DatasetOODTestStage()
    test2.cache_base_path = tmp_path
    test2.load_datasets(dataset_1=dev_dataset, dataset_2=op_dataset, dataset_1_id="dev", dataset_2_id="op")
    test2.run()
    output2 = test2.collect_report_consumables()

    assert os.path.exists(test.cache_path)
    assert test.cache_path == test2.cache_path
    assert output1 == output2
