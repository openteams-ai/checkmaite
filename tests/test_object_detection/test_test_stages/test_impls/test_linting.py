"""Test baseline evalutation"""

import os
from jatic_ri.object_detection.test_stages.impls.dataeval_linting_test_stage import (
    DatasetLintingTest,
)


def test_linting(dummy_dataset) -> None:
    """Test Linting implementation"""

    test = DatasetLintingTest()
    test.load_dataset(dataset=dummy_dataset, dataset_id="dataset_1")
    test.run(use_cache=False)
    output = test.collect_report_consumables()

    assert output
    assert len(output) == 1
    assert "duplicates" in output[0]
    assert "outliers" in output[0]


def test_linting_with_cache(dummy_dataset, tmp_path) -> None:
    test = DatasetLintingTest()
    test.cache_base_path = tmp_path
    test.load_dataset(dataset=dummy_dataset, dataset_id="dataset_1")
    test.run()

    assert os.path.exists(test.cache_path)


def test_linting_with_cached_values(dummy_dataset, tmp_path) -> None:
    test = DatasetLintingTest()
    test.cache_base_path = tmp_path
    test.load_dataset(dataset=dummy_dataset, dataset_id="dataset_1")
    test.run()

    test2 = DatasetLintingTest()
    test2.cache_base_path = tmp_path
    test2.load_dataset(dataset=dummy_dataset, dataset_id="dataset_1")
    test2.run()

    assert os.path.exists(test.cache_path)
    assert test.cache_path == test2.cache_path
    assert test.outputs == test2.outputs


def test_linting_report_consumables_with_no_outputs() -> None:
    assert DatasetLintingTest().collect_report_consumables() == []
