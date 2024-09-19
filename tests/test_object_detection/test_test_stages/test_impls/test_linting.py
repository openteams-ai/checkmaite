"""Test baseline evalutation"""

from jatic_ri.object_detection.test_stages.impls.dataeval_linting_test_stage import (
    DatasetLintingTest,
)


def test_linting(dummy_dataset) -> None:
    """Test Linting implementation"""

    test = DatasetLintingTest()
    test.load_dataset(dataset=dummy_dataset, dataset_id="dataset_1")
    test.run()
    output = test.collect_report_consumables()

    assert output
    assert len(output) == 1
    assert "duplicates" in output[0]
    assert "outliers" in output[0]
