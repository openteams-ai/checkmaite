"""Test Dataset Bias Analysis"""

from jatic_ri.object_detection.test_stages.impls.dataeval_bias_test_stage import (
    DatasetBiasTest,
)


def test_bias(dummy_dataset, tmp_path) -> None:
    """Test Linting implementation"""

    test = DatasetBiasTest()
    test.cache_base_path = tmp_path
    test.load_dataset(dataset=dummy_dataset, dataset_id="dataset_1")
    test.run()
    output = test.collect_report_consumables()

    assert output
    assert len(output) == 2
    for slide in output:
        assert isinstance(slide, dict)

def test_empty_cache() -> None:
    """Tests return from cache and default self.outputs"""

    test = DatasetBiasTest()
    output = test.collect_report_consumables()
    assert output == []
