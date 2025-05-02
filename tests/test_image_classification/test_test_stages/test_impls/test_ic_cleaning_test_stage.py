"""Test Image Classification Cleaning Test Stage"""

from pathlib import Path
from unittest.mock import MagicMock

from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.image_classification.test_stages.impls.dataeval_cleaning_test_stage import (
    DatasetCleaningTestStage,
)


def test_ic_cleaning(dummy_dataset_ic) -> None:
    """Test Cleaning implementation"""

    test = DatasetCleaningTestStage()
    test.load_dataset(dataset=dummy_dataset_ic, dataset_id="dummy_cleaning")
    test.run(use_stage_cache=False)
    output = test.collect_report_consumables()

    assert output
    assert len(output) == 7


def test_ic_cleaning_with_cached_values(dummy_dataset_ic, tmp_cache_path) -> None:
    """Verify cached"""
    test1 = DatasetCleaningTestStage()
    test1.load_dataset(dataset=dummy_dataset_ic, dataset_id="dummy_cleaning")
    test1.run()
    output1 = test1.collect_report_consumables()

    test2 = DatasetCleaningTestStage()
    test2._run = MagicMock()  # mock out _run to ensure cache hit
    test2.load_dataset(dataset=dummy_dataset_ic, dataset_id="dummy_cleaning")
    test2.run()
    output2 = test2.collect_report_consumables()

    assert test2._run.call_count == 0
    assert len(output1) == len(output2)
    assert all(len(output1[i]) == len(output2[i]) for i in range(len(output1)))
    assert all(output1[i].keys() == output2[i].keys() for i in range(len(output1)))


def test_ic_cleaning_create_deck(dummy_dataset_ic, tmp_path, artifact_dir) -> None:
    """This is used to test the output of the feasibility gradient slides"""
    test = DatasetCleaningTestStage()
    test.load_dataset(dataset=dummy_dataset_ic, dataset_id="dummy_cleaning")
    test.run()

    slides = test.collect_report_consumables()
    filename = create_deck(slides, path=Path(artifact_dir), deck_name="TestCleaningDeck")
    assert filename.exists()
