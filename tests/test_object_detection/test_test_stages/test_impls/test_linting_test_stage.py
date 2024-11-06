"""Test Object Detection Linting Test Stage"""

from unittest.mock import MagicMock
import pytest

from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.object_detection.test_stages.impls.dataeval_linting_test_stage import (
    DatasetLintingTestStage,
)


def test_od_linting(dummy_linting_dataset_od) -> None:
    """Test Linting implementation"""

    test = DatasetLintingTestStage()
    test.load_dataset(dataset=dummy_linting_dataset_od(), dataset_id="dummy_linting")
    test.run(use_cache=False)
    output = test.collect_report_consumables()

    assert output
    assert len(output) == 4


def test_od_linting_with_cached_values(dummy_linting_dataset_od, tmp_path) -> None:
    """Verify cached """
    test1 = DatasetLintingTestStage()
    test1.cache_base_path = tmp_path
    test1.load_dataset(dataset=dummy_linting_dataset_od(), dataset_id="dummy_linting")
    test1.run()
    output1 = test1.collect_report_consumables()

    test2 = DatasetLintingTestStage()
    test2.cache_base_path = tmp_path
    test2._run = MagicMock()  # mock out _run to ensure cache hit
    test2.load_dataset(dataset=dummy_linting_dataset_od(), dataset_id="dummy_linting")
    test2.run()
    output2 = test2.collect_report_consumables()

    assert test2._run.call_count == 0
    assert len(output1) == len(output2)
    assert all(len(output1[i]) == len(output2[i]) for i in range(len(output1)))
    assert all(output1[i].keys() == output2[i].keys() for i in range(len(output1)))


@pytest.mark.parametrize("offset_box", [True, False])
def test_od_linting_create_deck(offset_box, dummy_linting_dataset_od, tmp_path) -> None:
    """This is used to test the output of the feasibility gradient slides"""
    test = DatasetLintingTestStage()
    test.cache_base_path = tmp_path
    test.load_dataset(dataset=dummy_linting_dataset_od(offset_box), dataset_id="dummy_linting")
    test.run()

    slides = test.collect_report_consumables()
    create_deck(slides, path=tmp_path, deck_name="TestLintingDeck")
    assert (tmp_path / "TestLintingDeck.pptx").exists()
