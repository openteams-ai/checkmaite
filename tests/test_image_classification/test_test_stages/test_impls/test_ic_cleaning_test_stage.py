"""Test Image Classification Cleaning Test Stage"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.image_classification.test_stages import DatasetCleaningTestStage


def ignore_degenerate_data_warnings(test_fn):
    # See https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/merge_requests/197#note_167139
    for filter in [
        "ignore:invalid value encountered in scalar divide:RuntimeWarning",
        "ignore:Precision loss occurred in moment calculation due to catastrophic cancellation:RuntimeWarning",
    ]:
        test_fn = pytest.mark.filterwarnings(filter)(test_fn)

    return test_fn


@ignore_degenerate_data_warnings
def test_ic_cleaning(fake_ic_dataset_default) -> None:
    """Test Cleaning implementation"""

    test = DatasetCleaningTestStage()
    test.run(use_stage_cache=False, datasets=[fake_ic_dataset_default])
    output = test.collect_report_consumables()

    assert output
    assert len(output) == 7


@ignore_degenerate_data_warnings
def test_ic_cleaning_with_cached_values(fake_ic_dataset_default) -> None:
    """Verify cached"""
    test1 = DatasetCleaningTestStage()
    test1.run(datasets=[fake_ic_dataset_default])
    output1 = test1.collect_report_consumables()

    test2 = DatasetCleaningTestStage()
    test2._run = MagicMock()  # mock out _run to ensure cache hit
    test2.run(datasets=[fake_ic_dataset_default])
    output2 = test2.collect_report_consumables()

    assert test2._run.call_count == 0
    assert len(output1) == len(output2)
    assert all(len(output1[i]) == len(output2[i]) for i in range(len(output1)))
    assert all(output1[i].keys() == output2[i].keys() for i in range(len(output1)))


@ignore_degenerate_data_warnings
def test_ic_cleaning_create_deck(fake_ic_dataset_default, artifact_dir) -> None:
    """This is used to test the output of the feasibility gradient slides"""
    test = DatasetCleaningTestStage()
    test.run(datasets=[fake_ic_dataset_default])

    slides = test.collect_report_consumables()
    filename = create_deck(slides, path=Path(artifact_dir), deck_name="TestCleaningDeck")
    assert filename.exists()
