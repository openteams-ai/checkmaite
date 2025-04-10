"""Test Object Detection Linting Test Stage"""

from unittest.mock import MagicMock

import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.object_detection.test_stages.impls.dataeval_linting_test_stage import (
    DatasetLintingTestStage,
)


def ignore_degenerate_data_warnings(test_fn):
    # See https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/merge_requests/197#note_167139
    for filter in [
        "ignore:invalid value encountered in scalar divide:RuntimeWarning",
        "ignore:Precision loss occurred in moment calculation due to catastrophic cancellation:RuntimeWarning",
    ]:
        test_fn = pytest.mark.filterwarnings(filter)(test_fn)

    return test_fn


@ignore_degenerate_data_warnings
def test_od_linting(dummy_linting_dataset_od) -> None:
    """Test Linting implementation"""

    test = DatasetLintingTestStage()
    test.load_dataset(dataset=dummy_linting_dataset_od(), dataset_id="dummy_linting")
    test.run(use_stage_cache=False)
    output = test.collect_report_consumables()

    assert output
    assert len(output) == 4


@ignore_degenerate_data_warnings
def test_od_linting_with_cached_values(dummy_linting_dataset_od, tmp_path) -> None:
    """Verify cached"""
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


@ignore_degenerate_data_warnings
@pytest.mark.parametrize("offset_box", [True, False])
def test_od_linting_create_deck(offset_box, dummy_linting_dataset_od, tmp_path, artifact_dir) -> None:
    """This is used to test the output of the feasibility gradient slides"""
    test = DatasetLintingTestStage()
    test.cache_base_path = tmp_path
    test.load_dataset(dataset=dummy_linting_dataset_od(offset_box), dataset_id="dummy_linting")
    test.run()

    slides = test.collect_report_consumables()
    filename = create_deck(slides, artifact_dir, "TestLintingDeck")
    assert filename.exists()


@pytest.mark.filterwarnings(r"ignore:Image must be larger than \d+x\d+:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Bounding box .*? is out of bounds:UserWarning")
@ignore_degenerate_data_warnings
def test_coco():
    from os import path

    import tests
    from jatic_ri import PACKAGE_DIR
    from jatic_ri.object_detection.datasets import CocoDetectionDataset

    coco_dataset_dir = PACKAGE_DIR.parent.parent.joinpath(
        path.dirname(tests.__file__),
        ("testing_utilities/example_data/coco_resized_val2017"),
    )
    coco_dataset = CocoDetectionDataset(
        root=str(coco_dataset_dir),
        ann_file=str(coco_dataset_dir.joinpath("instances_val2017_resized_6.json")),
    )

    stage = DatasetLintingTestStage()

    stage.load_dataset(dataset=coco_dataset, dataset_id="asd")

    stage.run(use_stage_cache=False)
    pass  # no explosions
