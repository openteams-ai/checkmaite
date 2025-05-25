"""Test Object Detection Cleaning Test Stage"""

from unittest.mock import MagicMock

import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.object_detection.test_stages.impls.dataeval_cleaning_test_stage import (
    DatasetCleaningTestStage,
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
def test_od_cleaning(dummy_cleaning_dataset_od) -> None:
    """Test Cleaning implementation"""

    test = DatasetCleaningTestStage()
    test.load_dataset(dataset=dummy_cleaning_dataset_od(), dataset_id="dummy_cleaning")
    test.run(use_stage_cache=False)
    output = test.collect_report_consumables()

    assert output


@ignore_degenerate_data_warnings
def test_od_cleaning_with_images(dummy_cleaning_dataset_od) -> None:
    """Test Cleaning implementation with optional images"""

    test = DatasetCleaningTestStage()
    test.load_dataset(dataset=dummy_cleaning_dataset_od(), dataset_id="dummy_cleaning")
    test.run(use_stage_cache=False)
    output = test.collect_report_consumables()
    out_report = test._generate_image_outliers_report(True)
    tar_report = test._generate_target_outliers_report(True)

    assert out_report
    assert tar_report

    assert output
    assert len(output) == 14


@ignore_degenerate_data_warnings
def test_od_cleaning_with_cached_values(dummy_cleaning_dataset_od) -> None:
    """Verify cached"""
    test1 = DatasetCleaningTestStage()
    test1.load_dataset(dataset=dummy_cleaning_dataset_od(), dataset_id="dummy_cleaning")
    test1.run()
    output1 = test1.collect_report_consumables()

    test2 = DatasetCleaningTestStage()
    test2._run = MagicMock()  # mock out _run to ensure cache hit
    test2.load_dataset(dataset=dummy_cleaning_dataset_od(), dataset_id="dummy_cleaning")
    test2.run()
    output2 = test2.collect_report_consumables()

    assert test2._run.call_count == 0
    assert len(output1) == len(output2)
    assert all(len(output1[i]) == len(output2[i]) for i in range(len(output1)))
    assert all(output1[i].keys() == output2[i].keys() for i in range(len(output1)))


@ignore_degenerate_data_warnings
@pytest.mark.parametrize("offset_box", [True, False])
def test_od_cleaning_create_deck(offset_box, dummy_cleaning_dataset_od, artifact_dir) -> None:
    """This is used to test the output of the feasibility gradient slides"""
    test = DatasetCleaningTestStage()
    test.load_dataset(dataset=dummy_cleaning_dataset_od(offset_box), dataset_id="dummy_cleaning")
    test.run()

    slides = test.collect_report_consumables()
    filename = create_deck(slides, artifact_dir, "TestCleaningDeck")
    assert filename.exists()


@pytest.mark.filterwarnings(r"ignore:Image must be larger than \d+x\d+:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Bounding box .*? is invalid:UserWarning")
@pytest.mark.filterwarnings(r"ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.filterwarnings(r"ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings(r"ignore:Degrees of freedom <= 0 for slice:RuntimeWarning")
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

    stage = DatasetCleaningTestStage()

    stage.load_dataset(dataset=coco_dataset, dataset_id="asd")

    stage.run(use_stage_cache=False)
    pass  # no explosions
