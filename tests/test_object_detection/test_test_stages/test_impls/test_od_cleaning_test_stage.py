"""Test Object Detection Cleaning Test Stage"""

from unittest.mock import MagicMock

import pytest

from jatic_ri.object_detection.test_stages import DatasetCleaningTestStage


def ignore_degenerate_data_warnings(test_fn):
    # See https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/merge_requests/197#note_167139
    for filter in [
        "ignore:invalid value encountered in scalar divide:RuntimeWarning",
        "ignore:Precision loss occurred in moment calculation due to catastrophic cancellation:RuntimeWarning",
    ]:
        test_fn = pytest.mark.filterwarnings(filter)(test_fn)

    return test_fn


@ignore_degenerate_data_warnings
def test_od_cleaning(fake_od_dataset_default) -> None:
    """Test Cleaning implementation"""

    test = DatasetCleaningTestStage()
    run = test.run(use_stage_cache=False, datasets=[fake_od_dataset_default])
    output = run.collect_report_consumables(threshold=0.5)

    assert output


@ignore_degenerate_data_warnings
def test_od_cleaning_with_cached_values(fake_od_dataset_default) -> None:
    """Verify cached"""
    test1 = DatasetCleaningTestStage()
    run1 = test1.run(datasets=[fake_od_dataset_default])
    output1 = run1.collect_report_consumables(threshold=0.5)

    test2 = DatasetCleaningTestStage()
    test2._run = MagicMock()  # mock out _run to ensure cache hit
    run2 = test2.run(datasets=[fake_od_dataset_default])
    output2 = run2.collect_report_consumables(threshold=0.5)

    assert test2._run.call_count == 0
    assert len(output1) == len(output2)
    assert all(len(output1[i]) == len(output2[i]) for i in range(len(output1)))
    assert all(output1[i].keys() == output2[i].keys() for i in range(len(output1)))


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

    stage.run(use_stage_cache=False, datasets=[coco_dataset])
    pass  # no explosions
