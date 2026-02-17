from pathlib import Path

import pytest

from jatic_ri.core.object_detection.dataeval_cleaning_capability import DataevalCleaning
from jatic_ri.core.object_detection.dataset_loaders import CocoDetectionDataset
from jatic_ri.core.report._gradient import HAS_GRADIENT


def ignore_degenerate_data_warnings(test_fn):
    for filter in [
        "ignore:invalid value encountered in scalar divide:RuntimeWarning",
        "ignore:Precision loss occurred in moment calculation due to catastrophic cancellation:RuntimeWarning",
    ]:
        test_fn = pytest.mark.filterwarnings(filter)(test_fn)

    return test_fn


# Pytest does not support applying decorators to fixtures directly
# so we can't transform fake_od_dataset_default fixture here.
@ignore_degenerate_data_warnings
def test_run_and_collect(fake_od_dataset_default):
    capability = DataevalCleaning()

    output = capability.run(use_cache=False, datasets=[fake_od_dataset_default])

    assert output.model_dump()  # smoke test


@ignore_degenerate_data_warnings
@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_collect_reports(fake_od_dataset_default):
    capability = DataevalCleaning()
    output = capability.run(use_cache=False, datasets=[fake_od_dataset_default])
    with pytest.warns(DeprecationWarning):
        assert output.collect_report_consumables(threshold=0.5)  # smoke test


@ignore_degenerate_data_warnings
def test_collect_md_report(fake_od_dataset_default):
    capability = DataevalCleaning()
    output = capability.run(use_cache=False, datasets=[fake_od_dataset_default])
    assert output.collect_md_report(threshold=0.5)  # smoke test


@pytest.mark.filterwarnings(r"ignore:Image must be larger than \d+x\d+:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Bounding box .*? is invalid:UserWarning")
@pytest.mark.filterwarnings(r"ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.filterwarnings(r"ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings(r"ignore:Degrees of freedom <= 0 for slice:RuntimeWarning")
@ignore_degenerate_data_warnings
def test_coco_run():
    root = Path(__file__).parents[2] / "data_for_tests"
    coco_dataset_dir = root / "coco_resized_val2017"
    coco_dataset = CocoDetectionDataset(
        root=str(coco_dataset_dir),
        ann_file=str(coco_dataset_dir.joinpath("instances_val2017_resized_6.json")),
    )

    stage = DataevalCleaning()

    stage.run(use_cache=False, datasets=[coco_dataset])

    pass  # no explosions
