import pytest

from gradient.templates_and_layouts.create_deck import create_deck

import jatic_ri
from jatic_ri.object_detection.datasets import CocoDetectionDataset
from jatic_ri import PACKAGE_DIR
from pathlib import Path
from jatic_ri.object_detection.models import TorchvisionODModel
from jatic_ri.util.dashboard_utils import rehydrate_test_stage_od
from jatic_ri.object_detection.metrics import map50_torch_metric_factory
from jatic_ri._common.test_stages.interfaces.plugins import (
    MetricPlugin,
    MultiModelPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    ThresholdPlugin,
    TwoDatasetPlugin,
)


@pytest.fixture(scope='session')
def model_od():
    """
    generate real od model wrapper
    NOTE: this should be replaced by a faked od model when available
    """
    model_name = "ssdlite320_mobilenet_v3_large"
    model_wrapper = TorchvisionODModel(model_name=model_name)

    return model_wrapper


@pytest.fixture(scope='session')
def dataset_od():
    """
    generate real od dataset wrapper
    NOTE: this should be replaced by a faked od model when available
    """
    coco_dataset_dir = PACKAGE_DIR.parent.parent.joinpath(Path('tests/testing_utilities/example_data/coco_resized_val2017'))
    coco_dataset = CocoDetectionDataset(
        root=str(coco_dataset_dir),
        ann_file=str(coco_dataset_dir.joinpath('instances_val2017_resized_6.json')),
    )
    return coco_dataset



@pytest.mark.real_data
@pytest.mark.parametrize(
    "config_fixture_name",
    ['reallabel_config_od', 'nrtk_config_od', 'survivor_config_od', 'xaitk_config_od', 'feasibility_config_od', 'bias_config_od', 'linting_config_od', 'baseline_eval_config_od', 'shift_config_od'],
)
def test_rehydrate_and_run_od(config_fixture_name, request, model_od, dataset_od, artifact_dir):
    """Run end to end test from test stage config output to running the test. 
    This is only for local testing as it is very time consuming. 
    Once a faked model and dataset exist, it can be run in CI.
    """
    config = request.getfixturevalue(config_fixture_name)
    
    test_stage = rehydrate_test_stage_od(config=config)

    metric_od = map50_torch_metric_factory()

    if isinstance(test_stage, TwoDatasetPlugin):
        test_stage.load_datasets(
            dataset_od, "dataset1", dataset_od, "dataset2"
        )
    elif isinstance(test_stage, SingleDatasetPlugin):
        test_stage.load_dataset(dataset_od, "dataset1")

    if isinstance(test_stage, MetricPlugin):
        test_stage.load_metric(metric_od, metric_od.return_key)

    if isinstance(test_stage, ThresholdPlugin):
        test_stage.load_threshold(0.5)

    if isinstance(test_stage, SingleModelPlugin):
        test_stage.load_model(model_od, model_id="model_1")
    elif isinstance(test_stage, MultiModelPlugin):
        test_stage.load_models(models={'model_1': model_od})

    # run the stage, saving output to the class
    test_stage.run(use_cache=False)
    # collect the slides
    slides = test_stage.collect_report_consumables()

    # generate report
    report = create_deck(
        slides,
        artifact_dir, 
        deck_name=f"{test_stage.__class__.__name__}_report",
    )
