from copy import deepcopy
from pathlib import Path

import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri import PACKAGE_DIR
from jatic_ri._common.test_stages.interfaces.plugins import (
    ThresholdPlugin,
)
from jatic_ri._common.test_stages.interfaces.test_stage import Number
from jatic_ri.object_detection.datasets import CocoDetectionDataset
from jatic_ri.object_detection.metrics import map50_torch_metric_factory
from jatic_ri.object_detection.models import TorchvisionODModel
from jatic_ri.util.dashboard_utils import rehydrate_test_stage_od


@pytest.fixture(scope="session")
def model_od():
    """
    generate real od model wrapper
    NOTE: this should be replaced by a faked od model when available
    """
    model_name = "ssdlite320_mobilenet_v3_large"
    return TorchvisionODModel(model_name=model_name)


@pytest.fixture(scope="session")
def dataset_od():
    """
    generate real od dataset wrapper
    NOTE: this should be replaced by a faked od model when available
    """
    coco_dataset_dir = PACKAGE_DIR.parent.parent.joinpath(
        Path("tests/testing_utilities/example_data/coco_resized_val2017")
    )
    return CocoDetectionDataset(
        root=str(coco_dataset_dir),
        ann_file=str(coco_dataset_dir.joinpath("instances_val2017_resized_6.json")),
    )


@pytest.fixture(scope="session")
def dataset_od_mini():
    """
    generate real od dataset wrapper
    NOTE: this should be replaced by a faked od model when available
    """
    coco_dataset_dir = PACKAGE_DIR.parent.parent.joinpath(
        Path("tests/testing_utilities/example_data/coco_resized_val2017")
    )
    return CocoDetectionDataset(
        root=str(coco_dataset_dir),
        ann_file=str(coco_dataset_dir.joinpath("three_image.json")),
    )


@pytest.mark.real_data
@pytest.mark.parametrize(
    "config_fixture_name",
    [
        # "reallabel_config_od",
        pytest.param(
            "nrtk_config_od",
            marks=[
                pytest.mark.filterwarnings(r"ignore:.*?more than \d+ detections in a single image:UserWarning"),
                pytest.mark.filterwarnings("ignore:No artists with labels found:UserWarning"),
            ],
        ),
        # pytest.param(
        #     "survivor_config_od",
        #     marks=[pytest.mark.filterwarnings(r"ignore:.*?more than \d+ detections in a single image:UserWarning")],
        # ),
        "xaitk_config_od",
        # pytest.param(
        #     "feasibility_config_od",
        #     marks=[
        #         pytest.mark.xfail(
        #             reason="Feasability computation is broken. See https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/181",
        #         )
        #     ],
        # ),
        pytest.param(
            "bias_config_od",
            marks=[pytest.mark.filterwarnings(r"ignore:.*?did not meet the recommended \d+ occurrences:UserWarning")],
        ),
        pytest.param(
            "cleaning_config_od",
            marks=[
                pytest.mark.filterwarnings(r"ignore:Image must be larger than \d+x\d+:UserWarning"),
                pytest.mark.filterwarnings(r"ignore:Bounding box .*? is out of bounds:UserWarning"),
                pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning"),
            ],
        ),
        pytest.param(
            "baseline_eval_config_od",
            marks=[pytest.mark.filterwarnings(r"ignore:.*?more than \d+ detections in a single image:UserWarning")],
        ),
        "shift_config_od",
    ],
)
def test_rehydrate_and_run_od(config_fixture_name, request, model_od, dataset_od, dataset_od_mini, artifact_dir):
    """Run end to end test from test stage config output to running the test.
    This is only for local testing as it is very time consuming.
    Once a faked model and dataset exist, it can be run in CI.
    """
    config = request.getfixturevalue(config_fixture_name)

    test_stage = rehydrate_test_stage_od(config=config)

    metric_od = map50_torch_metric_factory()

    if config["TYPE"] == "XAITKTestStage":
        dataset = dataset_od_mini
    else:
        dataset = dataset_od

    # use deepcopy to enforce distinct datasets
    if test_stage.supports_datasets == Number.TWO:
        datasets = [deepcopy(dataset), deepcopy(dataset)]
    elif test_stage.supports_datasets == Number.ONE:
        datasets = [dataset]
    elif test_stage.supports_datasets == Number.ZERO:
        datasets = []
    else:
        raise ValueError("Test should be rewritten if more than two datasets used.")

    if test_stage.supports_metrics == Number.ONE:
        metrics = [metric_od]
    elif test_stage.supports_metrics == Number.ZERO:
        metrics = []
    else:
        raise ValueError("Test should be rewritten if multiple metrics used.")

    if isinstance(test_stage, ThresholdPlugin):
        test_stage.load_threshold(0.5)

    # use deepcopy to enforce distinct models
    if test_stage.supports_models == Number.MANY:
        models = [deepcopy(model_od), deepcopy(model_od), deepcopy(model_od)]
    elif test_stage.supports_models == Number.TWO:
        models = [deepcopy(model_od), deepcopy(model_od)]
    elif test_stage.supports_models == Number.ONE:
        models = [model_od]
    else:
        models = []

    # run the stage, saving output to the class
    test_stage.run(datasets=datasets, metrics=metrics, models=models, use_stage_cache=False)
    # collect the slides
    slides = test_stage.collect_report_consumables()

    # generate report
    create_deck(
        slides,
        artifact_dir,
        deck_name=f"{test_stage.__class__.__name__}_report",
    )
