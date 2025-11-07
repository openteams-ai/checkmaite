from copy import deepcopy

import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri._common.test_stages.interfaces.plugins import (
    ThresholdPlugin,
)
from jatic_ri._common.test_stages.interfaces.test_stage import Number
from jatic_ri.image_classification.datasets import YoloClassificationDataset
from jatic_ri.image_classification.metrics import accuracy_multiclass_torch_metric_factory
from jatic_ri.image_classification.models import TorchvisionICModel
from jatic_ri.util.dashboard_utils import rehydrate_test_stage_ic


@pytest.fixture(scope="session")
def model_ic():
    """
    generate real ic model wrapper
    NOTE: this should be replaced by a faked ic model when available
    """
    model_name = "resnext50_32x4d"
    return TorchvisionICModel(model_name=model_name)


@pytest.fixture(scope="session")
def dataset_ic(fake_dataset):
    """
    generate real ic dataset wrapper
    NOTE: this should be replaced by a faked ic model when available
    """
    dataset_root, _, _, _ = fake_dataset
    return YoloClassificationDataset(dataset_id="test_dataset", root_dir=dataset_root, split="test")


@pytest.mark.real_data
@pytest.mark.parametrize(
    "config_fixture_name",
    [
        pytest.param(
            "nrtk_config_ic", marks=[pytest.mark.filterwarnings("ignore:No artists with labels found:UserWarning")]
        ),
        # "survivor_config_ic",
        "feasibility_config_ic",
        pytest.param(
            "bias_config_ic",
            marks=[pytest.mark.filterwarnings(r"ignore:.*?did not meet the recommended \d+ occurrences:UserWarning")],
        ),
        pytest.param(
            "cleaning_config_ic",
            marks=[
                pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning"),
                pytest.mark.filterwarnings(
                    "ignore:Precision loss occurred in moment calculation due to catastrophic cancellation:RuntimeWarning"
                ),
            ],
        ),
        "baseline_eval_config_ic",
        "shift_config_ic",
    ],
)
def test_rehydrate_and_run_ic(config_fixture_name, request, model_ic, dataset_ic, artifact_dir):
    """Run end to end test from test stage config output to running the test.
    This is only for local testing as it is very time consuming.
    Once a faked model and dataset exist, it can be run in CI.

    xaitk is purposefully not tested
    See https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/345
    """
    config = request.getfixturevalue(config_fixture_name)

    test_stage = rehydrate_test_stage_ic(config=config)

    metric_ic = accuracy_multiclass_torch_metric_factory(num_classes=12)

    # use deepcopy to enforce distinct datasets
    if test_stage.supports_datasets == Number.TWO:
        datasets = [deepcopy(dataset_ic), deepcopy(dataset_ic)]
    elif test_stage.supports_datasets == Number.ONE:
        datasets = [dataset_ic]
    elif test_stage.supports_datasets == Number.ZERO:
        datasets = []
    else:
        raise ValueError("Test should be rewritten if more than two datasets used.")

    if test_stage.supports_metrics == Number.ONE:
        metrics = [metric_ic]
    elif test_stage.supports_metrics == Number.ZERO:
        metrics = []
    else:
        raise ValueError("Test should be rewritten if multiple metrics used.")

    if isinstance(test_stage, ThresholdPlugin):
        test_stage.load_threshold(0.5)

    # use deepcopy to enforce distinct datasets
    if test_stage.supports_models == Number.MANY:
        models = [deepcopy(model_ic), deepcopy(model_ic), deepcopy(model_ic)]
    elif test_stage.supports_models == Number.TWO:
        models = [deepcopy(model_ic), deepcopy(model_ic)]
    elif test_stage.supports_models == Number.ONE:
        models = [model_ic]
    else:
        models = []

    # run the stage, saving output to the class
    test_stage.run(models=models, datasets=datasets, metrics=metrics, use_stage_cache=False)
    # collect the slides
    slides = test_stage.collect_report_consumables()
    # generate report
    create_deck(
        slides,
        artifact_dir,
        deck_name=f"{test_stage.__class__.__name__}_report",
    )
