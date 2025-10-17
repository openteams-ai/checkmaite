import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri._common.test_stages.interfaces.plugins import (
    MetricPlugin,
    MultiModelPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    ThresholdPlugin,
    TwoDatasetPlugin,
)
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
        "survivor_config_ic",
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

    if isinstance(test_stage, TwoDatasetPlugin):
        test_stage.load_datasets(dataset_ic, "dataset1", dataset_ic, "dataset2")
    elif isinstance(test_stage, SingleDatasetPlugin):
        test_stage.load_dataset(dataset_ic, "dataset1")

    if isinstance(test_stage, MetricPlugin):
        test_stage.load_metric(metric_ic, metric_ic.return_key)

    if isinstance(test_stage, ThresholdPlugin):
        test_stage.load_threshold(0.5)

    if isinstance(test_stage, SingleModelPlugin):
        test_stage.load_model(model_ic, model_id="model_1")
    elif isinstance(test_stage, MultiModelPlugin):
        test_stage.load_models(models={"model_1": model_ic})

    # run the stage, saving output to the class
    test_stage.run(use_stage_cache=False)
    # collect the slides
    slides = test_stage.collect_report_consumables()
    # generate report
    create_deck(
        slides,
        artifact_dir,
        deck_name=f"{test_stage.__class__.__name__}_report",
    )
