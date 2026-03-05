import os
from copy import deepcopy

import pytest
from PIL import Image

from checkmaite.core.capability_core import Number
from checkmaite.core.image_classification.dataset_loaders import YoloClassificationDataset
from checkmaite.core.image_classification.metrics import accuracy_multiclass_torch_metric_factory
from checkmaite.core.image_classification.models import TorchvisionICModel
from checkmaite.ui.dashboard_utils import get_capability_from_app_config_ic

CLASSES = ["cat", "dog"]
NUM_IMAGES_PER_CLASS = 4
IMG_SHAPE = (64, 128)


def create_fake_yolo_dataset(
    root_dir,
    split,
    classes,
    num_images_per_class,
    image_shape,
) -> None:
    """Create a fake YOLO dataset structure.

    Parameters
    ----------
    root_dir
        The root directory where the dataset will be created.
    split
        The dataset split (e.g., "train", "test").
    classes
        A list of class names.
    num_images_per_class
        The number of images to create for each class.
    image_shape
        The shape (width, height) of the images to create.
    """
    os.makedirs(root_dir / split, exist_ok=True)
    for class_name in classes:
        class_dir = root_dir / split / class_name
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_images_per_class):
            img = Image.new("RGB", image_shape, color=(i, i, i))
            img.save(class_dir / f"{i}_{class_name}.jpg")


@pytest.fixture(scope="session")
def fake_dataset(
    tmp_path_factory,
):
    """Create a fake YOLO dataset for testing.

    Parameters
    ----------
    tmp_path_factory
        Pytest fixture for creating temporary directories.

    Returns
    -------
        A tuple containing:
        - The root directory of the created dataset.
        - The list of class names.
        - The number of images per class.
        - The shape of the images.
    """
    dataset_root = tmp_path_factory.mktemp("yolo_dataset")

    for split in ["test", "train"]:
        create_fake_yolo_dataset(
            root_dir=dataset_root,
            split=split,
            classes=CLASSES,
            num_images_per_class=NUM_IMAGES_PER_CLASS,
            image_shape=IMG_SHAPE,
        )
    return str(dataset_root), CLASSES, NUM_IMAGES_PER_CLASS, IMG_SHAPE


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


@pytest.mark.unsupported
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
def test_get_capability_from_app_config_and_run_ic(config_fixture_name, request, model_ic, dataset_ic):
    """Run end to end test from test stage config output to running the test.
    This is only for local testing as it is very time consuming.
    Once a faked model and dataset exist, it can be run in CI.

    xaitk is purposefully not tested
    See https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/345
    """
    config = request.getfixturevalue(config_fixture_name)

    loaded_capability = get_capability_from_app_config_ic(config=config)
    capability = loaded_capability["stage"]
    config_obj = loaded_capability["config"]

    metric_ic = accuracy_multiclass_torch_metric_factory(num_classes=12)

    # use deepcopy to enforce distinct datasets
    if capability.supports_datasets == Number.TWO:
        datasets = [deepcopy(dataset_ic), deepcopy(dataset_ic)]
    elif capability.supports_datasets == Number.ONE:
        datasets = [dataset_ic]
    elif capability.supports_datasets == Number.ZERO:
        datasets = []
    else:
        raise ValueError("Test should be rewritten if more than two datasets used.")

    if capability.supports_metrics == Number.ONE:
        metrics = [metric_ic]
    elif capability.supports_metrics == Number.ZERO:
        metrics = []
    else:
        raise ValueError("Test should be rewritten if multiple metrics used.")

    if capability.supports_models == Number.MANY:
        models = [deepcopy(model_ic), deepcopy(model_ic), deepcopy(model_ic)]
    elif capability.supports_models == Number.TWO:
        models = [deepcopy(model_ic), deepcopy(model_ic)]
    elif capability.supports_models == Number.ONE:
        models = [model_ic]
    else:
        models = []

    run = capability.run(models=models, datasets=datasets, metrics=metrics, use_cache=False, config=config_obj)

    run.collect_report_consumables(threshold=0.5)
