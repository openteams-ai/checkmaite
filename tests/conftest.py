import contextlib
import os
import platform
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

# https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/326
# must be set before torch is imported!
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import maite.protocols.image_classification as ic
import maite.protocols.object_detection as od
import numpy as np
import numpy.typing
import pytest
import torch
from maite.protocols import ArrayLike
from nrtk.impls.perturb_image.generic.PIL.enhance import BrightnessPerturber
from nrtk.impls.perturb_image_factory.generic.one_step import OneStepPerturbImageFactory
from PIL import Image

import jatic_ri
from tests.fake_ic_classes import FakeICDataset, FakeICMetric, FakeICModel
from tests.fake_od_classes import FakeODDataset, FakeODMetric, FakeODModel

if tuple(int(v) for v in np.__version__.split(".")[:2]) >= (2, 1):
    np_unstack = np.unstack
else:

    def np_unstack(x: np.typing.NDArray, /, *, axis: int = 0) -> list[np.typing.NDArray]:
        if platform.system() == "Darwin":
            x = x.astype(np.float32)
        return [y.squeeze(axis) for y in np.split(x, x.shape[axis], axis=axis)]


SEED = 42
torch.manual_seed(SEED)
RNG = np.random.default_rng(seed=SEED)


@pytest.fixture(autouse=True)
def _reset_random_state():
    """
    Reset random state before each test to ensure reproducibility.
    """
    global RNG
    torch.manual_seed(SEED)
    RNG = np.random.default_rng(seed=SEED)


@contextlib.contextmanager
def _tmp_cache_path(new: Path):
    """
    Temporarily sets the cache path to a new directory
    and reverts this upon exiting the context manager.
    """
    old = jatic_ri.cache_path()
    try:
        yield jatic_ri.cache_path(new)
    finally:
        jatic_ri.cache_path(old)


@pytest.fixture(autouse=True)
def tmp_cache_path(tmp_path):
    """
    Automatically sets the cache path to a temporary directory.
    """
    with _tmp_cache_path(tmp_path) as p:
        yield p


@pytest.fixture
def artifact_dir(tmp_path):
    """
    Provides a directory for storing test artifacts.

    By default, artifacts go to a temporary directory that is removed
    after the tests. Set SAVE_POETRY_ARTIFACTS=true to persist
    artifacts to ARTIFACT_DIR (or fallback to tmp if unset).
    """
    _save = os.environ.get("SAVE_POETRY_ARTIFACTS", "false").strip().lower() == "true"
    _env_dir = os.environ.get("ARTIFACT_DIR")

    if _save and _env_dir:
        path = Path(_env_dir).expanduser().resolve()
    else:
        path = tmp_path

    path.mkdir(parents=True, exist_ok=True)
    return path


class DummyObjectDetectionTarget(od.ObjectDetectionTarget):
    """5x Target data entries per image with labels [0-4]"""

    @property
    def boxes(self) -> ArrayLike:
        return torch.ones(size=(5, 4))

    @property
    def labels(self) -> ArrayLike:
        return torch.arange(0, 5)

    @property
    def scores(self) -> ArrayLike:
        return torch.zeros(size=(5, 5))


class DummyXAITKObjectDetectionTarget(DummyObjectDetectionTarget):
    """5x Target data entries per image with labels [0-4]"""

    @property
    def scores(self) -> ArrayLike:
        return torch.zeros(size=(5,))


@pytest.fixture
def dummy_model_od() -> od.Model:
    """Creates and returns a dummy maite-compliant model"""

    class DummyModel:
        metadata = {"id": "fake-id"}

        def __call__(self, input_batch: Sequence[ArrayLike]) -> Sequence[od.ObjectDetectionTarget]:
            return [DummyObjectDetectionTarget() for _ in input_batch]

    return DummyModel()


@pytest.fixture
def dummy_cleaning_dataset_od():
    """Creates and returns a dummy maite-compliant dataset"""

    def _dummy_cleaning_dataset_od(offset_box: bool = True) -> od.Dataset:
        class CleaningObjectDetectionTarget:
            def __init__(self, ind: int):
                self.ind = ind

            """Cleaning OD Target"""

            @property
            def boxes(self) -> ArrayLike:
                boxes = torch.tensor([4, 4, 24, 24]).tile(5, 1)
                if offset_box and self.ind == 9:
                    boxes[0][0:2] = 5
                    boxes[0][2:4] = 16
                return boxes

            @property
            def labels(self) -> ArrayLike:
                return torch.arange(0, 5)

            @property
            def scores(self) -> ArrayLike:
                return torch.zeros(size=(5, 5))

        class DummyDataset:
            """Dataset with 50 1x32x32 CHW images"""

            metadata = {"id": "Dummy_OD_Dataset"}

            images = torch.tensor(RNG.random((50, 1, 32, 32)))
            images[10] = images[0]  # add duplicate
            images[15] = 0.0  # add outliers
            images[16] = 1.0  # add outliers
            images[20] = images[2] * 0.99  # add near duplicate

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, ind: int):
                return self.images[ind], CleaningObjectDetectionTarget(ind), {"id": ind, "dummy": torch.ones((5,))}

        return DummyDataset()

    return _dummy_cleaning_dataset_od


@pytest.fixture
def dummy_dataset_od_with_target_metadata() -> od.Dataset:
    """Creates and returns a dummy maite-compliant dataset"""

    class DummyDataset:
        """Dataset with 10 3x16x16 CHW images"""

        metadata = {
            "id": "Dummy_OD_Dataset_With_Target_Metadata",
            "index2label": {"0": "cat", "1": "dog", "2": "bird", "3": "fish", "4": "horse"},
        }

        images = torch.ones(size=(10, 3, 16, 16))

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, ind: int):
            return self.images[ind], DummyObjectDetectionTarget(), {"id": ind, "dummy": 0, "target": [0, 1, 2, 3, 4]}

    return DummyDataset()


@pytest.fixture
def dummy_dataset_od() -> od.Dataset:
    """Creates and returns a dummy maite-compliant dataset"""

    class DummyDataset:
        """Dataset with 10 3x16x16 CHW images"""

        def __init__(self):
            # FIXME: this should be private to avoid using internals
            self.images = torch.ones(size=(10, 3, 16, 16))
            # FIXME: this needs a index2label entry
            self.metadata = {"id": type(self).__name__}

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, ind: int):
            return self.images[ind], DummyObjectDetectionTarget(), {"id": ind, "dummy": 0}

    return DummyDataset()


@pytest.fixture
def dummy_metric_od() -> od.Metric:
    """Creates and returns a dummy maite-compliant metric"""

    class DummyMetric:
        return_key = "fake-id"
        metadata = {"id": "fake-id"}

        def update(self, preds: Sequence[Any], targets: Sequence[Any]) -> None:
            pass

        def compute(self) -> dict[str, Any]:
            return {self.return_key: 1.0}

        def reset(self) -> None:
            pass

    return DummyMetric()


@pytest.fixture
def dummy_model_ic() -> ic.Model:
    """Creates and returns a dummy maite-compliant model"""

    class DummyModel:
        index2label: dict[str, Hashable] = {0: "dummy_0", 1: "dummy_1"}

        def __call__(self, input_batch: Sequence[ArrayLike]) -> Sequence[Any]:
            self.targets = RNG.random((len(input_batch), 2))

            return self.targets

    return DummyModel()


@pytest.fixture
def dummy_dataset_ic() -> ic.Dataset:
    class DummyDataset:
        def __init__(self):
            n = 10

            # FIXME all instance attributes should be private to avoid depending on anything non-MAITE compliant
            self.data = np_unstack(RNG.random((n, 3, 16, 16)))

            self.targets = np_unstack(RNG.random((n, 2)))
            for data_index in range(n):
                self.targets[data_index][data_index % 2] = 1

            self._metadata = [{"some_metadata": i} for i in range(n)]
            self.metadata = {"id": type(self).__name__}

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, ind: int) -> tuple[np.ndarray, np.ndarray, dict]:
            return self.data[ind], self.targets[ind], self._metadata[ind]

    return DummyDataset()


@pytest.fixture
def dummy_metric_ic() -> ic.Metric:
    """Creates and returns a dummy maite-compliant metric"""

    class DummyMetric:
        return_key = "metric_1"

        def update(self, preds: Sequence[Any], targets: Sequence[Any]) -> None:
            pass

        def compute(self) -> dict[str, Any]:
            return {self.return_key: 1.0}

        def reset(self) -> None:
            pass

    return DummyMetric()


@pytest.fixture
def threshold_od() -> float:
    """Threshold for OD"""
    return 0.3


@pytest.fixture
def metric_results() -> dict:
    """Metric results dictionary"""
    return {
        "map50": 0.12,
        "airports": 0.43,
        "elephants": 0.56,
    }


@pytest.fixture(scope="session")
def reallabel_config_od():
    """Default output configuration settings from the reallabel
    panel app"""
    pytest.importorskip("reallabel")
    # import inside the fixture to improve test startup time
    from jatic_ri.object_detection._panel.configurations.reallabel_app import RealLabelApp as RealLabelAppOD

    reallabel_app = RealLabelAppOD()
    reallabel_app._run_export()
    return reallabel_app.output_test_stages["reallabel_test_stage"]


@pytest.fixture(scope="session")
def survivor_config_od():
    """Default output configuration settings from the survivor
    panel app"""
    # import inside the fixture to improve test startup time
    pytest.importorskip("survivor")
    from jatic_ri.object_detection._panel.configurations.survivor_app import SurvivorAppOD

    survivor_app = SurvivorAppOD()
    survivor_app._run_export()
    return survivor_app.output_test_stages["survivor_test_stage"]


@pytest.fixture(scope="session")
def survivor_config_ic():
    pytest.importorskip("survivor")
    """Default output configuration settings from the survivor
    panel app"""
    # import inside the fixture to improve test startup time
    from jatic_ri.image_classification._panel.configurations.survivor_app import SurvivorAppIC

    survivor_app = SurvivorAppIC()
    survivor_app._run_export()
    return survivor_app.output_test_stages["survivor_test_stage"]


@pytest.fixture(scope="session")
def nrtk_config_od():
    # import inside the fixture to improve test startup time
    from jatic_ri.object_detection._panel.configurations.nrtk_app import NRTKAppOD

    nrtk_app = NRTKAppOD()
    nrtk_app.panel()
    nrtk_app.perturber_select.value = BrightnessPerturber
    nrtk_app.factory_selector.value = OneStepPerturbImageFactory
    nrtk_app.theta_key.value = "factor"
    nrtk_app.theta_value.value = 10.0
    nrtk_app.name_input.value = "TestFactory"
    nrtk_app.add_test_stage_callback(None)
    nrtk_app._run_export()
    return nrtk_app.output_test_stages["NRTKAppOD_0"]


@pytest.fixture(scope="session")
def nrtk_config_ic():
    # import inside the fixture to improve test startup time
    from jatic_ri.image_classification._panel.configurations.nrtk_app import NRTKAppIC

    nrtk_app = NRTKAppIC()
    nrtk_app.panel()
    nrtk_app.perturber_select.value = BrightnessPerturber
    nrtk_app.factory_selector.value = OneStepPerturbImageFactory
    nrtk_app.theta_key.value = "factor"
    nrtk_app.theta_value.value = 10.0
    nrtk_app.name_input.value = "TestFactory"
    nrtk_app.add_test_stage_callback(None)
    nrtk_app._run_export()
    return nrtk_app.output_test_stages["NRTKAppIC_0"]


@pytest.fixture(scope="session")
def xaitk_config_od():
    # import inside the fixture to improve test startup time
    from jatic_ri.object_detection._panel.configurations.xaitk_app import XAITKAppOD

    xaitk_app = XAITKAppOD()
    xaitk_app._run_export()
    return xaitk_app.output_test_stages["XAITKAppOD_0"]


@pytest.fixture(scope="session")
def xaitk_config_ic():
    # import inside the fixture to improve test startup time
    from jatic_ri.image_classification._panel.configurations.xaitk_app import XAITKAppIC

    xaitk_app = XAITKAppIC()
    xaitk_app._run_export()
    return xaitk_app.output_test_stages["XAITKAppIC_0"]


@pytest.fixture(scope="session")
def feasibility_config_od():
    return {
        "TYPE": "DatasetFeasibilityTestStage",
    }


@pytest.fixture(scope="session")
def feasibility_config_ic(feasibility_config_od):
    return feasibility_config_od


@pytest.fixture(scope="session")
def bias_config_od():
    return {
        "TYPE": "DatasetBiasTestStage",
    }


@pytest.fixture(scope="session")
def bias_config_ic(bias_config_od):
    return bias_config_od


@pytest.fixture(scope="session")
def cleaning_config_od():
    return {
        "TYPE": "DatasetCleaningTestStage",
    }


@pytest.fixture(scope="session")
def cleaning_config_ic(cleaning_config_od):
    return cleaning_config_od


@pytest.fixture(scope="session")
def shift_config_od():
    return {
        "TYPE": "DatasetShiftTestStage",
    }


@pytest.fixture(scope="session")
def shift_config_ic(shift_config_od):
    return shift_config_od


@pytest.fixture(scope="session")
def baseline_eval_config_od():
    # baseline evaluation (maite) requires no data inside the config
    return {
        "TYPE": "BaselineEvaluationTestStage",
    }


@pytest.fixture(scope="session")
def baseline_eval_config_ic(baseline_eval_config_od):
    # baseline evaluation (maite) requires no data inside the config
    return baseline_eval_config_od


@pytest.fixture(scope="session")
def fake_ic_dataset_default() -> FakeICDataset:
    """
    Fixture for getting the default fake Image Classification dataset with behaviors as described in /tests/fake_ic_classes.py

    IMPORTANT - if the default fake values do not meet your testing requirements, consult the RI Team (in GitLab)
    before implementing a custom instance of this class.  As much as possible, we would like to have as many test cases as possible use
    the default attributes/fixture, so expanding the defaults to cover additional scenarios is likely preferable to creating different fake
    data for different scenarios.
    """
    return FakeICDataset()


@pytest.fixture(scope="session")
def fake_ic_model_default() -> FakeICModel:
    """
    Fixture for getting the default fake Image Classification model with behaviors as described in /tests/fake_ic_classes.py

    IMPORTANT - if the default fake values do not meet your testing requirements, consult the RI Team (in GitLab)
    before implementing a custom instance of this class.  As much as possible, we would like to have as many test cases as possible use
    the default attributes/fixture, so expanding the defaults to cover additional scenarios is likely preferable to creating different fake
    data for different scenarios.
    """
    return FakeICModel()


@pytest.fixture(scope="session")
def fake_ic_metric_default() -> FakeICMetric:
    """
    Fixture for getting the default Fake Image Classification metric with behaviors as described in /tests/fake_ic_classes.py

    IMPORTANT - if the default fake values do not meet your testing requirements, consult the RI team (in GitLab)
    before implementing a custom instance of this class.  As much as possible, we would like to have as many test
    cases as possible use the default attributes/fixture, so expanding the defaults to cover additional scenarios
    is likely preferable to creating different fake data for different scenarios.
    """
    return FakeICMetric()


@pytest.fixture(scope="session")
def fake_od_dataset_default() -> FakeODDataset:
    """
    Fixture for getting the default Fake Object Detection Dataset with behaviors as described in /tests/fake_od_classes.py

    IMPORTANT - if the default fake values do not meet your testing requirements, consult the RI team (in GitLab)
    before implementing a custom instance of this class.  As much as possible, we would like to have as many test
    cases as possible use the default attributes/fixture, so expanding the defaults to cover additional scenarios
    is likely preferable to creating different fake data for different scenarios.
    """
    return FakeODDataset()


@pytest.fixture(scope="session")
def fake_od_model_default() -> FakeODModel:
    """
    Fixture for getting the default Fake Object Detection model with behaviors as described in /tests/fake_od_classes.py

    IMPORTANT - if the default fake values do not meet your testing requirements, consult the RI team (in GitLab)
    before implementing a custom instance of this class.  As much as possible, we would like to have as many test
    cases as possible use the default attributes/fixture, so expanding the defaults to cover additional scenarios
    is likely preferable to creating different fake data for different scenarios.
    """
    return FakeODModel()


@pytest.fixture(scope="session")
def fake_od_metric_default() -> FakeODMetric:
    """
    Fixture for getting the default Fake Object Detection metric with behaviors as described in /tests/fake_od_classes.py

    IMPORTANT - if the default fake values do not meet your testing requirements, consult the RI team (in GitLab)
    before implementing a custom instance of this class.  As much as possible, we would like to have as many test
    cases as possible use the default attributes/fixture, so expanding the defaults to cover additional scenarios
    is likely preferable to creating different fake data for different scenarios.
    """
    return FakeODMetric()


@pytest.fixture(scope="session")
def fake_od_dataset_reallabel_only() -> FakeODDataset:
    """
    NOTE - We should refactor the RealLabel test stage tests so this isn't necessary.
    The following tests fail AssertionErrors with the default FakeODDataset if dataset length is not 1
       * test_reallabel_test_stage_collect_report_consumables
       * test_reallabel_test_stage_collect_metrics_cached_data
    This could be just because otherwise valid output calculations are changing, but that assumption needs to be
    confirmed before changing the tests.
    Worth noting, too, that some survivor tests fail if dataset length is not 6... but 6 is better default than 1
    """
    from tests.fake_od_classes import DEFAULT_OD_DATASET_IMAGES, DEFAULT_OD_DATASET_TARGETS, DEFAULT_OD_DATUM_METADATA

    # To make the tests pass for now, trucate the default dataset to just one item
    return FakeODDataset(
        images=DEFAULT_OD_DATASET_IMAGES[:1],
        targets=DEFAULT_OD_DATASET_TARGETS[:1],
        datum_metadata=DEFAULT_OD_DATUM_METADATA[:1],
    )


@pytest.fixture(scope="session")
def dummy_cpu_image_batch():
    return torch.testing.make_tensor(
        (3, 3, 11, 17),
        low=0,
        high=torch.iinfo(torch.uint8).max,
        dtype=torch.uint8,
        device="cpu",
    )


@pytest.fixture
def yolo_dataset(artifact_dir):
    """Generate yolo dataset one image and two classes"""
    ## Set up dataset
    classes = ["cat", "dog"]
    num_images_per_class = 3
    img_shape = (64, 128)

    root_dir = Path(artifact_dir) / "temp_yolo_dataset"
    split = "test"
    os.makedirs(root_dir / split, exist_ok=True)
    for class_name in classes:
        class_dir = root_dir / split / class_name
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_images_per_class):
            img = Image.new("RGB", img_shape, color=(i, i, i))
            img.save(class_dir / f"{i}_{class_name}.jpg")

    return root_dir
