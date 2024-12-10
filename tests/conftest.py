"""Test baseline evalutation"""

import tempfile
from collections.abc import Sequence
from typing import Any, Hashable
from pathlib import Path
import os

import maite.protocols.image_classification as ic
import maite.protocols.object_detection as od
from nrtk.impls.perturb_image_factory.generic.one_step import OneStepPerturbImageFactory
from nrtk.impls.perturb_image.generic.PIL.enhance import BrightnessPerturber
import numpy as np
import pytest
import torch
from maite.protocols import ArrayLike
from tests.testing_utilities.example_maite_objects import (  # noqa: E501
    create_maite_wrapped_metric,
    FMOWDetectionDataset,
    USA_SUMMER_DATA_IMAGERY_DIR,
    USA_SUMMER_DATA_METADATA_FILE_PATH,
    Yolov5sModel,
    YOLOV5S_USA_ALL_SEASONS_V1_MODEL_PATH,
)

import jatic_ri


RNG = np.random.default_rng(seed=42)

# Set the test_stage default cache root to temp path
jatic_ri.DEFAULT_CACHE_ROOT = tempfile.gettempdir()

@pytest.fixture
def artifact_dir(tmpdir):
    # use env var if available, otherwise use tmpdir under the repo root
    return Path(os.environ.get('ARTIFACT_DIR', tmpdir))


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
        def __call__(self, input_batch: Sequence[ArrayLike]) -> Sequence[od.ObjectDetectionTarget]:
            return [DummyObjectDetectionTarget() for _ in input_batch]

    return DummyModel()


@pytest.fixture
def dummy_xaitk_model() -> od.Model:
    """Creates and returns a dummy maite-compliant model"""

    class DummyModel:
        index2label: dict[str, Hashable] = {0: "dummy_0", 1: "dummy_1", 2: "dummy_2", 3: "dummy_3", 4: "dummy_4"}

        def __call__(self, input_batch: Sequence[ArrayLike]) -> Sequence[od.ObjectDetectionTarget]:
            return [DummyXAITKObjectDetectionTarget() for _ in input_batch]

    return DummyModel()


@pytest.fixture
def dummy_linting_dataset_od():
    """Creates and returns a dummy maite-compliant dataset"""

    def _dummy_linting_dataset_od(offset_box: bool = True) -> od.Dataset:
        class LintingObjectDetectionTarget:
            def __init__(self, ind: int):
                self.ind = ind

            """Linting OD Target"""

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

            images = torch.tensor(RNG.random((50, 1, 32, 32)))
            images[10] = images[0]  # add duplicate
            images[15] = 0.0  # add outliers
            images[16] = 1.0  # add outliers
            images[20] = images[2] * 0.99  # add near duplicate

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, ind: int):
                return self.images[ind], LintingObjectDetectionTarget(ind), {"dummy": torch.ones((5,))}

        return DummyDataset()

    return _dummy_linting_dataset_od


@pytest.fixture
def dummy_dataset_od_with_target_metadata() -> od.Dataset:
    """Creates and returns a dummy maite-compliant dataset"""

    class DummyDataset:
        """Dataset with 10 1x16x16 CHW images"""

        images = torch.ones(size=(10, 1, 16, 16))

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, ind: int):
            return self.images[ind], DummyObjectDetectionTarget(), {"dummy": 0, "target": [0, 1, 2, 3, 4]}

    return DummyDataset()

@pytest.fixture
def dummy_dataset_od() -> od.Dataset:
    """Creates and returns a dummy maite-compliant dataset"""

    class DummyDataset:
        """Dataset with 10 1x16x16 CHW images"""

        images = torch.ones(size=(10, 1, 16, 16))

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, ind: int):
            return self.images[ind], DummyObjectDetectionTarget(), {"dummy": 0}

    return DummyDataset()


@pytest.fixture
def dummy_xaitk_dataset() -> od.Dataset:
    """Creates and returns a dummy maite-compliant dataset"""

    class DummyDataset:
        """Dataset with 10 1x16x16 CHW images"""

        images = torch.ones(size=(10, 1, 16, 16))

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, ind: int):
            return self.images[ind], DummyXAITKObjectDetectionTarget(), {"dummy": 0}

    return DummyDataset()


@pytest.fixture
def dummy_metric_od() -> od.Metric:
    """Creates and returns a dummy maite-compliant metric"""

    class DummyMetric:
        _return_key = "metric_1"

        def update(self, preds: Sequence[Any], targets: Sequence[Any]) -> None:
            pass

        def compute(self) -> dict[str, Any]:
            return {self._return_key: 1.0}

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
            self.data = RNG.random((10, 1, 16, 16))

            self.targets = RNG.random((10, 2))

            for data_index in range(self.targets.shape[0]):
                self.targets[data_index, data_index % 2] = 1

            self.metadata = [{"some_metadata": i} for i in range(self.data.shape[0])]

        def __len__(self) -> int:
            return self.data.shape[0]

        def __getitem__(self, ind: int) -> tuple[np.ndarray, np.ndarray, dict]:
            return (self.data[ind], self.targets[ind], self.metadata[ind])

    return DummyDataset()


@pytest.fixture
def dummy_metric_ic() -> ic.Metric:
    """Creates and returns a dummy maite-compliant metric"""

    class DummyMetric:
        _return_key = "metric_1"

        def update(self, preds: Sequence[Any], targets: Sequence[Any]) -> None:
            pass

        def compute(self) -> dict[str, Any]:
            return {self._return_key: 1.0}

        def reset(self) -> None:
            pass

    return DummyMetric()


@pytest.fixture
def model_od_yolov5() -> od.Model:
    """Yolov5s USA All seasons V1 model for OD"""
    return Yolov5sModel(
        model_path=str(YOLOV5S_USA_ALL_SEASONS_V1_MODEL_PATH),
        transforms=None,
        device="cpu",
    )


@pytest.fixture
def dataset_od_fwow() -> od.Dataset:
    """FWOW detection dataset using USA summer for OD"""
    return FMOWDetectionDataset(
        USA_SUMMER_DATA_IMAGERY_DIR,
        USA_SUMMER_DATA_METADATA_FILE_PATH,
    )


@pytest.fixture
def metric_od_map() -> od.Metric:
    """mAP 50 metric for OD"""
    return create_maite_wrapped_metric("mAP_50")


@pytest.fixture
def threshold_od() -> float:
    """threshold for OD, realistic value to be paired with metric_od_map"""
    return 0.3


@pytest.fixture
def metric_results() -> dict:
    """metric results dictionary"""
    return {
        "map50": 0.12,
        "airports": 0.43,
        "elephants": 0.56,
    }


@pytest.fixture(scope='session')
def reallabel_config_od():
    """Default output configuration settings from the reallabel
    panel app"""
    # import inside the fixture to improve test startup time
    from jatic_ri.object_detection._panel.configurations.reallabel_app import RealLabelApp as RealLabelAppOD
    reallabel_app = RealLabelAppOD()
    reallabel_app._run_export()
    return reallabel_app.output_test_stages["reallabel_test_stage"]

@pytest.fixture(scope='session')
def survivor_config_od():
    """Default output configuration settings from the survivor
    panel app"""
    # import inside the fixture to improve test startup time
    from jatic_ri.object_detection._panel.configurations.survivor_app import SurvivorApp as SurvivorAppOD
    survivor_app = SurvivorAppOD()
    survivor_app._run_export()
    return survivor_app.output_test_stages["survivor_test_stage"]

@pytest.fixture(scope='session')
def survivor_config_ic():
    """Default output configuration settings from the survivor
    panel app"""
    # import inside the fixture to improve test startup time
    from jatic_ri.image_classification._panel.configurations.survivor_app import SurvivorApp as SurvivorAppIC
    survivor_app = SurvivorAppIC()
    survivor_app._run_export()
    return survivor_app.output_test_stages["survivor_test_stage"]

@pytest.fixture(scope='session')
def nrtk_config_od():
    # import inside the fixture to improve test startup time
    from jatic_ri.object_detection._panel.configurations.nrtk_app import NRTKApp as NRTKAppOD
    nrtk_app = NRTKAppOD()
    nrtk_app.panel()
    nrtk_app.perturber_select.value = BrightnessPerturber
    nrtk_app.factory_selector.value = OneStepPerturbImageFactory
    nrtk_app.theta_key.value = 'factor'
    nrtk_app.theta_value.value = 10.0
    nrtk_app.name_input.value = "TestFactory"
    nrtk_app.add_test_stage_callback(None)
    nrtk_app._run_export()
    nrtk_config = nrtk_app.output_test_stages["NRTKApp_0"]
    nrtk_config['TYPE'] = 'NRTKTestStage'  # temporary fix
    return nrtk_config

@pytest.fixture(scope='session')
def nrtk_config_ic():
    # import inside the fixture to improve test startup time
    from jatic_ri.image_classification._panel.configurations.nrtk_app import NRTKApp as NRTKAppIC
    nrtk_app = NRTKAppIC()
    nrtk_app.panel()
    nrtk_app.perturber_select.value = BrightnessPerturber
    nrtk_app.factory_selector.value = OneStepPerturbImageFactory
    nrtk_app.theta_key.value = 'factor'
    nrtk_app.theta_value.value = 10.0
    nrtk_app.name_input.value = "TestFactory"
    nrtk_app.add_test_stage_callback(None)
    nrtk_app._run_export()
    nrtk_config = nrtk_app.output_test_stages["NRTKApp_0"]
    nrtk_config['TYPE'] = 'NRTKTestStage'  # temporary fix
    return nrtk_config

@pytest.fixture(scope='session')
def xaitk_config_od():
    # import inside the fixture to improve test startup time
    from jatic_ri.object_detection._panel.configurations.xaitk_app import XAITKApp as XAITKAppOD
    xaitk_app = XAITKAppOD()
    xaitk_app._run_export()
    return xaitk_app.output_test_stages["XAITKApp_0"]

@pytest.fixture(scope='session')
def xaitk_config_ic():
    # import inside the fixture to improve test startup time
    from jatic_ri.image_classification._panel.configurations.xaitk_app import XAITKApp as XAITKAppIC
    xaitk_app = XAITKAppIC()
    xaitk_app._run_export()
    return xaitk_app.output_test_stages["XAITKApp_0"]

@pytest.fixture(scope='session')
def feasibility_config_od():
    return {
        'TYPE': 'DatasetFeasibilityTestStage',
    }

@pytest.fixture(scope='session')
def feasibility_config_ic(feasibility_config_od):
    return feasibility_config_od

@pytest.fixture(scope='session')
def bias_config_od():
    return {
        'TYPE': 'DatasetBiasTestStage',
    }

@pytest.fixture(scope='session')
def bias_config_ic(bias_config_od):
    return bias_config_od

@pytest.fixture(scope='session')
def linting_config_od():  
    return {
        'TYPE': 'DatasetLintingTestStage',
    }

@pytest.fixture(scope='session')
def linting_config_ic(linting_config_od):  
    return linting_config_od

@pytest.fixture(scope='session')
def shift_config_od():
    return {
        'TYPE': 'DatasetShiftTestStage',
    }

@pytest.fixture(scope='session')
def shift_config_ic(shift_config_od):
    return shift_config_od

@pytest.fixture(scope='session')
def baseline_eval_config_od():
    # baseline evaluation (maite) requires no data inside the config
    return {
        'TYPE': 'BaselineEvaluationTestStage',
    }

@pytest.fixture(scope='session')
def baseline_eval_config_ic(baseline_eval_config_od):
    # baseline evaluation (maite) requires no data inside the config
    return baseline_eval_config_od
