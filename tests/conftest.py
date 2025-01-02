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
                return self.images[ind], LintingObjectDetectionTarget(ind), {"id": ind, "dummy": torch.ones((5,))}

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
            return self.images[ind], DummyObjectDetectionTarget(), {"id": ind, "dummy": 0, "target": [0, 1, 2, 3, 4]}

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
            return self.images[ind], DummyObjectDetectionTarget(), {"id": ind, "dummy": 0}

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
            return self.images[ind], DummyXAITKObjectDetectionTarget(), {"id": ind, "dummy": 0}

    return DummyDataset()


@pytest.fixture
def dummy_metric_od() -> od.Metric:
    """Creates and returns a dummy maite-compliant metric"""

    class DummyMetric:
        return_key = "fake_metric"

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

@pytest.fixture
def metric_batch_input_data_ic() -> dict[str, list[list[torch.Tensor]]]:
    """
     The following prediction and ground truth (target) data was generated by running a MAITE predict()
     workflow with compliant wrapped CIFAR-10 dataset and a pretrained model "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
     The results were altered slightly so that the default configurations of the wrapped accuracy and f1 score metrics should return accuracy=0.90
     and f1_score=0.85
    
     In this data there are 20 images total- 1 list containing 5 batches, each batch contains a list with 4 1-dimensional Tensors representing
     containing the 10 values representing the pseudo-probabilities for each of the 10 classes.
    
     Each __call__ implementation of any given ic.Model should return one Sequence[ArrayLike], where
     each item in the sequence corresponds to one prediction in the batch.  So the input to metric.update()
     should be one of the four batch sequences.
    """
    
    IC_PREDICTION_DATA: list[list[torch.Tensor]] = [
            [
                    torch.tensor([-0.2832, -0.4654, -0.5866,  3.3975, -0.4545,  0.0570, -0.3015, -0.4808, -0.1757, -0.5573]),
                    torch.tensor([-0.1634, -0.1646, -0.6448, -0.3842, -0.5051, -0.4712, -0.4654, -0.6231, 3.4097, -0.6745]),
                    torch.tensor([-0.0775,  0.1461, -0.5379, -0.4177, -0.5669, -0.6365, -0.4918, -0.6450, 3.2838, -0.5834]),
                    torch.tensor([ 2.2617, -0.7592,  1.2758, -0.3185, -0.1752, -0.6627, -0.4864, -0.5339, -0.1966, -0.5145])
            ],
            [
                    torch.tensor([-0.3537, -0.2256, -0.1936, -0.3826, -0.3499, -0.5073,  3.3957, -0.5077, -0.2492, -0.5025]),
                    torch.tensor([-0.7112, -0.6057, -0.5041,  0.1275, -0.3270,  0.1434,  3.3354, -0.4935, -0.4712, -0.6506]),
                    torch.tensor([-0.3078,  3.4402, -0.2747, -0.6082, -0.5243, -0.6291, -0.3965, -0.3440, -0.2732,  0.2556]),
                    torch.tensor([-0.2937, -0.4419, -0.4337, -0.2989, -0.3909, -0.3279,  3.3799, -0.5901, -0.2065, -0.4537])
            ],
            [
                    torch.tensor([-0.5285, -0.4450, -0.6761,  3.3978, -0.3468,  0.0582, -0.2942, -0.3642, -0.1514, -0.5457]),
                    torch.tensor([-0.5058,  3.3764, -0.3019, -0.5000, -0.5454, -0.6148, -0.3679, -0.4290, -0.2001,  0.3801]),
                    torch.tensor([ 2.9787, -0.6276,  0.8979, -0.4128, -0.6522, -0.2636, -0.6498, -0.5331, -0.3570, -0.3990]),
                    torch.tensor([-0.3237,  0.0699, -0.4151, -0.4021, -0.1599, -0.3998, -0.3533, -0.5239, -0.4047,  3.3992])
            ],
            [
                    torch.tensor([-0.5112, -0.7362, -0.4982,  0.1223, -0.3884,  3.1434, -0.4271, -0.3184, -0.2962, -0.4724]),
                    torch.tensor([-0.3638, -0.2724, -0.2569, -0.3294, -0.0768, -0.2099, -0.3563,  3.5357, -0.3638, -0.4512]),
                    torch.tensor([-0.3090, -0.0466, -0.3871, -0.2501, -0.1755, -0.4787, -0.3690, -0.5002, -0.3567,  3.3851]),
                    torch.tensor([ 0.0523, -0.5057, -0.5043, -0.4131, -0.4282, -0.5697, -0.4123, -0.7042, 3.3588, -0.7597])
            ],
            [
                    torch.tensor([-0.4867, -0.6812, -0.2931,  0.2918, -0.5057,  3.1666, -0.4555, -0.3837, -0.3678, -0.6348]),
                    torch.tensor([-0.2538, -0.1592, -0.3918, -0.3946, -0.1470, -0.3973, -0.3574,  3.4823, -0.2857, -0.2486]),
                    torch.tensor([-0.2821, -0.3694, -0.5154, -0.2898, -0.3963, -0.5335, -0.4082, -0.7029, 3.4225, -0.6853]),
                    torch.tensor([-0.4688, -0.3213, -0.0638, -0.3976, -0.2782, -0.3688,  3.4971, -0.5379, -0.4190, -0.5718])
            ]
    ]
    
    """And these are the corresponding ground-truth one-hots."""
    
    IC_TARGET_DATA: list[list[torch.Tensor]] = [
            [
                    torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
                    torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
                    torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
                    torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            ],
            [
                    torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]),
                    torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]),
                    torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
                    torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
            ],
            [
                    torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
                    torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
                    torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                    torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
            ],
            [
                    torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
                    torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]), # altered 7 to 8
                    torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
                    torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
            ],
            [
                    torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
                    torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]), # altered 7 to 8
                    torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
                    torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
            ]
    ]

    return {"predictions": IC_PREDICTION_DATA, "targets": IC_TARGET_DATA}
