"""Test baseline evalutation"""

import tempfile
from collections.abc import Sequence
from typing import Any

import maite.protocols.image_classification as ic
import maite.protocols.object_detection as od
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
        return torch.zeros(size=(5, ))


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
            images[20] = images[2]*0.99  # add near duplicate

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, ind: int):
                return self.images[ind], LintingObjectDetectionTarget(ind), {"dummy": torch.ones((5,))}

        return DummyDataset()
    return _dummy_linting_dataset_od


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
        _return_key = 'metric_1'

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
        def __call__(self, input_batch: Sequence[ArrayLike]) -> Sequence[Any]:
            self.targets = RNG.random(input_batch)

            return [self.targets for _ in input_batch]

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
            return (self._data[ind], self.targets[ind], self.metadata[ind])

    return DummyDataset()


@pytest.fixture
def dummy_metric_ic() -> ic.Metric:
    """Creates and returns a dummy maite-compliant metric"""

    class DummyMetric:
        def update(self, preds: Sequence[Any], targets: Sequence[Any]) -> None:
            pass

        def compute(self) -> dict[str, Any]:
            return {"dummy": 1.0}

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
        USA_SUMMER_DATA_IMAGERY_DIR, USA_SUMMER_DATA_METADATA_FILE_PATH,
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
        'map50': 0.12,
        'airports': 0.43,
        'elephants': 0.56,
        }
