"""Test baseline evalutation"""

from collections.abc import Sequence
from typing import Any

import maite.protocols.image_classification as ic
import maite.protocols.object_detection as od
import numpy as np
import pytest
import torch
from maite.protocols import ArrayLike

RNG = np.random.default_rng(seed=42)


class DummyObjectDetectionTarget(od.ObjectDetectionTarget):
    """5x Target data entries per image with labels [0-4]"""

    boxes = torch.ones(size=(5, 4))
    labels = torch.arange(0, 5)
    scores = torch.zeros(size=(5, 5))


class DummyXAITKObjectDetectionTarget(od.ObjectDetectionTarget):
    """5x Target data entries per image with labels [0-4]"""

    boxes = torch.ones(size=(5, 4))
    labels = torch.arange(0, 5)
    scores = torch.zeros(size=(5,))


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
        def update(self, preds: Sequence[Any], targets: Sequence[Any]) -> None:
            pass

        def compute(self) -> dict[str, Any]:
            return {"metric_1": 1.0}

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
