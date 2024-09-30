"""Test baseline evalutation"""

from collections.abc import Sequence
from typing import Any

import maite.protocols.object_detection as od
import pytest
import torch
from maite.protocols import ArrayLike


class DummyObjectDetectionTarget(od.ObjectDetectionTarget):
    """5x Target data entries per image with labels [0-4]"""

    boxes = torch.ones(size=(5, 4))
    labels = torch.arange(0, 5)
    scores = torch.zeros(size=(5, 5))


@pytest.fixture
def dummy_model() -> od.Model:
    """Creates and returns a dummy maite-compliant model"""

    class DummyModel(od.Model):
        def __call__(self, input_batch: Sequence[ArrayLike]) -> Sequence[od.ObjectDetectionTarget]:
            return [DummyObjectDetectionTarget() for _ in input_batch]

    return DummyModel()


@pytest.fixture
def dummy_dataset() -> od.Dataset:
    """Creates and returns a dummy maite-compliant dataset"""

    class DummyDataset(od.Dataset):
        """Dataset with 10 1x16x16 CHW images"""

        images = torch.ones(size=(10, 1, 16, 16))

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, ind: int):
            return self.images[ind], DummyObjectDetectionTarget(), {"dummy": 0}

    return DummyDataset()


@pytest.fixture
def dummy_metric() -> od.Metric:
    """Creates and returns a dummy maite-compliant metric"""

    class DummyMetric(od.Metric):
        def update(self, preds: Sequence[Any], targets: Sequence[Any]) -> None:
            pass

        def compute(self) -> None:
            pass

        def reset(self) -> None:
            pass

    return DummyMetric()
