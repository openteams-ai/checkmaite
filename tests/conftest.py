"""Test baseline evalutation"""

import maite.protocols.object_detection as od
import pytest
import torch


@pytest.fixture
def dummy_model() -> od.Model:
    """Creates and returns a dummy maite-compliant model"""

    class DummyModel(od.Model):
        def __call__(self) -> None:
            return None

    return DummyModel()


@pytest.fixture
def dummy_dataset() -> od.Dataset:
    """Creates and returns a dummy maite-compliant dataset"""

    class DummyObjectDetectionTarget:
        boxes = torch.ones(size=(25, 4))
        labels = torch.ones(size=(25,))
        scores = torch.zeros(size=(25,))


    class DummyDataset(od.Dataset):
        images = torch.ones(size=(25, 1, 16, 16))

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, ind: int):
            return self.images[ind], DummyObjectDetectionTarget(), {}

    return DummyDataset()


@pytest.fixture
def dummy_metric() -> od.Metric:
    """Creates and returns a dummy maite-compliant metric"""

    class DummyMetric(od.Metric):
        def update(self) -> None:
            pass

        def compute(self) -> None:
            pass

        def reset(self) -> None:
            pass

    return DummyMetric()
