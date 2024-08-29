"""Test baseline evalutation"""

import maite.protocols.object_detection as od
import pytest


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

    class DummyDataset(od.Dataset):
        def __len__(self) -> None:
            pass

        def __getitem__(self) -> None:
            pass

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
