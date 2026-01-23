import copy
from typing import Any

import pytest

from jatic_ri.core._common.dataeval_shift_capability import (
    DataevalShiftBase,
    DataevalShiftConfig,
    collect_drift,
    collect_ood,
)
from jatic_ri.core.report._gradient import HAS_GRADIENT


@pytest.fixture(scope="module")
def dummy_shift_capability():
    class DummyShiftCapability(DataevalShiftBase):
        pass

    return DummyShiftCapability


@pytest.fixture
def test_config():
    return DataevalShiftConfig(dim=32)


@pytest.fixture
def test_run(dummy_shift_capability, fake_od_dataset_default) -> Any:
    dev_dataset = fake_od_dataset_default
    op_dataset = copy.deepcopy(fake_od_dataset_default)

    capability = dummy_shift_capability()
    return capability.run(use_cache=False, datasets=[dev_dataset, op_dataset])  # smoke test


def test_collect_md_report(test_run):
    md = test_run.collect_md_report(threshold=0.0)
    assert md  # smoke test for non-empty markdown report


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_collect_report_consumables(test_run):
    consumables = test_run.collect_report_consumables(threshold=0.0)
    assert consumables  # smoke test for non-empty report consumables


def test_run_drift(dummy_shift_capability, fake_od_dataset_default, test_config):
    capability = dummy_shift_capability()
    outputs = capability._run_drift(
        dataset_1=fake_od_dataset_default,
        dataset_2=fake_od_dataset_default,
        config=test_config,
    )

    assert outputs.model_dump()  # smoke-test for Pydantic model


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_collect_drift(fake_od_dataset_default, dummy_shift_capability, test_config):
    capability = dummy_shift_capability()
    outputs = capability._run_drift(
        dataset_1=fake_od_dataset_default,
        dataset_2=fake_od_dataset_default,
        config=test_config,
    )
    assert collect_drift(
        "fake-deck",
        drift_outputs=outputs,
        dataset_ids=["DummyDataset1", "DummyDataset2"],
    )  # smoke-test for non-empty collect


def test_run_ood(dummy_shift_capability, fake_od_dataset_default, test_config):
    dataset_2 = copy.deepcopy(fake_od_dataset_default)

    capability = dummy_shift_capability()
    outputs = capability._run_ood(
        dataset_1=fake_od_dataset_default,
        dataset_2=dataset_2,
        config=test_config,
    )

    assert outputs.model_dump()  # smoke-test for Pydantic model


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_collect_ood(fake_od_dataset_default, dummy_shift_capability, test_config):
    dataset_2 = copy.deepcopy(fake_od_dataset_default)
    capability = dummy_shift_capability()
    outputs = capability._run_ood(
        dataset_1=fake_od_dataset_default,
        dataset_2=dataset_2,
        config=test_config,
    )
    assert collect_ood(
        "fake-deck",
        ood_outputs=outputs,
        dataset_ids=["DummyDataset1", "DummyDataset2"],
    )  # smoke-test for non-empty collect
