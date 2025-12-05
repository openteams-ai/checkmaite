import copy

import pytest

from jatic_ri.core._common.dataeval_shift_capability import (
    DataevalShiftBase,
    DataevalShiftConfig,
    collect_drift,
    collect_ood,
)


@pytest.fixture(scope="module")
def dummy_shift_capability():
    class DummyShiftCapability(DataevalShiftBase):
        pass

    return DummyShiftCapability


@pytest.fixture
def test_config():
    return DataevalShiftConfig(dim=32)


def test_run_and_collect(dummy_shift_capability, fake_od_dataset_default):
    dev_dataset = fake_od_dataset_default
    op_dataset = copy.deepcopy(fake_od_dataset_default)

    capability = dummy_shift_capability()
    output = capability.run(use_cache=False, datasets=[dev_dataset, op_dataset])

    report = output.collect_report_consumables(threshold=0.0)

    assert report  # smoke-test for non-empty report


def test_run_and_collect_drift(dummy_shift_capability, fake_od_dataset_default, test_config):
    capability = dummy_shift_capability()
    outputs = capability._run_drift(
        dataset_1=fake_od_dataset_default, dataset_2=fake_od_dataset_default, config=test_config
    )

    assert outputs.model_dump()  # smoke-test for Pydantic model

    assert collect_drift(
        "fake-deck", drift_outputs=outputs, dataset_ids=["DummyDataset1", "DummyDataset2"]
    )  # smoke-test for non-empty collect


def test_run_and_collect_ood(dummy_shift_capability, fake_od_dataset_default, test_config):
    dataset_2 = copy.deepcopy(fake_od_dataset_default)

    capability = dummy_shift_capability()
    outputs = capability._run_ood(dataset_1=fake_od_dataset_default, dataset_2=dataset_2, config=test_config)
    assert outputs.model_dump()  # smoke-test for Pydantic model

    assert collect_ood(
        "fake-deck",
        ood_outputs=outputs,
        dataset_ids=["DummyDataset1", "DummyDataset2"],
    )  # smoke-test for non-empty collect
