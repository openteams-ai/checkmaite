import copy
from typing import Any

import numpy as np
import pytest
from dataeval.shift import DriftOutput

from checkmaite.core._common.dataeval_shift_capability import (
    DataevalShiftBase,
    DataevalShiftConfig,
    DataevalShiftDriftOutputs,
    DataevalShiftOODKNNOutput,
    DataevalShiftOODOutputs,
    DataevalShiftOutputs,
    DataevalShiftRun,
    collect_drift,
    collect_ood,
)
from checkmaite.core.report._gradient import HAS_GRADIENT


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


def test_collect_md_report(fake_od_dataset_default, test_config):
    outputs = DataevalShiftOutputs(
        drift=DataevalShiftDriftOutputs(
            mmd=DriftOutput(False, 0.5, 0.1, "mmd", {"p_val": 0.9}),
            cvm=DriftOutput(False, 0.5, 0.1, "cvm", {"p_val": 0.9, "feature_drift": np.array([False])}),
            ks=DriftOutput(False, 0.5, 0.1, "ks", {"p_val": 0.9, "feature_drift": np.array([False])}),
        ),
        ood=DataevalShiftOODOutputs(
            ood_knn=DataevalShiftOODKNNOutput(
                is_ood=np.array([False, True]),
                instance_score=np.array([0.1, 0.9]),
                feature_score=None,
            )
        ),
    )
    run = DataevalShiftRun(
        capability_id="DataevalShift",
        config=test_config,
        dataset_metadata=[fake_od_dataset_default.metadata, fake_od_dataset_default.metadata],
        model_metadata=[],
        metric_metadata=[],
        outputs=outputs,
    )

    md = run.collect_md_report(threshold=0.0)
    assert md  # smoke test for non-empty markdown report


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_collect_report_consumables(test_run):
    with pytest.warns(DeprecationWarning):
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
