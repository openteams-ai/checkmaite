"""Test drift methods in BaseShiftTestStage"""

import copy
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri._common.test_stages.impls.dataeval_shift_test_stage import (
    DataevalShiftConfig,
    DataevalShiftDriftOutputs,
    DataevalShiftOODAEOutput,
    DataevalShiftOODOutputs,
    DataevalShiftOutputs,
    DataevalShiftRun,
    DataevalShiftUnivariateOutput,
    DatasetShiftTestStageBase,
    DriftMMDOutput,
    collect_drift,
    collect_ood,
)


@pytest.fixture(scope="module")
def dummy_shift_test_stage():
    class DummyShiftTestStage(DatasetShiftTestStageBase):
        pass

    return DummyShiftTestStage


class TestDatasetShift:
    """
    Tests shared `TestStage` functionality between the Drift and OOD methods

    Methods
    -------
    test_run_and_report
        Tests end-to-end integration
    test_create_cache
        Tests cache file generation
    test_cache_id
        Tests cache id generation
    test_deck_name
        Tests deck name set
    test_empty_deck_name
        Tests error thrown if _deck not overwritten
    """

    def test_run_and_report(self, dummy_shift_test_stage, dummy_dataset_od, artifact_dir) -> None:
        """Tests end-to-end integration of dummy data when loading, running, and collecting"""

        dev_dataset = dummy_dataset_od
        op_dataset = dummy_dataset_od
        op_dataset.images *= 0.5

        test_stage: DatasetShiftTestStageBase = dummy_shift_test_stage()
        output = test_stage.run(use_stage_cache=False, datasets=[dev_dataset, op_dataset])

        report = output.collect_report_consumables(threshold=0.0)

        assert report
        assert len(report) == 2  # Drift and OOD slide

        # Confirm slides have all three required arguments for gradient consumables
        for required_key in ("deck", "layout_name", "layout_arguments"):
            for slide in report:
                assert required_key in slide

        filename = create_deck(report, artifact_dir, "shift")
        assert filename.exists()

    def test_use_stage_cache(self, dummy_dataset_ic, dummy_dataset_od):
        """Tests that cached data can be created and read without modifications"""

        # Set cache required attributes for original test stage
        test_stage = DatasetShiftTestStageBase()
        test_stage._task = "dummy_task"

        # # Modify running results for simpler value checking
        test_stage._run = MagicMock()
        test_stage._run.return_value = DataevalShiftOutputs(
            drift=DataevalShiftDriftOutputs(
                mmd=DriftMMDOutput(drifted=False, distance=0.0, p_val=0.0, threshold=0.0, distance_threshold=0.0),
                cvm=DataevalShiftUnivariateOutput(
                    drifted=False,
                    distance=0.0,
                    p_val=0.0,
                    threshold=0.0,
                    feature_drift=np.array([False]),
                    feature_threshold=0.0,
                    p_vals=np.array([0.0]),
                    distances=np.array([0.0]),
                ),
                ks=DataevalShiftUnivariateOutput(
                    drifted=False,
                    distance=0.0,
                    p_val=0.0,
                    threshold=0.0,
                    feature_drift=np.array([False]),
                    feature_threshold=0.0,
                    p_vals=np.array([0.0]),
                    distances=np.array([0.0]),
                ),
            ),
            ood=DataevalShiftOODOutputs(
                ood_ae=DataevalShiftOODAEOutput(
                    is_ood=np.array([False, False, False]),
                    instance_score=np.array([0.0, 0.0, 0.0]),
                    feature_score=np.array([0.0, 0.0, 0.0]),
                ),
            ),
        )

        # Save run results into cache
        run = test_stage.run(use_stage_cache=True, datasets=[dummy_dataset_ic, dummy_dataset_od])
        base_outputs = run.outputs

        # Create new test stage that will only use cached results
        test_stage_cached = DatasetShiftTestStageBase()
        # Set all cache information to be the same as original test stage
        test_stage_cached._task = "dummy_task"
        # Mock out to ensure the cache overrides the use of _run
        test_stage_cached._run = MagicMock()
        cached_run = test_stage_cached.run(datasets=[dummy_dataset_ic, dummy_dataset_od])
        cached_outputs = cached_run.outputs

        torch.testing.assert_close(base_outputs.model_dump(), cached_outputs.model_dump())

        # Confirm internal _run is skipped if cache is loaded correctly
        test_stage_cached._run.assert_not_called()


class TestDrift:
    """
    Tests the drift methods correctly handle result generation and gradient consumables

    Methods
    -------
    test_run_drift
        Tests run method produces correct output format
    test_collect_drift
        Tests consumable has correct output format
    """

    def test_run_drift(self, dummy_shift_test_stage, dummy_dataset_od) -> None:
        """Tests that the `_run_drift` function produces necessary results for all 3 methods"""
        test_stage: DatasetShiftTestStageBase = dummy_shift_test_stage()
        test_stage.dim = 32
        results = test_stage._run_drift(dataset_1=dummy_dataset_od, dataset_2=dummy_dataset_od)

        results = results.model_dump()

        # Check run drift generates necessary keys for consumable
        for k, v in results.items():
            assert isinstance(v, dict)
            if k == "mmd":
                for output_key in ["drifted", "threshold", "p_val", "distance", "distance_threshold"]:
                    assert output_key in v
            else:
                for output_key in [
                    "drifted",
                    "threshold",
                    "p_val",
                    "distance",
                    "feature_drift",
                    "feature_threshold",
                    "p_vals",
                    "distances",
                ]:
                    assert output_key in v

    def test_collect_drift(self, dummy_shift_test_stage) -> None:
        """
        Tests that the `_collect_drift` function parses the output of `_run_drift`
        and creates Gradient consumable kwargs with computed results
        """

        results = collect_drift(
            deck="fake-deck",
            drift_outputs=DataevalShiftDriftOutputs(
                mmd=DriftMMDOutput(drifted=False, distance=-1, p_val=-1.0, threshold=0.0, distance_threshold=0.0),
                cvm=DataevalShiftUnivariateOutput(
                    drifted=True,
                    distance=0,
                    p_val=0.0,
                    threshold=0.0,
                    feature_drift=np.array([True]),
                    feature_threshold=0.0,
                    p_vals=np.array([0.0]),
                    distances=np.array([0.0]),
                ),
                ks=DataevalShiftUnivariateOutput(
                    drifted=False,
                    distance=1,
                    p_val=1.0,
                    threshold=0.0,
                    feature_drift=np.array([False]),
                    feature_threshold=0.0,
                    p_vals=np.array([1.0]),
                    distances=np.array([1.0]),
                ),
            ),
            dataset_ids=["DummyDataset1", "DummyDataset2"],
        )
        results = results["layout_arguments"]

        assert "DummyDataset1" in results["title"]
        assert "DummyDataset2" in results["title"]

        assert results["line_section_heading"] == "Metric: Drift"

        # Confirms variable string set to drifted versions
        # Access str through List -> Text -> List -> Subtext -> content
        text_content = results["line_section_body"]
        assert "has drifted" in text_content[1].content[0].content
        assert "Retrain model (augmentation, transfer learning)" in text_content[-1].content[0].content

        result_df = results["item_section_body"]

        assert all(result_df["Has drifted?"] == ["No", "Yes", "No"])
        assert all(result_df["Test statistic"] == [-1, 0, 1])
        assert all(result_df["P-value"] == [-1.0, 0.0, 1.0])


class TestOOD:
    """
    Tests the OOD methods correctly handle result generation and gradient consumables

    Methods
    -------
    test_run_ood
        Tests run method produces correct output format
    test_collect_ood
        Tests consumable has correct output format
    """

    def test_run_ood(self, dummy_shift_test_stage, dummy_dataset_od) -> None:
        """Tests that the `_run_ood` function produces necessary results for both methods"""

        dataset_2 = copy.deepcopy(dummy_dataset_od)
        dataset_2.images = torch.zeros_like(dummy_dataset_od.images)

        test_stage: DatasetShiftTestStageBase = dummy_shift_test_stage()
        test_stage.dim = 32
        results = test_stage._run_ood(dataset_1=dummy_dataset_od, dataset_2=dataset_2)

        results = results.model_dump()
        assert list(results) == ["ood_ae"]

        # Check run ood generates necessary keys for consumable
        for v in results.values():
            assert isinstance(v, dict)  # Converted from OODOutput
            for output_key in ["is_ood", "instance_score", "feature_score"]:  # feature_score is unneeded
                assert output_key in v

    def test_collect_ood(self, dummy_shift_test_stage) -> None:
        """
        Tests that the `_collect_ood` function parses the output of `_run_ood`
        and creates Gradient consumable kwargs with computed results
        """

        # Outer gradient kwargs checked by BaseShiftTestStage
        results = collect_ood(
            deck="fake-deck",
            ood_outputs=DataevalShiftOODOutputs(
                ood_ae=DataevalShiftOODAEOutput(
                    is_ood=np.array([True, True, False]),
                    instance_score=np.array([1.0, 0.75, 0.0]),
                    feature_score=np.array([[1.0, 1.0, 1.0], [1.0, 0.75, 1.0], [0.0, 0.0, 0.0]]),
                )
            ),
            dataset_ids=["DummyDataset1", "DummyDataset2"],
        )
        results = results["layout_arguments"]

        assert "DummyDataset1" in results["title"]
        assert "DummyDataset2" in results["title"]

        assert results["line_section_heading"] == "Metric: Out-of-distribution (OOD)"

        # Confirms variable string set to drifted versions
        # Access str through List -> Text -> List -> Subtext -> content
        text_content = results["line_section_body"]
        assert "66.7" in text_content[1].content[0].content
        assert "Retrain model (augmentation, transfer learning)" in text_content[-2].content[0].content

        result_df = results["item_section_body"]

        assert all(result_df["OOD Count"] == [2])
        assert all(result_df["OOD Percent"] == [66.7])
        assert all(result_df["Threshold"] == [0.75])


def test_shift_gradient_pptx(dummy_shift_test_stage, tmp_path, artifact_dir) -> None:
    """This is used to test the output of the shift gradient slides"""

    dummy_drift = {
        k: {
            "drifted": False,
            "threshold": 0.05,
            "p_val": 1.0,
            "distance": 0.0,
            "feature_drift": np.array([False]),
            "feature_threshold": 0.05,
            "p_vals": np.array([1.0]),
            "distances": np.array([0.0]),
        }
        for k in ("cvm", "ks")
    }
    dummy_drift["mmd"] = {
        "drifted": True,
        "threshold": 0.05,
        "p_val": 1.0,
        "distance": 0.0,
        "distance_threshold": 0.05,
    }

    dummy_ood = {
        "ood_ae": {
            "is_ood": np.array([True, False, False]),
            "instance_score": np.array([0.0, 0.5, 1.0]),
            "feature_score": np.array([0.33, 0.67, 1.0]),
        }
    }

    run = DataevalShiftRun(
        dataset_metadata=[{"id": "VOC1"}, {"id": "VOC2"}],
        model_metadata=[],
        metric_metadata=[],
        test_stage_id="",
        config=DataevalShiftConfig(),
        outputs=DataevalShiftOutputs(
            drift=DataevalShiftDriftOutputs(
                mmd=DriftMMDOutput(**dummy_drift["mmd"]),
                cvm=DataevalShiftUnivariateOutput(**dummy_drift["cvm"]),
                ks=DataevalShiftUnivariateOutput(**dummy_drift["ks"]),
            ),
            ood=DataevalShiftOODOutputs(ood_ae=DataevalShiftOODAEOutput(**dummy_ood["ood_ae"])),
        ),
    )

    slides: list[dict[str, Any]] = run.collect_report_consumables(threshold=0.5)
    filename = create_deck(slides, path=artifact_dir, deck_name="shift")
    assert filename.exists()
