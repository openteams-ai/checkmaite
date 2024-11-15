"""Test drift methods in BaseShiftTestStage"""

import os
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri._common.test_stages.impls.dataeval_shift_test_stage import DatasetShiftTestStageBase

@pytest.fixture(scope="module")
def dummy_shift_test_stage():
    class DummyShiftTestStage(DatasetShiftTestStageBase):

        @property
        def cache_id(self) -> str:
            """Unique identifier for cached results"""
            return f"dummy_cache_id.json"
    
        _deck: str = "dummy_deck"
    
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

    def test_run_and_report(self, dummy_shift_test_stage, dummy_dataset_od) -> None:
        """Tests end-to-end integration of dummy data when loading, running, and collecting"""

        dev_dataset = dummy_dataset_od
        op_dataset = dummy_dataset_od
        op_dataset.images *= 0.5

        test_stage: DatasetShiftTestStageBase = dummy_shift_test_stage()
        test_stage.load_datasets(dataset_1=dev_dataset, dataset_2=op_dataset, dataset_1_id="dev", dataset_2_id="op")
        test_stage.run(use_cache=False)

        assert test_stage.outputs is not None
        assert len(test_stage.outputs.keys()) == 2
        assert "drift" in test_stage.outputs
        assert "ood" in test_stage.outputs

        report = test_stage.collect_report_consumables()

        assert report
        assert len(report) == 2  # Drift and OOD slide

        # Confirm slides have all three required arguments for gradient consumables
        for required_key in ("deck", "layout_name", "layout_arguments"):
            for slide in report:
                assert required_key in slide

    def test_create_cache(self, dummy_shift_test_stage, dummy_dataset_od) -> None:
        """Test that the cache file is written after the run method is called without data modifications"""

        test_stage: DatasetShiftTestStageBase = dummy_shift_test_stage()
        test_stage.load_datasets(
            dataset_1=dummy_dataset_od,
            dataset_2=dummy_dataset_od,
            dataset_1_id="DummyDataset1",
            dataset_2_id="DummyDataset2",
        )
        test_stage.run()

        assert os.path.exists(test_stage.cache_path)

    def test_use_cache(self, dummy_dataset_ic, dummy_dataset_od, tmp_path):
        """Tests that cached data can be created and read without modifications"""

        # Set cache required attributes for original test stage
        test_stage = DatasetShiftTestStageBase()
        test_stage.cache_base_path = tmp_path
        test_stage._task = "dummy_task"
        test_stage.load_datasets(dummy_dataset_ic, "Dataset1", dummy_dataset_od, "Dataset2")

        # # Modify running results for simpler value checking
        test_stage._run = MagicMock()
        test_stage._run.return_value = {
            "drift": "dummy_drift",
            "ood": "dummy_ood",
        }

        # Save run results into cache
        test_stage.run(use_cache=True)
        base_outputs = test_stage.outputs

        # Create new test stage that will only use cached results
        test_stage_cached = DatasetShiftTestStageBase()
        # Set all cache information to be the same as original test stage
        test_stage_cached.cache_base_path = tmp_path
        test_stage_cached._task = "dummy_task"
        test_stage_cached.load_datasets(dummy_dataset_ic, "Dataset1", dummy_dataset_od, "Dataset2")
        # Mock out to ensure the cache overrides the use of _run
        test_stage_cached._run = MagicMock()
        test_stage_cached.run()
        cached_outputs = test_stage_cached.outputs

        # Confirm the original and loaded results are the same
        assert base_outputs == cached_outputs
        # Confirm internal _run is skipped if cache is loaded correctly
        test_stage_cached._run.assert_not_called()

    def test_task_cache_id(self) -> None:
        test_stage = DatasetShiftTestStageBase()
        test_stage._task = "dummy_task"
        test_stage.load_datasets(None, "DummyDataset1", None, "DummyDataset2")  # type: ignore

        assert test_stage.cache_id == "shift_dummy_task_DummyDataset1_DummyDataset2.json"

    def test_no_task_cache_id(self) -> None:
        """Tests the unique cache id fails if task is not set"""

        test_stage = DatasetShiftTestStageBase()
        test_stage.load_datasets(None, "DummyDataset1", None, "DummyDataset2")  # type: ignore
        
        with pytest.raises(AttributeError):
            test_stage.cache_id

    def test_deck_name(self, dummy_shift_test_stage):
        """Tests that the _deck property of the BaseShiftTestStage is correctly overwritten"""
        
        test_stage = dummy_shift_test_stage()
        assert test_stage._deck == "dummy_deck"

    def test_empty_deck_name(self):
        """Tests that not setting _deck in a subclass raises AttributeError when called"""

        class NoDeckShiftTestStage(DatasetShiftTestStageBase):
            deck: str = "WrongProperty"

        with pytest.raises(AttributeError):
            NoDeckShiftTestStage()._deck


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
        zeros = torch.zeros_like(dummy_dataset_od.images)
        ones = torch.ones_like(dummy_dataset_od.images)
        test_stage: DatasetShiftTestStageBase = dummy_shift_test_stage()
        results = test_stage._run_drift(
            images_1=zeros,
            images_2=ones,
        )

        assert "drift" in results
        assert "ood" not in results

        results = results["drift"]
        assert list(results) == ["Maximum Mean Discrepency", "Cramér-von Mises", "Kolmogorov-Smirnov"]

        # Check run drift generates necessary keys for consumable
        for v in results.values():
            assert isinstance(v, dict)  # Converted from DriftOutput
            for output_key in ["is_drift", "p_val", "distance"]:
                assert output_key in v

    def test_collect_drift(self, dummy_shift_test_stage) -> None:
        """
        Tests that the `_collect_drift` function parses the output of `_run_drift`
        and creates Gradient consumable kwargs with computed results
        """

        test_stage: DatasetShiftTestStageBase = dummy_shift_test_stage()
        test_stage.load_datasets(None, "DummyDataset1", None, "DummyDataset2")  # type: ignore

        # One test set to drifted regardless of values
        dummy_outputs = {
            "Maximum Mean Discrepency": {
                "is_drift": False,
                "distance": -1,
                "p_val": -1.0,
            },
            "Cramér-von Mises": {
                "is_drift": True,
                "distance": 0,
                "p_val": 0.0,
            },
            "Kolmogorov-Smirnov": {
                "is_drift": False,
                "distance": 1,
                "p_val": 1.0,
            },
        }

        # Outer gradient kwargs checked by BaseShiftTestStage
        results = test_stage._collect_drift(outputs=dummy_outputs)
        results = results["layout_arguments"]

        assert "DummyDataset1" in results["title"]
        assert "DummyDataset2" in results["title"]

        assert results["text_column_heading"] == "Metric: Drift"

        # Confirms variable string set to drifted versions
        # Access str through List -> Text -> List -> Subtext -> content
        text_content = results["text_column_body"]
        assert "has drifted" in text_content[1].content[0].content
        assert "Retrain model (augmentation, transfer learning)" in text_content[-1].content[0].content

        result_df = results["data_column_table"]

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

        zeros = np.array(torch.zeros_like(dummy_dataset_od.images))
        ones = np.array(torch.ones_like(dummy_dataset_od.images))

        test_stage: DatasetShiftTestStageBase = dummy_shift_test_stage()
        results = test_stage._run_ood(
            images_1=zeros,
            images_2=ones,
        )

        assert "ood" in results
        assert "drift" not in results

        results = results["ood"]
        assert list(results) == ["OOD_AE", "OOD_VAEGMM"]

        # Check run ood generates necessary keys for consumable
        for v in results.values():
            assert isinstance(v, dict)  # Converted from OODOutput
            for output_key in ["is_ood", "instance_score"]:  # feature_score is unneeded
                assert output_key in v

    def test_collect_ood(self, dummy_shift_test_stage) -> None:
        """
        Tests that the `_collect_ood` function parses the output of `_run_ood`
        and creates Gradient consumable kwargs with computed results
        """

        test_stage: DatasetShiftTestStageBase = dummy_shift_test_stage()
        test_stage.load_datasets(None, "DummyDataset1", None, "DummyDataset2")  # type: ignore

        # One test with outliers, one without; regardless of values
        dummy_outputs = {
            "OOD_AE": {
                "is_ood": np.array([True, True, False]),
                "instance_score": np.array([1.0, 0.75, 0.0]),
                "feature_score": np.array(
                    [
                        [1.0, 1.0, 1.0],
                        [1.0, 0.75, 1.0],
                        [0.0, 0.0, 0.0],
                    ],
                ),
            },
            "OOD_VAEGMM": {
                "is_ood": np.array([False, False, False]),
                "instance_score": np.array([0.0, 0.25, 0.0]),
                "feature_score": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.25, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                ),
            },
        }

        # Outer gradient kwargs checked by BaseShiftTestStage
        results = test_stage._collect_ood(outputs=dummy_outputs)  # noqa: SLF001 > ignore private method access
        results = results["layout_arguments"]

        assert "DummyDataset1" in results["title"]
        assert "DummyDataset2" in results["title"]

        assert results["text_column_heading"] == "Metric: Out-of-distribution (OOD)"

        # Confirms variable string set to drifted versions
        # Access str through List -> Text -> List -> Subtext -> content
        text_content = results["text_column_body"]
        assert "66.7" in text_content[1].content[0].content
        assert "Retrain model (augmentation, transfer learning)" in text_content[-2].content[0].content

        result_df = results["data_column_table"]

        assert all(result_df["OOD Count"] == [2, 0])
        assert all(result_df["OOD Percent"] == [66.7, 0.0])
        assert all(result_df["Threshold"] == [0.75, 0.25])


def test_shift_gradient_pptx(dummy_shift_test_stage, tmp_path) -> None:
    """This is used to test the output of the shift gradient slides"""
    teststage: DatasetShiftTestStageBase = dummy_shift_test_stage()
    teststage.load_datasets(None, "VOC1", None, "VOC2")  # type: ignore -> Only needs a dataset_id

    dummy_drift: dict[str, dict[str, Any]] = {
        k: {
            "is_drift": False,
            "threshold": 0.05,
            "p_val": 1.0,
            "distance": 0.0,
            "feature_drift": np.array([False]),
            "feature_threshold": 0.05,
            "p_vals": np.array([1.0]),
            "distances": np.array([0.0]),
        }
        for k in ("Maximum Mean Discrepency", "Cramér-von Mises", "Kolmogorov-Smirnov")
    }

    dummy_ood: dict[str, dict[str, Any]] = {
        k: {
            "is_ood": np.array([True, False, False]),
            "instance_score": np.array([0.0, 0.5, 1.0]),
            "feature_score": np.array([0.33, 0.67, 1.0]),
        }
        for k in ("OOD_AE", "OOD_VAE")
    }

    dummy_output: dict[str, dict[str, dict[str, Any]]] = {
        "drift": dummy_drift,
        "ood": dummy_ood,
    }

    teststage.outputs = dummy_output

    slides: list[dict[str, Any]] = teststage.collect_report_consumables()
    create_deck(slides, path=tmp_path / "DatasetShiftDeck.pptx", deck_name="DatasetShiftDeck")
