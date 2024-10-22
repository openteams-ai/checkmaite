"""Test drift methods in DatasetShiftTestStage"""

import os
from typing import Any

import numpy as np
import pytest
import torch

from jatic_ri.object_detection.test_stages.impls.dataeval_shift_test_stage import (
    DatasetShiftTestStage,
)


class TestDatasetShift:
    """
    Tests shared `TestStage` functionality between the Drift and OOD methods

    Methods
    -------
    test_run_and_report
        Tests end-to-end integration
    test_create_cache
        Tests cache file generation
    test_no_outputs
        Tests empty report consumable
    test_cache_id
        Tests cache id generation
    """

    def test_run_and_report(self, dummy_dataset_od) -> None:
        """Tests end-to-end integration of dummy data when loading, running, and collecting"""

        dev_dataset = dummy_dataset_od
        op_dataset = dummy_dataset_od
        op_dataset.images *= 0.5

        stage = DatasetShiftTestStage()
        stage.load_datasets(dataset_1=dev_dataset, dataset_2=op_dataset, dataset_1_id="dev", dataset_2_id="op")
        stage.run(use_cache=False)

        assert stage.outputs is not None
        assert len(stage.outputs.keys()) == 2
        assert "drift" in stage.outputs
        assert "ood" in stage.outputs

        report = stage.collect_report_consumables()

        assert report
        assert len(report) == 2  # Drift and OOD slide

        # Confirm slides have all three required arguments for gradient consumables
        for required_key in ("deck", "layout_name", "layout_arguments"):
            for slide in report:
                assert required_key in slide

    def test_create_cache(self, dummy_dataset_od, tmp_path) -> None:
        """Test that the cache file is written to after the run method is called"""

        stage = DatasetShiftTestStage()
        stage.cache_base_path = tmp_path
        stage.load_datasets(
            dataset_1=dummy_dataset_od,
            dataset_2=dummy_dataset_od,
            dataset_1_id="dev",
            dataset_2_id="op",
        )
        stage.run()

        assert os.path.exists(stage.cache_path)

    def test_cache_id(self) -> None:
        """Tests the unique cache id is set based on the ids of dataset 1 and dataset 2"""

        stage = DatasetShiftTestStage()
        stage.load_datasets(None, "Dummy1", None, "Dummy2")  # type: ignore
        assert stage.cache_id == "shift_Dummy1_Dummy2.json"


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

    def test_run_drift(self, dummy_dataset_od) -> None:
        """Tests that the `_run_drift` function produces necessary results for all 3 methods"""
        zeros = list(torch.zeros_like(dummy_dataset_od.images))
        ones = list(torch.ones_like(dummy_dataset_od.images))
        test_stage = DatasetShiftTestStage()
        results = test_stage._run_drift(  # noqa: SLF001 > ignore private method call
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

    def test_collect_drift(self) -> None:
        """
        Tests that the `_collect_drift` function parses the output of `_run_drift`
        and creates Gradient consumable kwargs with computed results
        """

        test_stage = DatasetShiftTestStage()
        test_stage.load_datasets(None, "Dummy1", None, "Dummy2")  # type: ignore

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

        results = test_stage._collect_drift(outputs=dummy_outputs)  # noqa: SLF001 > ignore private method access

        assert isinstance(results, list)
        assert len(results) == 1

        results = results[0]["layout_arguments"]
        # Outer gradient kwargs checked by DatasetShiftTestStage

        assert "Dummy1" in results["title"]
        assert "Dummy2" in results["title"]

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

    def test_run_ood(self, dummy_dataset_od) -> None:
        """Tests that the `_run_ood` function produces necessary results for both methods"""

        zeros = list(torch.zeros_like(dummy_dataset_od.images))
        ones = list(torch.ones_like(dummy_dataset_od.images))

        test_stage = DatasetShiftTestStage()
        results = test_stage._run_ood(  # noqa: SLF001 > ignore private method call
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

    def test_collect_ood(self) -> None:
        """
        Tests that the `_collect_ood` function parses the output of `_run_ood`
        and creates Gradient consumable kwargs with computed results
        """

        test_stage = DatasetShiftTestStage()
        test_stage.load_datasets(None, "Dummy1", None, "Dummy2")  # type: ignore

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

        results = test_stage._collect_ood(outputs=dummy_outputs)  # noqa: SLF001 > ignore private method access

        assert isinstance(results, list)
        assert len(results) == 1

        results = results[0]["layout_arguments"]
        # Outer gradient kwargs checked by DatasetShiftTestStage

        assert "Dummy1" in results["title"]
        assert "Dummy2" in results["title"]

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


@pytest.mark.skip(reason="For internal gradient pptx testing. Not for pipeline")
def test_gradient_pptx() -> None:
    from gradient.templates_and_layouts.create_deck import create_deck

    teststage: DatasetShiftTestStage = DatasetShiftTestStage()
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
    create_deck(slides, deck_name="DatasetDriftDeck")
