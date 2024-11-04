"""Test Dataset Bias Analysis"""

import json
import pytest
import pandas as pd

from pathlib import Path
from typing import Any

import numpy as np
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.object_detection.test_stages.impls.dataeval_bias_test_stage import (
    DatasetBiasTestStage,
    create_text_data_slide
)

@pytest.fixture(scope="module")
def balance_outputs() -> dict[str, Any]:
    return {
        "balance": np.array([0.99999822, 0.13363788, 0.04505382, 0.02994455]),
        "factors": np.array([[0.99999843, 0.04133555, 0.09725766],[0.04133555, 0.08433558, 0.1301489 ],[0.09725766, 0.1301489 , 0.99999856]]),
        "classwise": np.array([[0.99999822, 0.13363788, 0.0, 0.0], [0.99999822, 0.13363788, 0.0, 0.0]]),
    }

@pytest.fixture(scope="module")
def coverage_outputs() -> dict[str, Any]:
    return {
        "indices": np.array([447, 412,   8,  32,  63]),
        "critical_value": 0.8459038956941765,
        "radii": np.arange(10),
    }

@pytest.fixture(scope="module")
def diversity_outputs() -> dict[str, Any]:
    return {
        "diversity_index": np.array([0.18103448, 0.18103448, 0.88636364]),
        "classwise": np.array([[0.17241379, 0.39473684],[0.2, 0.2]])
    }

@pytest.fixture(scope="module")
def parity_outputs() -> dict[str, Any]:
    return {
        "score": np.array([7.35731943, 5.46711299, 0.51506212]),
        "p_value": np.array([0.28906231, 0.24263543, 0.77295762]),
    }

class TestBiasUtilityFunctions:
    """Test private helper functions used by Bias"""

    @pytest.mark.parametrize("with_table", [True, False])
    def test_create_text_data_slide(self, with_table: bool):
        """Tests TextData arguments are correctly populated with and without a DataFrame"""
        table = pd.DataFrame({"dummy": [0]}) if with_table else None
        result_template = create_text_data_slide(
            title="TITLE",
            heading="HEADING",
            text=["A", "B", "C"],
            table=table,
        )
        assert result_template["deck"] == "object_detection_dataset_evaluation"
        assert result_template["layout_name"] == "TextData"

        layout_args = result_template["layout_arguments"] 
        assert layout_args["title"] == "TITLE"
        assert layout_args["text_column_heading"] == "HEADING"
        assert layout_args["text_column_half"]
        assert isinstance(layout_args["text_column_body"], list)
        assert ("data_column_table" in layout_args) == with_table

        create_deck([result_template], path=Path("artifacts"), deck_name=f"test_create_text_data_slide_table={str(with_table)}")


class TestDatasetBiasRun:
    """Test shared Bias TestStage _run functionality between balance, coverage, diversity, and parity"""

    def test_run_and_report(self, dummy_dataset_od) -> None:
        """Test output formats at each stage of the Bias test stage"""
        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dataset=dummy_dataset_od, dataset_id="DummyDataset")
        test_stage.run(use_cache=False)

        outputs: dict[str, Any] = test_stage.outputs
        methods = ("balance", "coverage", "diversity", "parity")
        assert all(method in outputs for method in methods)  # All method keys found after run

        output = test_stage.collect_report_consumables()
        assert len(output) == 4

    def test_run_non_homogenous_images(self, dummy_dataset_od):
        """Test case where images are different sizes.
        
        Coverage requires ArrayLikes, so skip running coverage
        """
        # Modify images to be non-homogenous (like VOC)
        dummy_dataset_od.images = [np.ones(shape=(1, i, i)) for i in range(10)]

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dataset=dummy_dataset_od, dataset_id="DummyDataset")
        test_stage.run(use_cache=False)

        outputs: dict[str, Any] = test_stage.outputs
        methods = ("balance", "diversity", "parity")
        assert all(method in outputs for method in methods)  # All method keys found after run

        output = test_stage.collect_report_consumables()
        assert len(output) == 3


class TestBiasCache:
    """Tests the Bias Cache attribute correctly writes, saves, and reads cached runs"""

    def test_cache_data(self, dummy_dataset_od, tmpdir) -> None:
        """Test that the cache file is written after the run method without data modifications"""

        test_stage = DatasetBiasTestStage()
        test_stage.cache_base_path = tmpdir
        test_stage.load_dataset(dummy_dataset_od, "DummyDataset")

        # To write cache, use_cache must be True, but don't want to read from previous cache writes
        assert not Path(test_stage.cache_path).exists()
        test_stage.run()
        outputs = test_stage.outputs
        assert Path(test_stage.cache_path).exists()

        # Get current runs written cache
        with open(test_stage.cache_path) as f:
            cached_outputs: dict[str, Any] = json.load(fp=f)

        # For every method, check cached values per metric equal returned outputs
        assert outputs.keys() == cached_outputs.keys()
        for method, results in outputs.items():
            for k, v in results.items():
                np.testing.assert_array_equal(cached_outputs[method][k], v)

    def test_cache_path(self) -> None:
        """Tests the unique cache id is set based on the id of dataset"""

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(None, "DummyDataset")
        assert test_stage.cache_id == "bias_DummyDataset.json"


class TestBiasCollectReportConsumables:
    """Tests the individual report_consumables methods for every Bias metric, as well as the combined collect_report_consumables"""

    def test_report_balance(self, dummy_dataset_od, balance_outputs):
        """Test balance specific rollup values and action"""

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dummy_dataset_od, "DummyDataset")

        slide = test_stage._report_balance(outputs=balance_outputs)
        layout_args = slide["layout_arguments"]

        # Check rollup calculation and associated action
        assert f"0 factors" in layout_args["text_column_body"][1].content[0].content
        assert layout_args["text_column_body"][-1].content[0].content == "* No action required"

        create_deck([slide], path=Path("artifacts"), deck_name="test_report_balance")

    def test_report_coverage(self, dummy_dataset_od, coverage_outputs: dict[str, Any], tmpdir):
        """Test the coverage specific gradient output"""
        
        test_stage = DatasetBiasTestStage()
        test_stage.cache_base_path = tmpdir
        test_stage.load_dataset(dummy_dataset_od, "DummyDataset")

        slide = test_stage._report_coverage(coverage_outputs)
        layout_args = slide["layout_arguments"]

        # Check text and visual slide arguments
        assert layout_args["title"] == "Dataset: DummyDataset | Category Bias"
        assert layout_args["text_column_heading"] == "Metric: Coverage"
        assert layout_args["text_column_half"]
        assert layout_args["image_caption"] == "Examples"

        # Test calculated dataframe values
        cov_df: pd.DataFrame = layout_args["data_column_table"]

        assert tuple(cov_df.columns) == ("Poor Coverage", "Threshold")
        assert cov_df.shape == (1, 2)  # One row, 2 columns
        assert cov_df["Poor Coverage"][0] == "5 of 10 (50.0%)" 
        assert cov_df["Threshold"][0] == round(coverage_outputs["critical_value"], 3)

        # Test image and path IO
        folder = test_stage.image_folder
        assert folder.is_dir()

        gradient_column_args = ("data_column_2_image_1", "data_column_2_image_2", "data_column_2_image_3")

        for i, col in enumerate(gradient_column_args):  # Set to top_k images
            img_path = folder / Path(f"coverage_example_{i}.png")
            assert img_path.exists()  # Check if image was saved
            assert layout_args[col] == img_path  # Confirm gradient argument uses the same path

        assert not (folder / Path(f"coverage_example_3.png")).exists()  # Only top 3 images should be saved

        create_deck([slide], path=Path("artifacts"), deck_name="test_report_coverage")

    def test_report_coverage_low_samples(self, dummy_dataset_od, coverage_outputs: dict[str, Any], tmpdir):
        """
        In the rare cases there are less uncovered indices than gradient images,
        test that the gradient args for the remaining image path spots do not get filled with a path
        
        Since the general case tests the shared args, those checks are left out of this special case
        """
        
        test_stage = DatasetBiasTestStage()
        test_stage.cache_base_path = tmpdir
        test_stage.load_dataset(dummy_dataset_od, "DummyDataset")

        coverage_outputs["indices"] = np.array([0], dtype=np.intp)

        slide = test_stage._report_coverage(coverage_outputs)
        layout_args = slide["layout_arguments"]

        # Test calculated dataframe values
        cov_df: pd.DataFrame = layout_args["data_column_table"]
        assert cov_df["Poor Coverage"][0] == "1 of 10 (10.0%)" 

        # Test image and path IO
        folder = test_stage.image_folder
        assert folder.is_dir()

        gradient_column_unused_args = ("data_column_2_image_2", "data_column_2_image_3")

        img_path = folder / Path(f"coverage_example_0.png")
        assert img_path.exists()
        assert layout_args["data_column_2_image_1"] == img_path

        for i, col in enumerate(gradient_column_unused_args):  # Set to top_k images
            img_path = folder / Path(f"coverage_example_{i+1}.png")
            assert not img_path.exists()  # Check if image was saved
            assert col not in layout_args
        
        create_deck([slide], path=Path("artifacts"), deck_name="test_report_coverage_low_samples")

    def test_report_diversity(self, dummy_dataset_od, diversity_outputs):
        """Test diversity specific rollup values and action"""

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dummy_dataset_od, "DummyDataset")

        slide = test_stage._report_diversity(diversity_outputs)
        layout_args = slide["layout_arguments"]

        # Check rollup calculation and associated action
        assert "2 factors" in layout_args["text_column_body"][1].content[0].content
        assert layout_args["text_column_body"][-1].content[0].content == "* Ensure balanced representation of all classes for all metadata" 

        create_deck([slide], path=Path("artifacts"), deck_name="test_report_diversity")

    def test_report_parity(self, dummy_dataset_od, parity_outputs):
        """Test parity specific rollup values and action"""

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dummy_dataset_od, "DummyDataset")

        slide = test_stage._report_parity(parity_outputs)
        layout_args = slide["layout_arguments"]

        # Check rollup calculation and associated action
        assert "0 factors" in layout_args["text_column_body"][1].content[0].content
        assert layout_args["text_column_body"][-1].content[0].content == "* No action required"

        create_deck([slide], path=Path("artifacts"), deck_name="test_report_parity")

    def test_bias_gradient_pptx(
            self,
            dummy_dataset_od,
            coverage_outputs,
            balance_outputs,
            diversity_outputs,
            parity_outputs
        ) -> None:
        """Test all gradient slide kwargs collected together"""

        test_stage: DatasetBiasTestStage = DatasetBiasTestStage()
        test_stage.load_dataset(dummy_dataset_od, "DummyDataset")

        test_stage.outputs = {
            test_stage.BALANCE_KEY: balance_outputs,
            test_stage.COVERAGE_KEY: coverage_outputs,
            test_stage.DIVERSITY_KEY: diversity_outputs,
            test_stage.PARITY_KEY: parity_outputs,
        }

        slides: list[dict[str, Any]] = test_stage.collect_report_consumables()
        create_deck(slides, path=Path("artifacts"), deck_name="test_bias_gradient_pptx")
