"""Test Dataset Bias Analysis"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.image_classification.test_stages.impls.dataeval_bias_test_stage import DatasetBiasTestStage
from jatic_ri.util.utils import save_figure_to_tempfile


@pytest.fixture(scope="module")
def fake_image() -> str:
    image = np.ones((28, 28, 3), dtype=int) * 200
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    return save_figure_to_tempfile(fig)


@pytest.fixture(scope="module")
def balance_outputs(fake_image) -> dict[str, Any]:
    return {
        "balance": np.array([0.99999822, 0.13363788, 0.04505382, 0.02994455]),
        "factors": np.array(
            [
                [0.99999843, 0.04133555, 0.09725766],
                [0.04133555, 0.08433558, 0.1301489],
                [0.09725766, 0.1301489, 0.99999856],
            ]
        ),
        "classwise": np.array([[0.99999822, 0.13363788, 0.0, 0.0], [0.99999822, 0.13363788, 0.0, 0.0]]),
        "image": fake_image,
    }


@pytest.fixture(scope="module")
def coverage_outputs(fake_image) -> dict[str, Any]:
    return {
        "indices": np.array([447, 412, 8, 32, 63]),
        "critical_value": 0.8459038956941765,
        "radii": np.arange(10),
        "image": fake_image,
    }


@pytest.fixture(scope="module")
def diversity_outputs(fake_image) -> dict[str, Any]:
    return {
        "diversity_index": np.array([0.18103448, 0.18103448, 0.88636364]),
        "classwise": np.array([[0.17241379, 0.39473684], [0.2, 0.2]]),
        "image": fake_image,
    }


@pytest.fixture(scope="module")
def parity_outputs() -> dict[str, Any]:
    return {
        "metadata_names": ["A", "B", "C"],
        "score": np.array([7.35731943, 5.46711299, 0.51506212]),
        "p_value": np.array([0.28906231, 0.24263543, 0.77295762]),
    }


def ignore_bias_warnings(test_fn):
    for filter in [
        "ignore:All samples look discrete with so few data points:UserWarning",
        r"ignore:The following factors did not meet the recommended \d+ occurrences for each value-label combination:UserWarning",
    ]:
        test_fn = pytest.mark.filterwarnings(filter)(test_fn)

    return test_fn


class TestICDatasetBiasRun:
    """Test shared Bias TestStage _run functionality between balance, coverage, diversity, and parity"""

    @ignore_bias_warnings
    @pytest.mark.parametrize("homogeneous_size", [True, False])
    def test_run_and_report(self, dummy_dataset_ic, homogeneous_size) -> None:
        """Test output formats at each stage of the Bias test stage"""
        if not homogeneous_size:
            # Modify images to be non-homogenous (like VOC)
            dummy_dataset_ic.data = [np.ones(shape=(3, i, i)) for i in range(1, 10)]

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dataset=dummy_dataset_ic, dataset_id="DummyDataset")
        test_stage.run(use_stage_cache=False)

        outputs: dict[str, Any] = test_stage.outputs
        assert outputs.keys() == {"balance", "coverage", "diversity", "parity"}
        if homogeneous_size:
            assert "image" in outputs["coverage"]

        output = test_stage.collect_report_consumables()
        assert len(output) == 4


class TestICBiasCache:
    """Tests the Bias Cache attribute correctly writes, saves, and reads cached runs"""

    @ignore_bias_warnings
    def test_cache_data(self, dummy_dataset_ic) -> None:
        """Test that the cache file is written after the run method without data modifications"""

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dummy_dataset_ic, "DummyDataset")

        # To write cache, use_stage_cache must be True, but don't want to read from previous cache writes
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
        assert test_stage.cache_id == "bias_ic_DummyDataset.json"


class TestICBiasCollectReportConsumables:
    """Tests the individual report_consumables methods for every Bias metric, as well as the combined collect_report_consumables"""

    def test_report_balance(self, dummy_dataset_ic, balance_outputs, artifact_dir):
        """Test balance specific rollup values and action"""

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dummy_dataset_ic, "DummyDataset")

        slide = test_stage._report_balance(outputs=balance_outputs)
        layout_args = slide["layout_arguments"]

        # Check rollup calculation and associated action
        assert "0 factors" in layout_args["text_column_body"][1].content[0].content
        assert layout_args["text_column_body"][-1].content[0].content == "* No action required"

        # Check if image was saved
        img_path = layout_args["data_column_image"]
        assert img_path.exists()

        filename = create_deck([slide], path=artifact_dir, deck_name="test_report_balance")
        assert filename.exists()

    def test_report_coverage(self, dummy_dataset_ic, coverage_outputs: dict[str, Any], artifact_dir):
        """Test the coverage specific gradient output"""

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dummy_dataset_ic, "DummyDataset")

        slide = test_stage._report_coverage(coverage_outputs)
        layout_args = slide["layout_arguments"]

        # Check text and visual slide arguments
        assert layout_args["title"] == "Dataset: DummyDataset | Category Bias"
        assert layout_args["text_column_heading"] == "Metric: Coverage"
        assert layout_args["text_column_half"]
        assert (
            layout_args["text_column_body"][-1].content[0].content
            == "* Increase representation of rare but relevant samples in areas of poor coverage"
        )

        # Test calculated dataframe values
        cov_df: pd.DataFrame = layout_args["data_column_table"]

        assert tuple(cov_df.columns) == ("Poor Coverage", "Threshold")
        assert cov_df.shape == (1, 2)  # One row, 2 columns
        assert cov_df["Poor Coverage"][0] == "5 of 10 (50.0%)"
        assert cov_df["Threshold"][0] == round(coverage_outputs["critical_value"], 2)

        # Test image and path IO
        folder = test_stage.image_folder
        assert folder.is_dir()

        # Check if image was saved
        img_path = layout_args["data_column_image"]
        assert img_path.exists()

        filename = create_deck([slide], path=artifact_dir, deck_name="test_report_coverage")
        assert filename.exists()

    def test_report_diversity(self, dummy_dataset_ic, diversity_outputs, artifact_dir):
        """Test diversity specific rollup values and action"""

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dummy_dataset_ic, "DummyDataset")

        slide = test_stage._report_diversity(diversity_outputs)
        layout_args = slide["layout_arguments"]

        # Check rollup calculation and associated action
        assert "2 factors" in layout_args["text_column_body"][1].content[0].content
        assert (
            layout_args["text_column_body"][-1].content[0].content
            == "* Ensure balanced representation of all classes for all metadata"
        )

        # Check if image was saved
        img_path = layout_args["data_column_image"]
        assert img_path.exists()

        filename = create_deck([slide], path=artifact_dir, deck_name="test_report_diversity")
        assert filename.exists()

    def test_report_parity(self, dummy_dataset_ic, parity_outputs, artifact_dir):
        """Test parity specific rollup values and action"""

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dummy_dataset_ic, "DummyDataset")

        slide = test_stage._report_parity(parity_outputs)
        layout_args = slide["layout_arguments"]

        # Check rollup calculation and associated action
        assert "0 factors" in layout_args["text_column_body"][1].content[0].content
        assert layout_args["text_column_body"][-1].content[0].content == "* No action required"

        filename = create_deck([slide], path=artifact_dir, deck_name="test_report_parity")
        assert filename.exists()

    def test_bias_gradient_pptx(
        self, dummy_dataset_ic, coverage_outputs, balance_outputs, diversity_outputs, parity_outputs, artifact_dir
    ) -> None:
        """Test all gradient slide kwargs collected together"""

        test_stage: DatasetBiasTestStage = DatasetBiasTestStage()
        test_stage.load_dataset(dummy_dataset_ic, "DummyDataset")

        test_stage.outputs = {
            test_stage.BALANCE_KEY: balance_outputs,
            test_stage.COVERAGE_KEY: coverage_outputs,
            test_stage.DIVERSITY_KEY: diversity_outputs,
            test_stage.PARITY_KEY: parity_outputs,
        }

        slides: list[dict[str, Any]] = test_stage.collect_report_consumables()

        filename = create_deck(slides, path=artifact_dir, deck_name="test_bias_gradient_pptx")
        assert filename.exists()
