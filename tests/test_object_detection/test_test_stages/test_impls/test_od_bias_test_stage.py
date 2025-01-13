"""Test Dataset Bias Analysis"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pytest
import pandas as pd

from pathlib import Path
from typing import Any

from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.object_detection.test_stages.impls.dataeval_bias_test_stage import DatasetBiasTestStage
from jatic_ri.util.utils import save_figure_to_tempfile


@pytest.fixture(scope="module")
def fake_image() -> str:
    image = np.ones((28,28,3), dtype=int)*200
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


class TestODDatasetBiasRun:
    """Test shared Bias TestStage _run functionality between balance, coverage, diversity, and parity"""

    def test_run_and_report_no_target_metadata(self, dummy_dataset_od) -> None:
        """Test output formats at each stage of the Bias test stage"""
        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dataset=dummy_dataset_od, dataset_id="DummyDataset")
        test_stage.run(use_cache=False)

        outputs: dict[str, Any] = test_stage.outputs
        methods = ("balance", "coverage", "diversity", "parity")
        assert all(method in outputs for method in methods)  # All method keys found after run

        output = test_stage.collect_report_consumables()
        assert len(output) == 4

    def test_run_and_report_with_target_metadata(self, dummy_dataset_od_with_target_metadata) -> None:
        """Test output formats at each stage of the Bias test stage"""
        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dataset=dummy_dataset_od_with_target_metadata, dataset_id="DummyDataset")
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

    def test_coco(self):
        from jatic_ri.object_detection.datasets import CocoDetectionDataset
        from jatic_ri import PACKAGE_DIR
        from os import path
        import tests

        coco_dataset_dir = PACKAGE_DIR.parent.parent.joinpath(
            path.dirname(tests.__file__),
            ('testing_utilities/example_data/coco_resized_val2017'),
        )
        coco_dataset = CocoDetectionDataset(
            root=str(coco_dataset_dir),
            ann_file=str(coco_dataset_dir.joinpath('instances_val2017_resized_6.json')),
        )

        stage = DatasetBiasTestStage()

        stage.load_dataset(dataset=coco_dataset, dataset_id='asd')
                            
        stage.run(use_cache=False)
        pass  # no explosions


class TestODBiasCache:
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
        assert test_stage.cache_id == "bias_od_DummyDataset.json"


class TestODBiasCollectReportConsumables:
    """Tests the individual report_consumables methods for every Bias metric, as well as the combined collect_report_consumables"""

    def test_report_balance(self, dummy_dataset_od, balance_outputs, artifact_dir):
        """Test balance specific rollup values and action"""

        test_stage = DatasetBiasTestStage()
        test_stage.cache_base_path = artifact_dir
        test_stage.load_dataset(dummy_dataset_od, "DummyDataset")

        slide = test_stage._report_balance(outputs=balance_outputs)
        layout_args = slide["layout_arguments"]

        # Check rollup calculation and associated action
        assert f"0 factors" in layout_args["text_column_body"][1].content[0].content
        assert layout_args["text_column_body"][-1].content[0].content == "* No action required"

        # Check if image was saved
        img_path = layout_args['data_column_image']
        assert img_path.exists()

        filename = create_deck([slide], path=Path(artifact_dir), deck_name="test_report_balance")
        assert filename.exists()

    def test_report_coverage(self, dummy_dataset_od, coverage_outputs: dict[str, Any], artifact_dir):
        """Test the coverage specific gradient output"""

        test_stage = DatasetBiasTestStage()
        test_stage.cache_base_path = artifact_dir
        test_stage.load_dataset(dummy_dataset_od, "DummyDataset")

        slide = test_stage._report_coverage(coverage_outputs)
        layout_args = slide["layout_arguments"]

        # Check text and visual slide arguments
        assert layout_args["title"] == "Dataset: DummyDataset | Category Bias"
        assert layout_args["text_column_heading"] == "Metric: Coverage"
        assert layout_args["text_column_half"]
        assert (
            layout_args['text_column_body'][-1].content[0].content ==
            "* Increase respresentation of rare but relevant samples in areas of poor coverage"
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
        img_path = layout_args['data_column_image']
        assert img_path.exists()

        filename = create_deck([slide], path=Path(artifact_dir), deck_name="test_report_coverage")
        assert filename.exists()

    def test_report_diversity(self, dummy_dataset_od, diversity_outputs, artifact_dir):
        """Test diversity specific rollup values and action"""

        test_stage = DatasetBiasTestStage()
        test_stage.cache_base_path = artifact_dir
        test_stage.load_dataset(dummy_dataset_od, "DummyDataset")

        slide = test_stage._report_diversity(diversity_outputs)
        layout_args = slide["layout_arguments"]

        # Check rollup calculation and associated action
        assert "2 factors" in layout_args["text_column_body"][1].content[0].content
        assert (
            layout_args["text_column_body"][-1].content[0].content
            == "* Ensure balanced representation of all classes for all metadata"
        )

        # Check if image was saved
        img_path = layout_args['data_column_image']
        assert img_path.exists()

        filename = create_deck([slide], path=Path(artifact_dir), deck_name="test_report_diversity")
        assert filename.exists()

    def test_report_parity(self, dummy_dataset_od, parity_outputs, artifact_dir):
        """Test parity specific rollup values and action"""

        test_stage = DatasetBiasTestStage()
        test_stage.cache_base_path = artifact_dir
        test_stage.load_dataset(dummy_dataset_od, "DummyDataset")

        slide = test_stage._report_parity(parity_outputs)
        layout_args = slide["layout_arguments"]

        # Check rollup calculation and associated action
        assert "0 factors" in layout_args["text_column_body"][1].content[0].content
        assert layout_args["text_column_body"][-1].content[0].content == "* No action required"

        filename = create_deck([slide], path=Path(artifact_dir), deck_name="test_report_parity")
        assert filename.exists()

    def test_bias_gradient_pptx(
            self,
            dummy_dataset_od,
            coverage_outputs,
            balance_outputs,
            diversity_outputs,
            parity_outputs,
            artifact_dir
        ) -> None:
        """Test all gradient slide kwargs collected together"""

        test_stage: DatasetBiasTestStage = DatasetBiasTestStage()
        test_stage.cache_base_path = artifact_dir
        test_stage.load_dataset(dummy_dataset_od, "DummyDataset")

        test_stage.outputs = {
            test_stage.BALANCE_KEY: balance_outputs,
            test_stage.COVERAGE_KEY: coverage_outputs,
            test_stage.DIVERSITY_KEY: diversity_outputs,
            test_stage.PARITY_KEY: parity_outputs,
        }

        slides: list[dict[str, Any]] = test_stage.collect_report_consumables()

        filename = create_deck(slides, path=Path(artifact_dir), deck_name="test_bias_gradient_pptx")
        assert filename.exists()
