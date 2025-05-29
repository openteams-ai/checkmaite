"""Test Dataset Bias Analysis"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri._common.test_stages.impls.dataeval_bias_test_stage import (
    DataevalBiasBalanceOutputs,
    DataevalBiasConfig,
    DataevalBiasCoverageOutputs,
    DataevalBiasDiversityOutputs,
    DataevalBiasOutputs,
    DataevalBiasParityOutputs,
    DataevalBiasRun,
)
from jatic_ri.image_classification.test_stages.impls.dataeval_bias_test_stage import DatasetBiasTestStage
from jatic_ri.util.utils import save_figure_to_tempfile


@pytest.fixture(scope="module")
def fake_image() -> str:
    image = np.ones((28, 28, 3), dtype=int) * 200
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    return save_figure_to_tempfile(fig)


@pytest.fixture
def run(dummy_dataset_ic, fake_image):
    return DataevalBiasRun(
        test_stage_id=DatasetBiasTestStage().id,
        config=DataevalBiasConfig(device="cpu"),
        dataset_ids=[dummy_dataset_ic.metadata["id"]],
        model_ids=[],
        metric_id="",
        outputs=DataevalBiasOutputs(
            balance=DataevalBiasBalanceOutputs(
                balance=np.array([0.99999822, 0.13363788, 0.04505382, 0.02994455]),
                factors=np.array(
                    [
                        [0.99999843, 0.04133555, 0.09725766],
                        [0.04133555, 0.08433558, 0.1301489],
                        [0.09725766, 0.1301489, 0.99999856],
                    ]
                ),
                classwise=np.array([[0.99999822, 0.13363788, 0.0, 0.0], [0.99999822, 0.13363788, 0.0, 0.0]]),
                factor_names=[],
                class_names=[],
                image=fake_image,
            ),
            diversity=DataevalBiasDiversityOutputs(
                diversity_index=np.array([0.18103448, 0.18103448, 0.88636364]),
                classwise=np.array([[0.17241379, 0.39473684], [0.2, 0.2]]),
                factor_names=[],
                class_names=[],
                image=fake_image,
            ),
            parity=DataevalBiasParityOutputs(
                score=np.array([7.35731943, 5.46711299, 0.51506212]),
                p_value=np.array([0.28906231, 0.24263543, 0.77295762]),
                factor_names=["A", "B", "C"],
            ),
            coverage=DataevalBiasCoverageOutputs(
                total=len(dummy_dataset_ic),
                uncovered_indices=np.array([447, 412, 8, 32, 63]),
                coverage_radius=0.8459038956941765,
                critical_value_radii=np.arange(10),
                image=fake_image,
            ),
        ),
    )


def ignore_bias_warnings(test_fn):
    for filter in [
        "ignore:All samples look discrete with so few data points:UserWarning",
        r"ignore:Factors \[.*\] did not meet the recommended \d+ occurrences for each value-label combination:UserWarning",
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
            dummy_dataset_ic.data = [np.ones(shape=(3, i, i), dtype=np.float32) for i in range(1, 10)]

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dataset=dummy_dataset_ic, dataset_id="DummyDataset")
        run = test_stage.run(use_stage_cache=False)

        if homogeneous_size:
            assert run.outputs.coverage.image is not None

        output = test_stage.collect_report_consumables()
        assert len(output) == 4


class TestICBiasCollectReportConsumables:
    """Tests the individual report_consumables methods for every Bias metric, as well as the combined collect_report_consumables"""

    def test_report_balance(self, run, artifact_dir):
        """Test balance specific rollup values and action"""

        test_stage = DatasetBiasTestStage()

        slide = test_stage._report_balance(run.outputs.balance, run.dataset_ids[0])
        layout_args = slide["layout_arguments"]

        # Check rollup calculation and associated action
        assert "0 factors" in layout_args["line_section_body"][1].content[0].content
        assert layout_args["line_section_body"][-1].content[0].content == "• No action required"

        # Check if image was saved
        img_path = layout_args["item_section_body"]
        assert img_path.exists()

        filename = create_deck([slide], path=artifact_dir, deck_name="test_report_balance")
        assert filename.exists()

    def test_report_coverage(self, run, artifact_dir):
        """Test the coverage specific gradient output"""

        test_stage = DatasetBiasTestStage()

        slide = test_stage._report_coverage(run.outputs.coverage, run.dataset_ids[0])
        layout_args = slide["layout_arguments"]

        # Check text and visual slide arguments
        assert layout_args["title"] == f"Dataset: {run.dataset_ids[0]} | Category Bias"
        assert layout_args["line_section_heading"] == "Metric: Coverage"
        assert layout_args["line_section_half"] is True
        assert (
            layout_args["line_section_body"][-1].content[0].content
            == "• Increase representation of rare but relevant samples in areas of poor coverage"
        )

        # Test calculated dataframe values
        cov_df: pd.DataFrame = layout_args["item_section_table"]

        assert tuple(cov_df.columns) == ("Poor Coverage", "Threshold")
        assert cov_df.shape == (1, 2)  # One row, 2 columns
        assert cov_df["Poor Coverage"][0] == "5 of 10 (50.0%)"
        assert cov_df["Threshold"][0] == round(run.outputs.coverage.coverage_radius, 2)

        # Check if image was saved
        img_path = layout_args["item_section_bottom"]
        assert img_path.exists()

        filename = create_deck([slide], path=artifact_dir, deck_name="test_report_coverage")
        assert filename.exists()

    def test_report_diversity(self, run, artifact_dir):
        """Test diversity specific rollup values and action"""

        test_stage = DatasetBiasTestStage()

        slide = test_stage._report_diversity(run.outputs.diversity, run.dataset_ids[0])
        layout_args = slide["layout_arguments"]

        # Check rollup calculation and associated action
        assert "2 factors" in layout_args["line_section_body"][1].content[0].content
        assert (
            layout_args["line_section_body"][-1].content[0].content
            == "• Ensure balanced representation of all classes for all metadata"
        )

        # Check if image was saved
        img_path = layout_args["item_section_body"]
        assert img_path.exists()

        filename = create_deck([slide], path=artifact_dir, deck_name="test_report_diversity")
        assert filename.exists()

    def test_report_parity(self, run, artifact_dir):
        """Test parity specific rollup values and action"""

        test_stage = DatasetBiasTestStage()

        slide = test_stage._report_parity(run.outputs.parity, run.dataset_ids[0])
        layout_args = slide["layout_arguments"]

        # Check rollup calculation and associated action
        assert "0 factors" in layout_args["line_section_body"][1].content[0].content
        assert layout_args["line_section_body"][-1].content[0].content == "• No action required"

        filename = create_deck([slide], path=artifact_dir, deck_name="test_report_parity")
        assert filename.exists()

    def test_bias_gradient_pptx(self, run, artifact_dir) -> None:
        """Test all gradient slide kwargs collected together"""

        test_stage: DatasetBiasTestStage = DatasetBiasTestStage()

        test_stage._stored_run = run

        slides: list[dict[str, Any]] = test_stage.collect_report_consumables()

        filename = create_deck(slides, path=artifact_dir, deck_name="test_bias_gradient_pptx")
        assert filename.exists()
