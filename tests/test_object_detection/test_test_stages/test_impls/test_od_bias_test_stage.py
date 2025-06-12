"""Test Dataset Bias Analysis"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri._common.test_stages.impls.dataeval_bias_test_stage import (
    DataevalBiasBalanceOutputs,
    DataevalBiasConfig,
    DataevalBiasCoverageOutputs,
    DataevalBiasDiversityOutputs,
    DataevalBiasOutputs,
    DataevalBiasRun,
)
from jatic_ri.object_detection.test_stages.impls.dataeval_bias_test_stage import DatasetBiasTestStage
from jatic_ri.util.utils import save_figure_to_tempfile


@pytest.fixture(scope="module")
def fake_image() -> str:
    image = np.ones((28, 28, 3), dtype=int) * 200
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    return save_figure_to_tempfile(fig)


@pytest.fixture
def run(dummy_dataset_od, fake_image):
    return DataevalBiasRun(
        test_stage_id=DatasetBiasTestStage().id,
        config=DataevalBiasConfig(device="cpu"),
        dataset_ids=[dummy_dataset_od.metadata["id"]],
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
                image_metadata=fake_image,
                image_classwise=fake_image,
            ),
            diversity=DataevalBiasDiversityOutputs(
                diversity_index=np.array([0.18103448, 0.18103448, 0.88636364]),
                classwise=np.array([[0.17241379, 0.39473684], [0.2, 0.2]]),
                factor_names=[],
                class_names=[],
                image=fake_image,
            ),
            coverage=DataevalBiasCoverageOutputs(
                total=len(dummy_dataset_od),
                uncovered_indices=np.array([447, 412, 8, 32, 63]),
                coverage_radius=0.8459038956941765,
                critical_value_radii=np.arange(10),
                image=fake_image,
            ),
        ),
    )


class TestODDatasetBiasRun:
    """Test shared Bias TestStage _run functionality between balance, coverage, diversity, and parity"""

    @pytest.mark.filterwarnings(
        r"ignore:Factors \[.*\] did not meet the recommended \d+ occurrences for each value-label combination:UserWarning"
    )
    @pytest.mark.parametrize("dataset_type", ["default", "with_target_metadata", "non_homogenous_size"])
    def test_run_and_report(self, dummy_dataset_od, dummy_dataset_od_with_target_metadata, dataset_type) -> None:
        if dataset_type == "default":
            dataset = dummy_dataset_od
        elif dataset_type == "with_target_metadata":
            dataset = dummy_dataset_od_with_target_metadata
        elif dataset_type == "non_homogenous_size":
            dataset = dummy_dataset_od
            # Modify images to be non-homogenous (like VOC)
            dataset.data = [np.ones(shape=(3, i, i)) for i in range(1, 10)]
        else:
            raise ValueError(dataset_type)

        test_stage = DatasetBiasTestStage()
        test_stage.load_dataset(dataset=dataset, dataset_id="DummyDataset")
        run = test_stage.run(use_stage_cache=False)

        if dataset_type != "non_homogenous_size":
            assert run.outputs.coverage.image is not None

        output = test_stage.collect_report_consumables()
        assert len(output) == 5

    @pytest.mark.filterwarnings(
        r"ignore:Factors \[.*\] did not meet the recommended \d+ occurrences for each value-label combination:UserWarning"
    )
    def test_coco(self):
        from os import path

        import tests
        from jatic_ri import PACKAGE_DIR
        from jatic_ri.object_detection.datasets import CocoDetectionDataset

        coco_dataset_dir = PACKAGE_DIR.parent.parent.joinpath(
            path.dirname(tests.__file__),
            ("testing_utilities/example_data/coco_resized_val2017"),
        )
        coco_dataset = CocoDetectionDataset(
            root=str(coco_dataset_dir),
            ann_file=str(coco_dataset_dir.joinpath("instances_val2017_resized_6.json")),
        )

        stage = DatasetBiasTestStage()

        stage.load_dataset(dataset=coco_dataset, dataset_id="asd")

        stage.run(use_stage_cache=False)
        pass  # no explosions

    def test_no_metadata(self):
        """Test that the bias test stage works when the dataset has no metadata"""
        from os import path

        import tests
        from jatic_ri import PACKAGE_DIR
        from jatic_ri.object_detection.datasets import CocoDetectionDataset

        coco_dataset_dir = PACKAGE_DIR.parent.parent.joinpath(
            path.dirname(tests.__file__),
            ("testing_utilities/example_data/coco_resized_val2017"),
        )
        coco_dataset = CocoDetectionDataset(
            root=str(coco_dataset_dir),
            ann_file=str(coco_dataset_dir.joinpath("instances_val2017_resized_6.json")),
        )

        coco_keys = ["license", "file_name", "coco_url", "height", "width", "date_captured", "flickr_url", "id"]
        for _, _, metadata in coco_dataset:
            assert set(metadata.keys()) == set(coco_keys)

        metadata_to_exclude = [
            "license",
            "file_name",
            "coco_url",
            "height",
            "width",
            "date_captured",
            "flickr_url",
            "id",
        ]

        stage = DatasetBiasTestStage(metadata_to_exclude=metadata_to_exclude)

        stage.load_dataset(dataset=coco_dataset, dataset_id="no_metadata")

        stage.run(use_stage_cache=False)
        pass  # no explosions


class TestODBiasCollectReportConsumables:
    """Tests the individual report_consumables methods for every Bias metric, as well as the combined collect_report_consumables"""

    def test_report_balance(self, run, artifact_dir):
        """Test balance specific rollup values and action"""

        test_stage = DatasetBiasTestStage()

        slide = test_stage._report_balance_metadata_factors(run.outputs.balance)
        layout_args = slide["layout_arguments"]

        # Check if image was saved
        img_path = layout_args["item_section_body"]
        assert img_path.exists()

        filename = create_deck([slide], path=artifact_dir, deck_name="test_report_balance")
        assert filename.exists()

    def test_report_coverage(self, run, artifact_dir):
        """Test the coverage specific gradient output"""

        test_stage = DatasetBiasTestStage()

        slide = test_stage._report_coverage(run.outputs.coverage)

        filename = create_deck([slide], path=artifact_dir, deck_name="test_report_coverage")
        assert filename.exists()

    def test_report_diversity(self, run, artifact_dir):
        """Test diversity specific rollup values and action"""

        test_stage = DatasetBiasTestStage()

        slide = test_stage._report_diversity(run.outputs.diversity)
        layout_args = slide["layout_arguments"]

        # Check if image was saved
        img_path = layout_args["item_section_body"]
        assert img_path.exists()

        filename = create_deck([slide], path=artifact_dir, deck_name="test_report_diversity")
        assert filename.exists()

    def test_bias_gradient_pptx(self, run, artifact_dir) -> None:
        """Test all gradient slide kwargs collected together"""

        test_stage: DatasetBiasTestStage = DatasetBiasTestStage()
        test_stage._stored_run = run

        slides: list[dict[str, Any]] = test_stage.collect_report_consumables()

        filename = create_deck(slides, path=Path(artifact_dir), deck_name="test_bias_gradient_pptx")
        assert filename.exists()
