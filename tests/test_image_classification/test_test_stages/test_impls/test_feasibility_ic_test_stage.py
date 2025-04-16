"""Test Image Classification DataEval Feasibility Test Stage"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.image_classification.test_stages.impls.dataeval_feasibility_test_stage import DatasetFeasibilityTestStage


@pytest.fixture(scope="class")
def ber_outputs() -> dict[str, dict[str, float]]:
    return {
        "feasibility": {
            "ber": 0.75,
            "ber_lower": 0.5287908970299657,
        }
    }


class TestFeasibilityTestStage:
    """Tests the image classification DatasetFeasibilityTestStage implementation"""

    def test_cache_id(self):
        """Confirm cache is created after run, and is used if written"""
        test_stage = DatasetFeasibilityTestStage()
        test_stage.load_dataset(None, "Dataset")  # type: ignore
        test_stage.load_threshold(0.5)

        assert test_stage.cache_id == "feasibility_Dataset_0.5.json"

    @patch("jatic_ri.image_classification.test_stages.impls.dataeval_feasibility_test_stage.read_dataset")
    def test_run(self, mock_read_dataset: MagicMock):
        """Tests run against known outputs"""

        test_stage = DatasetFeasibilityTestStage()
        test_stage.load_dataset(None, "ICDataset")  # type: ignore

        mock_read_dataset.return_value = (
            np.ones((10, 1, 16, 16)),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            [],  # Unused by test stage
        )

        results = test_stage._run()

        assert results == {"feasibility": {"ber": 0.5, "ber_lower": 0.5}}

    def test_collect_report_consumable(self, ber_outputs, tmp_path):
        """"""

        test_stage = DatasetFeasibilityTestStage()
        test_stage.load_dataset(None, "ICDataset")  # type: ignore
        test_stage.load_threshold(0.5)
        test_stage.outputs = ber_outputs

        slides = test_stage.collect_report_consumables()

        assert len(slides) == 1
        slide = slides[0]

        assert slide["deck"] == "image_classification_dataset_evaluation"
        assert slide["layout_name"] == "TextData"

        layout_args = slide["layout_arguments"]
        table: pd.DataFrame = layout_args["data_column_table"]
        assert table["Feasible"][0] == "True"
        assert table["Bayes Error Rate"][0] == 0.75
        assert table["Lower Bayes Error Rate"][0] == 0.529
        assert table["Performance Goal"][0] == 0.5

        create_deck(slides, path=tmp_path, deck_name="DatasetFeasibilityDeck")
        assert (tmp_path / "DatasetFeasibilityDeck.pptx").exists()

    def test_cache(self, dummy_dataset_ic) -> None:
        test_stage = DatasetFeasibilityTestStage()
        test_stage.load_threshold(0.5)
        test_stage.load_dataset(dummy_dataset_ic, "Dataset1")

        test_stage.run(use_stage_cache=True)
        base_outputs = test_stage.outputs

        test_stage_cached = DatasetFeasibilityTestStage()
        test_stage_cached.load_threshold(0.5)
        test_stage_cached.load_dataset(dummy_dataset_ic, "Dataset1")
        test_stage_cached._run = MagicMock()
        test_stage_cached.run()
        cached_outputs = test_stage_cached.outputs

        assert base_outputs == cached_outputs
