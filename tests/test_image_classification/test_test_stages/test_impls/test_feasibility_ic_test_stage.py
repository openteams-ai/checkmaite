"""Test Image Classification DataEval Feasibility Test Stage"""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.image_classification.test_stages import (
    DatasetImageClassificationFeasibilityConfig,
    DatasetImageClassificationFeasibilityOutputs,
    DatasetImageClassificationFeasibilityTestStage,
)
from jatic_ri.image_classification.test_stages._impls.dataeval_feasibility_test_stage import (
    DatasetImageClassificationFeasibilityRun,
)


@pytest.fixture(scope="class")
def ber_outputs() -> DatasetImageClassificationFeasibilityOutputs:
    return DatasetImageClassificationFeasibilityOutputs(ber=0.7, ber_lower=0.49013621203813906)


class TestFeasibilityTestStage:
    """Tests the image classification DatasetFeasibilityTestStage implementation"""

    def test_run(self, fake_ic_dataset_default):
        """Tests run against dummy dataset"""

        test_stage = DatasetImageClassificationFeasibilityTestStage()

        results = test_stage.run(datasets=[fake_ic_dataset_default], models=[], metrics=[])

        assert isinstance(results.outputs, DatasetImageClassificationFeasibilityOutputs)
        assert results.outputs.ber == 0.7
        assert results.outputs.ber_lower == 0.49013621203813906

    def test_collect_report_consumable(self, ber_outputs, tmp_path):
        """"""
        output = DatasetImageClassificationFeasibilityRun(
            dataset_metadata=[{"id": "ICDataset"}],
            metric_metadata=[],
            model_metadata=[],
            test_stage_id="test-stage-id",
            config=DatasetImageClassificationFeasibilityConfig(),
            outputs=ber_outputs,
        )

        slides = output.collect_report_consumables(threshold=0.5)

        assert len(slides) == 1
        slide = slides[0]

        assert slide["deck"] == "test-stage-id"
        assert slide["layout_name"] == "SectionByItem"

        layout_args = slide["layout_arguments"]
        table: pd.DataFrame = layout_args["item_section_body"]
        assert table["Feasible"][0] == "True"
        assert table["Bayes Error Rate"][0] == 0.7
        assert table["Lower Bayes Error Rate"][0] == 0.490
        assert table["Performance Goal"][0] == 0.5

        create_deck(slides, path=tmp_path, deck_name="DatasetFeasibilityDeck")
        assert (tmp_path / "DatasetFeasibilityDeck.pptx").exists()

    def test_cache(self, dummy_dataset_ic) -> None:
        test_stage = DatasetImageClassificationFeasibilityTestStage()

        run = test_stage.run(use_stage_cache=True, datasets=[dummy_dataset_ic])
        base_outputs = run.outputs

        test_stage_cached = DatasetImageClassificationFeasibilityTestStage()

        test_stage_cached._run = MagicMock()
        cached_run = test_stage_cached.run(datasets=[dummy_dataset_ic])
        cached_outputs = cached_run.outputs

        assert base_outputs == cached_outputs
