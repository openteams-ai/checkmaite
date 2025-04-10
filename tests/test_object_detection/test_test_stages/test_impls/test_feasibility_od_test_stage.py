"""Test DataEval Feasibility Test Stage"""

from collections.abc import Sequence
from typing import Any

import maite.protocols.object_detection as od
import pandas as pd
import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.object_detection.test_stages.impls.dataeval_feasibility_test_stage import DatasetFeasibilityTestStage


class TestFeasibilityTestStage:
    """Tests the DatasetFeasibilityTest test stage implementation"""

    def test_feasibility_teststage(self, dummy_model_od, dummy_dataset_od, default_eval_tool_no_cache) -> None:
        """Test FeasibilityTestStage implementation runs without error on dummy data"""

        # Only need to mock first `predict` return value
        # Tuple[Sequence[TargetBatchType], _]]

        # Enforce type annotations
        batch: od.TargetBatchType = []
        target_batches: Sequence[od.TargetBatchType] = []

        # Simulate batch creation of DataLoader due to type hinting issues
        for i in range(len(dummy_dataset_od)):
            data = dummy_dataset_od[i]
            batch.append(data[1])
            if i % 2:  # Batch size of 2
                target_batches.append(batch)
                batch = []

        if batch:  # Append values that remain when dummy_dataset is not divisible by batch size (2)
            target_batches.append(batch)

        test_stage = DatasetFeasibilityTestStage()
        test_stage.load_model(model=dummy_model_od, model_id="model_1")
        test_stage.load_threshold(threshold=10)
        test_stage.load_dataset(dataset=dummy_dataset_od, dataset_id="dataset_1")
        test_stage.load_eval_tool(default_eval_tool_no_cache)
        test_stage.run(use_stage_cache=False)

        assert "uap" in test_stage.outputs
        assert test_stage.outputs["uap"]

    @pytest.mark.parametrize("uap", [0.0, 0.5, 1.0])
    def test_collect_report_consumables(self, uap) -> None:
        """Test dataframe and gradient slide creation"""

        threshold = 0.5
        test_stage = DatasetFeasibilityTestStage()
        test_stage.load_threshold(threshold)
        test_stage.load_dataset(None, "DUMMY_ID")  # type: ignore
        test_stage.outputs = {"uap": uap}

        slides = test_stage.collect_report_consumables()
        assert len(slides) == 1  # Only UAP

        slide = slides[0]

        # Check slide metadata
        assert slide["deck"] == "object_detection_dataset_evaluation"
        assert slide["layout_name"] == "TextData"

        # Check table populated with results
        layout_args = slide["layout_arguments"]
        table: pd.DataFrame = layout_args["data_column_table"]
        assert table["Performance Goal"][0] == threshold
        assert table["Upperbound Average Precision"][0] == uap
        assert table["Feasible"][0] == str(uap >= threshold)

    def test_cache_id(self) -> None:
        """Tests the unique cache id is set based on the ids of dataset and model"""

        test_stage = DatasetFeasibilityTestStage()
        test_stage.load_dataset(None, "Dataset1")  # type: ignore
        test_stage.load_model(None, "Model1")  # type: ignore
        test_stage.load_threshold(0.5)
        assert test_stage.cache_id == "feasibility_Dataset1_Model1_0.5.json"


def test_feasibility_gradient_pptx(artifact_dir) -> None:
    """This is used to test the output of the feasibility gradient slides"""
    teststage: DatasetFeasibilityTestStage = DatasetFeasibilityTestStage()
    teststage.load_dataset(None, "VOC")  # type: ignore -> Only needs a dataset_id
    teststage.outputs = {"uap": 0.75}
    teststage.threshold = 0.5

    slides: list[dict[str, Any]] = teststage.collect_report_consumables()
    filename = create_deck(slides, path=artifact_dir, deck_name="feasibility")
    assert filename.exists()
