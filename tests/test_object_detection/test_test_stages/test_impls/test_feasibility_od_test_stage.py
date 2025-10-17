"""Test DataEval Feasibility Test Stage"""

from collections.abc import Sequence
from typing import Any

import maite.protocols.object_detection as od
import pandas as pd
import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.object_detection.test_stages.impls.dataeval_feasibility_test_stage import (
    DatasetObjectDetectionFeasibilityOutputs,
    DatasetObjectDetectionFeasibilityTestStage,
)


@pytest.mark.skip(reason="OD feasibility test stage is not available until MAITE>=0.8.0 is supported")
class TestFeasibilityTestStage:
    """Tests the DatasetFeasibilityTest test stage implementation"""

    def test_feasibility_teststage(self, fake_od_model_default, fake_od_dataset_default) -> None:
        """Test FeasibilityTestStage implementation runs without error on dummy data"""

        # Only need to mock first `predict` return value
        # Tuple[Sequence[Sequence[TargetType]], _]]

        # Enforce type annotations
        batch: Sequence[od.TargetType] = []
        target_batches: Sequence[Sequence[od.TargetType]] = []

        # Simulate batch creation of DataLoader due to type hinting issues
        for i in range(len(fake_od_dataset_default)):
            data = fake_od_dataset_default[i]
            batch.append(data[1])
            if i % 2:  # Batch size of 2
                target_batches.append(batch)
                batch = []

        if batch:  # Append values that remain when dummy_dataset is not divisible by batch size (2)
            target_batches.append(batch)

        test_stage = DatasetObjectDetectionFeasibilityTestStage()
        test_stage.load_model(model=fake_od_model_default, model_id="model_1")
        test_stage.load_threshold(threshold=10)
        test_stage.load_dataset(dataset=fake_od_dataset_default, dataset_id="dataset_1")
        test_stage.run(use_stage_cache=False)

        assert isinstance(test_stage.outputs, DatasetObjectDetectionFeasibilityOutputs)
        assert test_stage.outputs.uap == 0.75

    @pytest.mark.parametrize("uap", [0.0, 0.5, 1.0])
    def test_collect_report_consumables(self, uap) -> None:
        """Test dataframe and gradient slide creation"""

        threshold = 0.5
        test_stage = DatasetObjectDetectionFeasibilityTestStage()
        test_stage.load_threshold(threshold)
        test_stage.load_dataset(None, "DUMMY_ID")  # pyright: ignore[reportArgumentType]
        test_stage.load_model(None, "model")  # pyright: ignore[reportArgumentType]
        test_stage.outputs = DatasetObjectDetectionFeasibilityOutputs(uap=uap)  # pyright: ignore[reportAttributeAccessIssue]

        slides = test_stage.collect_report_consumables()
        assert len(slides) == 1  # Only UAP

        slide = slides[0]

        # Check slide metadata
        assert slide["deck"] == "object_detection_dataset_evaluation"
        assert slide["layout_name"] == "SectionByItem"

        # Check table populated with results
        layout_args = slide["layout_arguments"]
        table: pd.DataFrame = layout_args["item_section_body"]
        assert table["Performance Goal"][0] == threshold
        assert table["Upperbound Average Precision"][0] == uap
        assert table["Feasible"][0] == str(uap >= threshold)

    def test_feasibility_gradient_pptx(self, artifact_dir) -> None:
        """This is used to test the output of the feasibility gradient slides"""
        teststage: DatasetObjectDetectionFeasibilityTestStage = DatasetObjectDetectionFeasibilityTestStage()

        # we only need the IDs here
        teststage.load_dataset(None, "VOC")
        teststage.load_model(None, "model")
        teststage.load_threshold(0.5)

        teststage.outputs = DatasetObjectDetectionFeasibilityOutputs(uap=0.75)

        slides: list[dict[str, Any]] = teststage.collect_report_consumables()
        filename = create_deck(slides, path=artifact_dir, deck_name="feasibility")
        assert filename.exists()
