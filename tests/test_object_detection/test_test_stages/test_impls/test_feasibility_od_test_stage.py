"""Test DataEval Feasibility Test Stage"""

from collections.abc import Sequence
from typing import Any

import maite.protocols.object_detection as od
import pandas as pd
import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.object_detection.test_stages import (
    DatasetObjectDetectionFeasibilityConfig,
    DatasetObjectDetectionFeasibilityOutputs,
    DatasetObjectDetectionFeasibilityTestStage,
)
from jatic_ri.object_detection.test_stages._impls.dataeval_feasibility_test_stage import (
    DatasetObjectDetectionFeasibilityRun,
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
        run = test_stage.run(use_stage_cache=False, datasets=[fake_od_dataset_default], models=[fake_od_model_default])

        assert isinstance(run.outputs, DatasetObjectDetectionFeasibilityOutputs)
        assert run.outputs.uap == 0.75

    @pytest.mark.parametrize("uap", [0.0, 0.5, 1.0])
    def test_collect_report_consumables(self, uap) -> None:
        """Test dataframe and gradient slide creation"""

        threshold = 0.5
        config = DatasetObjectDetectionFeasibilityConfig()
        run = DatasetObjectDetectionFeasibilityRun(
            test_stage_id="fake-id",
            config=config,
            dataset_metadata=[],
            model_metadata=[],
            metric_metadata=[],
            outputs=DatasetObjectDetectionFeasibilityOutputs(uap=0.75),
        )

        slides = run.collect_report_consumables(threshold=threshold, deck="object_detection_dataset_evaluation")
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

        config = DatasetObjectDetectionFeasibilityConfig()
        run = DatasetObjectDetectionFeasibilityRun(
            test_stage_id="fake-id",
            config=config,
            dataset_metadata=[],
            model_metadata=[],
            metric_metadata=[],
            outputs=DatasetObjectDetectionFeasibilityOutputs(uap=0.75),
        )

        slides: list[dict[str, Any]] = run.collect_report_consumables(threshold=0.5)
        filename = create_deck(slides, path=artifact_dir, deck_name="feasibility")
        assert filename.exists()
