"""Test DataEval Feasibility Test Stage"""

from unittest.mock import MagicMock, patch

import maite.protocols.object_detection as od

from jatic_ri.object_detection.test_stages.impls.feasibility_test_stage import FeasibilityTestStage


@patch("jatic_ri.object_detection.test_stages.impls.feasibility_test_stage.evaluate")
def test_feasibility_teststage(mock_evaluate: MagicMock, dummy_model, dummy_dataset) -> None:
    """Test FeasibilityTestStage implementation"""

    # Creates Sequence[DummyObjectDetectionTarget] with length (batch_size) of 3
    target_batch: od.TargetBatchType = [dummy_dataset[i][1] for i in range(3)]
    # List of predictions per batch
    target_batches = [target_batch, target_batch]
    # uap does not care about the metric values or metadata
    mock_evaluate.return_value = ([], target_batches, [])

    test = FeasibilityTestStage()
    test.load_model(model=dummy_model, model_id="model_1")
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dummy_dataset, dataset_id="dataset_1")
    test.run()
    test.collect_report_consumables()

    assert test.outputs is not None
    assert test.outputs["uap"]


def test_empty_return() -> None:
    assert FeasibilityTestStage().collect_report_consumables() == []
