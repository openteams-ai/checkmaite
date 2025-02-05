from jatic_ri.util.cache import SimpleRICacheOD
from jatic_ri.util.evaluation import EvaluationTool, SimpleDataLoader

from jatic_ri.object_detection.datasets import DetectionTarget

import pytest
from unittest.mock import MagicMock
import torch

# This is utilized from the od mock model operated on the od mock dataset. 
mock_pred_detection_object = DetectionTarget(
                                    boxes=torch.ones((5, 4)),
                                    labels=torch.tensor([0, 1, 2, 3, 4]),
                                    scores=torch.zeros((5, 5)),
    )
# This is utilized from the od mock dataset values in conftest.py. 
mock_image_data = torch.ones((1, 16, 16))

TInput = str  
TTarget = str
TMetadata = str

@pytest.fixture
def mock_dataset():
    """Fixture to create a mock dataset."""
    dataset = MagicMock()
    dataset.__len__.return_value = 10
    dataset.__getitem__.side_effect = lambda index: (f"input{index}", f"target{index}", f"metadata{index}")
    return dataset

@pytest.fixture
def data_loader(mock_dataset):
    """Fixture to create the SimpleDataLoader instance."""
    return SimpleDataLoader(mock_dataset, batch_size=3)

def test_simple_data_loader_init(data_loader, mock_dataset):
    """Test that the SimpleDataLoader is initialized correctly."""
    assert data_loader.dataset == mock_dataset
    assert data_loader.batch_size == 3

def test_iter(data_loader, mock_dataset):
    """Test that the iteration yields correct batches."""
    iterator = iter(data_loader)

    # First batch (index 0 to 2)
    batch_inputs, batch_targets, batch_metadata = next(iterator)
    assert len(batch_inputs) == 3
    assert batch_inputs == ['input0', 'input1', 'input2']
    assert batch_targets == ['target0', 'target1', 'target2']
    assert batch_metadata == ['metadata0', 'metadata1', 'metadata2']

    # Second batch (index 3 to 5)
    batch_inputs, batch_targets, batch_metadata = next(iterator)
    assert len(batch_inputs) == 3
    assert batch_inputs == ['input3', 'input4', 'input5']
    assert batch_targets == ['target3', 'target4', 'target5']
    assert batch_metadata == ['metadata3', 'metadata4', 'metadata5']

    # Third batch (index 6 to 8)
    batch_inputs, batch_targets, batch_metadata = next(iterator)
    assert len(batch_inputs) == 3
    assert batch_inputs == ['input6', 'input7', 'input8']
    assert batch_targets == ['target6', 'target7', 'target8']
    assert batch_metadata == ['metadata6', 'metadata7', 'metadata8']

    # Fourth batch (index 9 to 9, only one item)
    batch_inputs, batch_targets, batch_metadata = next(iterator)
    assert len(batch_inputs) == 1
    assert batch_inputs == ['input9']
    assert batch_targets == ['target9']
    assert batch_metadata == ['metadata9']

def test_collate(data_loader):
    """Test that the _collate method works correctly."""
    # Simulate a small batch of data as singles
    batch_data_as_singles = [
        ("input0", "target0", "metadata0"),
        ("input1", "target1", "metadata1"),
        ("input2", "target2", "metadata2")
    ]

    # Call the private _collate method directly
    batch_inputs, batch_targets, batch_metadata = data_loader._collate(batch_data_as_singles)

    assert batch_inputs == ["input0", "input1", "input2"]
    assert batch_targets == ["target0", "target1", "target2"]
    assert batch_metadata == ["metadata0", "metadata1", "metadata2"]

def test_empty_dataset():
    """Test that the SimpleDataLoader handles an empty dataset correctly."""
    empty_dataset = MagicMock()
    empty_dataset.__len__.return_value = 0
    empty_dataset.__getitem__.side_effect = lambda index: (None, None, None)

    data_loader = SimpleDataLoader(empty_dataset, batch_size=3)
    iterator = iter(data_loader)

    with pytest.raises(StopIteration):
        next(iterator)

def test_single_item_batch(data_loader):
    """Test that a single item batch is handled correctly."""
    dataset = MagicMock()
    dataset.__len__.return_value = 1
    dataset.__getitem__.side_effect = lambda index: (f"input{index}", f"target{index}", f"metadata{index}")

    data_loader = SimpleDataLoader(dataset, batch_size=3)
    iterator = iter(data_loader)

    batch_inputs, batch_targets, batch_metadata = next(iterator)
    assert len(batch_inputs) == 1
    assert batch_inputs == ["input0"]
    assert batch_targets == ["target0"]
    assert batch_metadata == ["metadata0"]


def test_prediction(dummy_model_od, dummy_dataset_od, tmpdir) -> None:

    model=dummy_model_od
    model_id="dummy1"
    dataset=dummy_dataset_od
    dataset_id="dummy1"

    dataloader = SimpleDataLoader(dataset, 2)
    evaluationtool = EvaluationTool(ri_cache=SimpleRICacheOD(cache_root_dir=tmpdir))

    pred, data = evaluationtool.predict(
        model=model,
        model_id=model_id,
        dataset=dataset,
        dataset_id=dataset_id,
        dataloader=dataloader,
        batch_size=2
    )

    # First, this test goes through the detection object that was found by the model.
    # There is only one to consider in this test.
    detection_object = pred[0][0]
    
    # Assert that the bounding box tensors of the computed prediction match the expected values.
    assert torch.equal(detection_object.boxes, mock_pred_detection_object.boxes)
    # Assert that the blabel tensors of the computed prediction match the expected values.
    assert torch.equal(detection_object.labels, mock_pred_detection_object.labels)
    # Assert that the score tensors of the computed prediction match the expected values.
    assert torch.equal(detection_object.scores, mock_pred_detection_object.scores)

    # Next, we test the data that was returned in batch format to test if it matches our original input.
    # Assert that the image data returned to the user match the expected values.
    assert torch.equal(data[0][0][0], mock_image_data)

def test_metric_compute(dummy_model_od, dummy_dataset_od, dummy_metric_od, tmpdir) -> None:
    metric_expected = {"fake_metric": 1.0}
    metric = dummy_metric_od
    model=dummy_model_od
    model_id="dummy1"
    dataset=dummy_dataset_od
    dataset_id="dummy1"

    dataloader = SimpleDataLoader(dataset, 2)
    evaluationtool = EvaluationTool(ri_cache=SimpleRICacheOD(cache_root_dir=tmpdir))

    pred, data = evaluationtool.predict(
        model=model,
        model_id=model_id,
        dataset=dataset,
        dataset_id=dataset_id,
        dataloader=dataloader,
        batch_size=2
    )
    metric = evaluationtool.compute_metric(
        metric=metric,
        filename="na",
        prediction=pred,
        data=data
    )
    # Assert that the dummy metric utilizes it's compute function.
    assert metric == metric_expected

def test_evaluation(dummy_model_od, dummy_dataset_od, dummy_metric_od, tmpdir):
    metric_results_expected = {"fake_metric": 1.0}
    metric = dummy_metric_od
    metric_id="fake_metric"
    model=dummy_model_od
    model_id="dummy1"
    dataset=dummy_dataset_od
    dataset_id="dummy1"

    dataloader = SimpleDataLoader(dataset, 2)
    evaluationtool = EvaluationTool(ri_cache=SimpleRICacheOD(cache_root_dir=tmpdir))

    result = evaluationtool.evaluate(
        model=model,
        model_id=model_id,
        dataset=dataset,
        dataset_id=dataset_id,
        dataloader=dataloader,
        metric=metric,
        metric_id=metric_id,
        batch_size=2,
        return_augmented_data=True,
        return_preds=True
    )

    # The metric results are the first part of the evaluate response.
    metric_result_value= result[0]
    # The second object is a tuple containing all of the prediction data used to compute the metric results.
    # The model's prediction is the first portion.
    pred= result[1]
    # Further more, we want to focus on the detection object inside the prediction.
    detection_object = pred[0][0]
    # The batch of image data used to produce the prediction is the second portion.
    data= result[2]

    # Assert that the metric results computed match the results expected.
    assert metric_result_value == metric_results_expected
    # Assert that the bounding box tensors of the computed prediction match the expected values.
    assert torch.equal(detection_object.boxes, mock_pred_detection_object.boxes)
    # Assert that the blabel tensors of the computed prediction match the expected values.
    assert torch.equal(detection_object.labels, mock_pred_detection_object.labels)
    # Assert that the score tensors of the computed prediction match the expected values.
    assert torch.equal(detection_object.scores, mock_pred_detection_object.scores)
    # Assert that the image data returned to the user match the expected values.
    assert torch.equal(data[0][0][0], mock_image_data)
