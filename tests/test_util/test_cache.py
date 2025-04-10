import os

import numpy as np
import pytest
import torch
from maite.workflows import evaluate

from jatic_ri.object_detection.datasets import DetectionTarget
from jatic_ri.util.cache import JSONCache, NumpyEncoder, SimpleRICacheIC, SimpleRICacheOD, TensorEncoder
from tests.fake_ic_classes import FakeICDataset, FakeICModel

metric_results_expected = {"fake_metric": torch.tensor(1.0)}

pred_data_dummy_od = (
    [
        [
            DetectionTarget(
                boxes=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                labels=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                scores=torch.tensor([[1, 2, 3], [4, 5, 6]]),
            )
        ]
    ],
    [
        (
            [torch.tensor([[1, 2, 3], [4, 5, 6]])],
            [
                DetectionTarget(
                    boxes=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                    labels=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                    scores=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                )
            ],
            [{"license": 0, "file_name": "000000000000.jpg", "id": 0000}],
        )
    ],
)


def test_cache_read_none() -> None:
    cache = JSONCache()
    result = cache.read_cache("does_not_exist")
    assert result is None


@pytest.mark.parametrize("compress", [True, False])
def test_cache_write_read_dict_of_list(compress, tmp_path) -> None:
    cache = JSONCache(compress=compress)
    data = {"a": 1, "b": [2, 2]}
    cache_path = tmp_path / "test_file.json"
    cache.write_cache(cache_path, data)
    assert os.path.exists(cache_path)
    assert data == cache.read_cache(cache_path)


@pytest.mark.parametrize("compress", [True, False])
def test_cache_write_read_list_of_dict(compress, tmp_path) -> None:
    cache = JSONCache(compress=compress)
    data = [{"a": 1}, {"b": [2], "c": 3}]
    cache_path = tmp_path / "test_file.json"
    cache.write_cache(cache_path, data)
    assert os.path.exists(cache_path)
    assert data == cache.read_cache(cache_path)


def test_make_dirs(tmp_path) -> None:
    cache = JSONCache()
    cache_path = tmp_path / "nested" / "folder" / "test_file.json"
    cache.write_cache(cache_path, {"a": 1})
    assert os.path.exists(cache_path)


def test_numpy_encoder() -> None:
    ne = NumpyEncoder()
    assert ne.default(np.float64(0.25)) == 0.25
    assert ne.default(np.int8(3)) == 3
    with pytest.raises(TypeError):
        ne.default("dummy_string")


@pytest.mark.parametrize("compress", [True, False])
def test_numpy_cache(compress, tmp_path) -> None:
    cache = JSONCache(NumpyEncoder, compress=compress)
    array = [1, 2, 3]
    data = {"a": np.array(array), "b": np.float64(0.25), "c": np.int8(3), "d": "dummy_string"}
    cache_path = tmp_path / "test_file.json"
    cache.write_cache(cache_path, data)
    assert os.path.exists(cache_path)
    cache_result = cache.read_cache(cache_path)
    assert isinstance(cache_result, dict)
    assert cache_result == {"a": array, "b": 0.25, "c": 3, "d": "dummy_string"}


def test_tensor_encoder() -> None:
    te = TensorEncoder()
    assert te.default(torch.tensor([1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]
    with pytest.raises(TypeError):
        te.default("dummy_string")


def test_write_read_predictions_od(tmpdir):
    """Testing the writing and reading of an OD prediction to cache"""

    filename = "test_predictions.json"

    cache = SimpleRICacheOD(cache_root_dir=tmpdir)

    cache.write_predictions(filename=filename, prediction=pred_data_dummy_od)

    cache_file_path = tmpdir.join(filename)

    # Asserts that the cache write triggered correctly.
    assert cache_file_path.exists(), f"{cache_file_path} was not created."

    result = cache.read_predictions(filename)

    assert result is not None
    # The result is a detection object for one image.
    # Given the complext return structure, this tests that the return types are the correct instances.
    prediction = result[0]
    data = result[1]
    assert isinstance(result, tuple)
    assert isinstance(prediction, list)
    assert isinstance(data, list)

    # This test focuses on the prediction object.
    # The prediction object is a list of detection objects.
    # There is only one image, so one set of detection objects to test.
    detection_object = prediction[0][0]
    # Asserts that the bounding boxes have been retrieved an encoded correctly.
    assert torch.equal(detection_object.boxes, torch.as_tensor(pred_data_dummy_od[0][0][0].boxes))
    # Asserts that the labels have been retrieved an encoded correctly.
    assert torch.equal(detection_object.labels, torch.as_tensor(pred_data_dummy_od[0][0][0].labels))
    # Asserts that the scores have been retrieved an encoded correctly.
    assert torch.equal(detection_object.scores, torch.as_tensor(pred_data_dummy_od[0][0][0].scores))


def test_write_read_metric_result_od(tmpdir):
    """Testing the writing and reading of an OD prediction to cache"""

    filename = "test_metric_result.json"

    cache = SimpleRICacheOD(cache_root_dir=tmpdir)

    cache.write_metric(filename=filename, metric_results=metric_results_expected)

    cache_file_path = tmpdir.join(filename)

    # Asserts that the cache write was triggered correctly.
    assert cache_file_path.exists(), f"{cache_file_path} was not created."

    result = cache.read_metric(filename)

    # Asserts that the metric result has been retrieved in the correct format.
    assert isinstance(result, dict)
    # Asserts that the metric result has been encoded in the correct format.
    assert torch.equal(result["fake_metric"], metric_results_expected["fake_metric"])


def test_write_read_predictions_ic(tmpdir, fake_ic_model_default: FakeICModel, fake_ic_dataset_default: FakeICDataset):
    """Testing the writing and reading of IC predictions to cache"""

    filename = "test_ic_predictions.json"
    cache = SimpleRICacheIC(cache_root_dir=tmpdir)

    # Using the MAITE evaluate workflow and the default fake IC dataset and model, generate some dummy predictions.
    _, fake_preds_batches, fake_data_batches = evaluate(
        model=fake_ic_model_default,
        dataset=fake_ic_dataset_default,
        return_preds=True,
        return_augmented_data=True,
        batch_size=5,
    )
    cache.write_predictions(filename=filename, prediction=((fake_preds_batches, fake_data_batches)))

    cache_file_path = tmpdir.join(filename)

    # Asserts that the cache write triggered correctly.
    assert cache_file_path.exists(), f"{cache_file_path} was not created."

    result = cache.read_predictions(filename)

    # Confirm cache hit, not None
    assert isinstance(result, tuple)

    cache_preds_batches = result[0]
    cache_data_batches = result[1]

    # Check same number of batches.   4 = default dataset len 20 / batch size 5
    assert len(cache_preds_batches) == len(cache_data_batches) == len(fake_data_batches) == len(fake_preds_batches) == 4

    # Check prediction values equal after extracted from cache
    for i in range(len(cache_preds_batches)):
        for j in range(len(cache_preds_batches[i])):
            assert torch.equal(cache_preds_batches[i][j], fake_preds_batches[i][j])
