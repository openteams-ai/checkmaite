import os

import numpy as np
import pytest
import torch

import shutil
from unittest.mock import MagicMock
from pathlib import Path
from typing import Sequence, Any

from jatic_ri.util.cache import JSONCache, NumpyEncoder, TensorEncoder, SimpleRICacheOD
from jatic_ri.object_detection.datasets import DetectionTarget

from copy import deepcopy


pred_data_dummy_od = (
        [
            [
                DetectionTarget(
                    boxes=torch.tensor(torch.tensor([[1, 2, 3], [4, 5, 6]])),
                    labels=torch.tensor(torch.tensor([[1, 2, 3], [4, 5, 6]])),
                    scores=torch.tensor(torch.tensor([[1, 2, 3], [4, 5, 6]])),
                )
            ]
        ],
        [
            (
                [torch.tensor([[1, 2, 3], [4, 5, 6]])],
                [
                    DetectionTarget(
                    boxes=torch.tensor(torch.tensor([[1, 2, 3], [4, 5, 6]])),
                    labels=torch.tensor(torch.tensor([[1, 2, 3], [4, 5, 6]])),
                    scores=torch.tensor(torch.tensor([[1, 2, 3], [4, 5, 6]])),
                )
                ],
                [
                    {
                        'license': 0, 
                        'file_name': '000000000000.jpg', 
                        'id': 0000
                    }
                ]
            )
        ]
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
    pred_data = deepcopy(pred_data_dummy_od)
    
    filename = "test_predictions.json"

    cache = SimpleRICacheOD(cache_path=tmpdir)

    cache.write_predictions(filename=filename, prediction=pred_data)

    cache_file_path = tmpdir.join(filename)
    
    assert cache_file_path.exists(), f"{cache_file_path} was not created."

    result = cache.read_predictions(filename)

    assert isinstance(result, tuple)
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)
    assert torch.equal(result[0][0][0].boxes, pred_data_dummy_od[0][0][0].boxes)
    assert torch.equal(result[0][0][0].labels, pred_data_dummy_od[0][0][0].labels)
    assert torch.equal(result[0][0][0].scores, pred_data_dummy_od[0][0][0].scores)
