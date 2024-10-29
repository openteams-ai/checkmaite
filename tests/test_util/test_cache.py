import os

import numpy as np
import pytest
import torch

from jatic_ri.util.cache import JSONCache, NumpyEncoder, TensorEncoder


def test_cache_read_none() -> None:
    cache = JSONCache()
    result = cache.read_cache("does_not_exist")
    assert result is None


def test_cache_write_read_dict_of_list(tmp_path) -> None:
    cache = JSONCache()
    data = {"a": 1, "b": [2, 2]}
    cache_path = tmp_path / "test_file.json"
    cache.write_cache(cache_path, data)
    assert os.path.exists(cache_path)
    assert data == cache.read_cache(cache_path)


def test_cache_write_read_list_of_dict(tmp_path) -> None:
    cache = JSONCache()
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


def test_numpy_cache(tmp_path) -> None:
    cache = JSONCache(NumpyEncoder)
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


def test_tensor_numpy_mixed_cache(tmp_path) -> None:
    cache = JSONCache(TensorEncoder)
    data = {
        "tensor": torch.tensor([1.0, 2.0]),
        "array": np.array([4.0, 5.0]),
        "value": 42,
        "string": "dummy_string",
    }
    cache_path = tmp_path / "test_file.json"
    cache.write_cache(cache_path, data)
    assert os.path.exists(cache_path)
    cache_result = cache.read_cache(cache_path)
    assert isinstance(cache_result, dict)
    assert cache_result == {
        "tensor": [1.0, 2.0],
        "array": [4.0, 5.0],
        "value": 42,
        "string": "dummy_string",
    }
