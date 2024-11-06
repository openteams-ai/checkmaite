"""Caching utility class for Test Stages"""

import json
from os import makedirs, path
from typing import Any, Generic, Optional, TypeVar

import numpy as np
import torch
import zstandard as zstd

from jatic_ri._common.test_stages.interfaces.test_stage import Cache

TData = TypeVar("TData", dict, list)


class JSONCache(Generic[TData], Cache[TData]):
    """Basic JSON file based caching plugin"""

    def __init__(self, encoder: type = json.JSONEncoder, compress: bool = False) -> None:
        self.encoder = encoder
        self.compress = compress

    def read_cache(self, cache_path: str) -> Optional[TData]:
        """Read cache from file and returns as dictionary"""
        if path.exists(cache_path):
            with open(cache_path, "rb") as file:
                raw = file.read()
            contents = zstd.decompress(raw).decode("utf-8") if self.compress else raw.decode("utf-8")
            return json.loads(contents)
        return None

    def write_cache(self, cache_path: str, data: TData) -> None:
        """Writes dictionary to file using serializer"""
        dirname = path.dirname(cache_path)
        if dirname and not path.exists(dirname):
            makedirs(dirname)
        contents = json.dumps(data, cls=self.encoder).encode(encoding="utf-8")
        if self.compress:
            contents = zstd.compress(contents)
        with open(cache_path, "wb") as file:
            file.write(contents)


class NumpyEncoder(json.JSONEncoder):
    """Convert numpy objects to serializable native objects"""

    def default(self, o: Any) -> Any:  # noqa: ANN401
        """JSON encoding entry point"""
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# Custom JSON encoder to handle PyTorch tensors and NumPy arrays
class TensorEncoder(json.JSONEncoder):
    """Convert Pytorch tensors and numpy arrays to serializable objects"""

    def default(self, o: Any) -> Any:  # noqa: ANN401
        """JSON encoding entry point"""
        if isinstance(o, torch.Tensor):
            return o.tolist()  # Convert tensor to list
        if isinstance(o, np.ndarray):
            return o.tolist()  # Convert NumPy array to list
        if isinstance(o, (np.floating, np.integer)):
            return o.item()  # Convert NumPy scalar to a Python scalar
        return super().default(o)


# class ParquetCache(Cache[pd.DataFrame]):
#     """Basic file based caching for dataframes to parquet"""

#     def read_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
#         """Read cache from file and returns as dictionary"""
#         if path.exists(cache_path):
#             return pd.read_parquet(cache_path)
#         return None

#     def write_cache(self, cache_path: str, data: pd.DataFrame) -> None:
#         """Writes dictionary to file using serializer"""
#         data.to_parquet(cache_path)
