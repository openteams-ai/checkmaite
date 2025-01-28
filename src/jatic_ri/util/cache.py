"""Caching utility class for Test Stages"""

import glob
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from os import getcwd, makedirs, path, remove
from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch
import zstandard as zstd
from maite._internals.protocols.generic import Metric

from jatic_ri._common.test_stages.interfaces.test_stage import Cache
from jatic_ri.object_detection.datasets import DetectionTarget

TData = TypeVar("TData", dict, list)
TMetric = TypeVar("TMetric", bound=Metric)

TMetricResultKeyValue = Union[torch.Tensor, float]
TMetricResultCache = dict[Any, TMetricResultKeyValue]


class JSONCache(Cache[TData]):
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


class RICache(ABC):
    """Abstract Class for using cache for evaluation and prediction"""

    @abstractmethod
    def read_predictions(
        self, filename: str
    ) -> Optional[tuple[Sequence[list[DetectionTarget]], Sequence[tuple[Any, Any, Any]]]]:
        """Reads a prediction from the cache"""
        pass

    @abstractmethod
    def write_predictions(
        self,
        filename: str,
        prediction: tuple[Sequence[list[DetectionTarget]], Sequence[tuple[Any, Any, Any]]],
    ) -> None:
        """Writes a prediction to the cache"""
        pass

    @abstractmethod
    def read_metric(self, filename: str) -> Union[TMetricResultCache, None]:
        """Reads a metric from the cache"""
        pass

    @abstractmethod
    def write_metric(self, filename: str, metric_results: TMetricResultCache) -> None:
        """Writes a metric to the cache"""
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clears the cache dir"""
        pass


class SimpleRICacheOD(RICache):
    """
    A wrapper for the RICache class that sets up a cache directory and provides methods for reading
    and writing prediction and metric data to/from the cache using JSON format.

    This is designed for Object Detection in particular.

    Parameters
    ----------
    json_cache:
        an attribute on this class that is hardcoded to JSONCache with a TensorEncorder and compression enabled.
        It controls the underlying read/write of the cache data.
    cache_root_dir:
        this dictates the root path where all of the folders for the cache files will be created.
    """

    def __init__(self, cache_root_dir: str = "") -> None:
        """
        Initializes the SimpleRICacheOD instance, setting up the cache directory and the JSON cache.

        Parameters
        ----------
        cache_root_dir:
            The directory where the cache files will be stored. If not provided, the current working
            directory will be used. Default is "" (empty string).
        """
        if cache_root_dir is None or cache_root_dir == "":
            cache_root_dir = getcwd()
        self.json_cache = JSONCache(encoder=TensorEncoder, compress=True)
        self.cache_root_dir = cache_root_dir
        makedirs(path.dirname(cache_root_dir), exist_ok=True)

    def read_predictions(
        self, filename: str
    ) -> Optional[tuple[Sequence[list[DetectionTarget]], Sequence[tuple[Any, Any, Any]]]]:
        """
        Reads prediction data from the cache using JSONCache.

        Parameters
        ----------
        filename : str
            The name of the cache file containing the prediction data.

        Returns
        -------
        tuple or None
            A tuple containing two sequences:
            - A sequence of lists of `DetectionTarget` objects.
            - A sequence of tuples containing metadata information.
            Returns None if the cache read is unsuccessful.
        """
        cachefile = path.join(self.cache_root_dir, filename)
        cache_hit = self.json_cache.read_cache(cachefile)
        if cache_hit:
            return self.read_prediction_serializer(cache_hit)
        return cache_hit

    def write_predictions(
        self,
        filename: str,
        prediction: tuple[Sequence[list[DetectionTarget]], Sequence[tuple[Any, Any, Any]]],
    ) -> None:
        """
        Writes prediction data to the cache using JSONCache.

        Parameters
        ----------
        filename : str
            The name of the cache file where the prediction data will be written.

        prediction : tuple
            A tuple containing two sequences:
            - A sequence of lists of `DetectionTarget` objects.
            - A sequence of tuples containing metadata information.
        """
        pred_data = self.write_prediction_serializer(prediction)
        cachefile = path.join(self.cache_root_dir, filename)
        self.json_cache.write_cache(cachefile, pred_data)

    def read_metric(self, filename: str) -> Union[TMetricResultCache, None]:
        """
        Reads metric data from the cache.

        Parameters
        ----------
        filename : str
            The name of the cache file containing the metric results.

        Returns
        -------
        Cached Metric result or None
        """

        cachefile = path.join(self.cache_root_dir, filename)
        cache_hit = self.json_cache.read_cache(cachefile)
        if cache_hit:
            return {
                key: torch.tensor(value) if isinstance(value, (int, float)) else value
                for key, value in cache_hit.items()
            }
        return cache_hit

    def write_metric(self, filename: str, metric_results: TMetricResultCache) -> None:
        """
        Writes metric results to the JSONcache.

        Parameters
        ----------
        filename : str
            The name of the cache file where the metric data will be written.

        metric_results : TMetricResultCache
            The metric data to be written to the cache.
        """

        cachefile = path.join(self.cache_root_dir, filename)
        self.json_cache.write_cache(cachefile, metric_results)

    def write_prediction_serializer(
        self,
        pred_and_data: tuple[
            Sequence[list[DetectionTarget]],
            Sequence[tuple[Any, Any, Any]],
        ],
    ) -> tuple[Sequence[list[dict[str, Any]]], Sequence[tuple[list[Any], list[Any], list[Any]]]]:
        """
        Serializes a prediction output to be compatible with JSON format.

        Parameters
        ----------
        pred_and_data : tuple
            A tuple containing two sequences:
            - A sequence of lists of `DetectionTarget` objects.
            - A sequence of tuples containing metadata information.

        Returns
        -------
        tuple
            A tuple containing two sequences:
            - A sequence of lists of dictionaries representing serialized `DetectionTarget` objects.
            - A sequence of tuples containing metadata, where the detection data is serialized.
        """

        preds = pred_and_data[0]
        data = pred_and_data[1]

        serial_pred = []
        for pred in preds:
            tmp = [{"boxes": dt.boxes, "labels": dt.labels, "scores": dt.scores} for dt in pred]
            serial_pred.append(tmp)

        serial_data = []
        for tpl in data:
            tpl_1 = [{"boxes": dt.boxes, "labels": dt.labels, "scores": dt.scores} for dt in tpl[1]]
            serial_data.append((tpl[0], tpl_1, tpl[2]))

        return (serial_pred, serial_data)

    def read_prediction_serializer(
        self,
        pred_data_cache: tuple[
            Sequence[list[dict[str, Any]]], Sequence[tuple[list[Any], list[dict[str, Any]], list[Any]]]
        ],
    ) -> tuple[
        Sequence[list[DetectionTarget]],
        Sequence[tuple[Any, Any, Any]],
    ]:
        """
        Restores prediction data from its serialized JSON representation.

        Parameters
        ----------
        pred_data_cache : tuple
            A tuple containing two sequences:
            - A sequence of lists of dictionaries representing serialized `DetectionTarget` objects.
            - A sequence of tuples containing metadata, where the detection data is serialized.

        Returns
        -------
        tuple
            A tuple containing two sequences:
            - A sequence of lists of `DetectionTarget` objects.
            - A sequence of tuples containing metadata information.
        """
        preds = pred_data_cache[0]
        data = pred_data_cache[1]

        detection_pred = []
        for pred in preds:
            tmp = [
                DetectionTarget(
                    boxes=torch.tensor(dt["boxes"]),
                    labels=torch.tensor(dt["labels"]),
                    scores=torch.tensor(dt["scores"]),
                )
                for dt in pred
            ]
            detection_pred.append(tmp)

        detection_data = []
        for tpl in data:
            tpl_0 = [torch.tensor(tens) for tens in tpl[0]]
            tpl_1 = [
                DetectionTarget(
                    boxes=torch.tensor(dt["boxes"]),
                    labels=torch.tensor(dt["labels"]),
                    scores=torch.tensor(dt["scores"]),
                )
                for dt in tpl[1]
            ]

            detection_data.append((tpl_0, tpl_1, tpl[2]))

        return (detection_pred, detection_data)

    def clear_cache(self) -> None:
        """
        Clears the cache directory by removing all JSON cache files.

        This method deletes all `.json` files in the cache directory to free up space.

        Raises
        ------
        OSError
            If the cache files cannot be deleted, an OSError is raised with a description of the failure.
        """
        json_files = glob.glob(path.join(self.cache_root_dir, "*.json"))
        # Instead of catching exceptions within the loop, check the file before attempting removal
        try:
            for json_file in json_files:
                remove(json_file)
        except OSError as e:
            raise OSError(f"Failed to clear cache: {e}") from e


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
