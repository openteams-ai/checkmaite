"""Caching utility class for Test Stages"""

import glob
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from os import getcwd, makedirs, path, remove
from typing import Any, Optional, TypeAlias, TypeVar, Union

import numpy as np
import torch
import zstandard as zstd
from maite.protocols import ArrayLike

from jatic_ri._common.test_stages.interfaces.test_stage import Cache
from jatic_ri.object_detection.datasets import DetectionTarget

logger = logging.getLogger()

TData = TypeVar("TData", dict, list)

# RI conventions state each value must (1) be safely cast to a float, and (2) possess <value>.numpy() method
# We have recommended future MAITE release defines a protocol for these criteria
TMetricResult = dict[str, Any]

# Supported target types
# For IC, the ArrayLike is typically a (Cl,) vector, one-hot encoded for multi-class, single-label ground truth.
# For OD, the structure of DetectionTarget boxes, labels, and scores is documented elsewhere.
TTargetType = TypeVar("TTargetType", bound=Union[ArrayLike, DetectionTarget])

# The data structure generally passed around by MAITE's predict tools.  This is a simplification of the type
# system built-out in maite._internals.protocols that can be applied to type hints for only this use case.
CacheablePredsAndData: TypeAlias = tuple[  # One tuple containing...
    Sequence[  # first, Sequences of batches where...
        Sequence[TTargetType]
    ],  # each batch is a Sequence of predictions, and...
    Sequence[  # second, Sequences of batches where...
        tuple[  # each batch is a "data tuple" containing corresponding three more sequences...
            Sequence[ArrayLike],  # (1) Inputs: images in ArrayLike shape (C, H, W),
            Sequence[TTargetType],  # (2) Targets: ground truths, and
            Sequence[dict[str, Any]],  # (3) datum-wise Metadata.
        ]
    ],
]


class JSONCache(Cache[TData]):
    """Basic JSON file based caching plugin"""

    def __init__(self, encoder: type = json.JSONEncoder, compress: bool = False) -> None:
        self.encoder = encoder
        self.compress = compress

    def read_cache(self, cache_path: str) -> Optional[TData]:
        """Read cache from file and returns as dictionary"""
        logger.info(f"Checking for existing cache at {cache_path}")
        if path.exists(cache_path):
            with open(cache_path, "rb") as file:
                raw = file.read()
            contents = zstd.decompress(raw).decode("utf-8") if self.compress else raw.decode("utf-8")
            return json.loads(contents)
        return None

    def write_cache(self, cache_path: str, data: TData) -> None:
        """Writes dictionary to file using serializer"""
        logger.info(f"Writing cache to {cache_path}")
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

    def default(self, o: Any) -> Any:
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

    def default(self, o: Any) -> Any:
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
    def read_predictions(self, filename: str) -> Optional[CacheablePredsAndData]:
        """Reads a prediction from the cache"""
        pass

    @abstractmethod
    def write_predictions(
        self,
        filename: str,
        prediction: CacheablePredsAndData,
    ) -> None:
        """Writes a prediction to the cache"""
        pass

    @abstractmethod
    def read_metric(self, filename: str) -> Optional[TMetricResult]:
        """Reads a metric from the cache"""
        pass

    @abstractmethod
    def write_metric(self, filename: str, metric_results: TMetricResult) -> None:
        """Writes a metric to the cache"""
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clears the cache dir"""
        pass


class SimpleRICacheJSON(RICache):
    """
    An abstract wrapper for the RICache class that sets up a cache directory and provides methods for reading
    and writing prediction and metric data to/from the cache using JSON format.

    Each task type (e.g. Image Classification, Object Detection) structures predictions/targets differently and
    therefore needs a custom implementation of the JSON serializer.  So in the base JSON serializer class, the
    methods for read_prediction_serializer and write_prediction_serializer are still abstract.

    JSON represents data in 'arrays' and 'objects', and from this representation the type(s) of the source Python
    objects cannot necessarily be inferred.  For example, `SimpleRICacheOD` and `SimpleRICacheIC` can accept and
    serialize any ArrayLike, but has chosen to deserialize as torch.Tensor.

    Parameters
    ----------
    cache_root_dir:
        this dictates the root path where all of the folders for the cache files will be created.
    compress_json : bool = True
        Determines whether the JSON stored on disk will be compressed (and therefore, not human-readable).
    """

    def __init__(self, cache_root_dir: str = "", compress_json: bool = True) -> None:
        """
        Initializes the SimpleRICacheJSON instance, setting up the cache directory and the JSON cache.

        Parameters
        ----------
        cache_root_dir:
            The directory where the cache files will be stored. If not provided, the current working
            directory will be used. Default is "" (empty string).
        """
        if cache_root_dir is None or cache_root_dir == "":
            cache_root_dir = getcwd()
        self.json_cache = JSONCache(encoder=TensorEncoder, compress=compress_json)
        self.cache_root_dir = cache_root_dir
        makedirs(path.dirname(cache_root_dir), exist_ok=True)

    def read_predictions(self, filename: str) -> Optional[CacheablePredsAndData]:
        """
        Reads prediction data from the cache using JSONCache.

        Parameters
        ----------
        filename : str
            The name of the cache file containing the prediction data.

        Returns
        -------
        CacheablePredsAndData or None
            A 2-tuple of batches of predictions and targets, or
            None if the cache read is unsuccessful.
        """
        cachefile = path.join(self.cache_root_dir, filename)
        cache_hit = self.json_cache.read_cache(cachefile)
        if cache_hit:
            return self.read_prediction_serializer(cache_hit)
        return cache_hit

    def write_predictions(
        self,
        filename: str,
        prediction: CacheablePredsAndData,
    ) -> None:
        """
        Writes prediction data to the cache using JSONCache.

        Parameters
        ----------
        filename : str
            The name of the cache file where the prediction data will be written.

        prediction : CacheablePredsAndData
            A 2-tuple of batches of predictions and targets.
        """
        pred_data = self.write_prediction_serializer(prediction)
        cachefile = path.join(self.cache_root_dir, filename)
        self.json_cache.write_cache(cachefile, pred_data)

    def read_metric(self, filename: str) -> Optional[TMetricResult]:
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

    def write_metric(self, filename: str, metric_results: TMetricResult) -> None:
        """
        Writes metric results to the JSONcache.

        Parameters
        ----------
        filename : str
            The name of the cache file where the metric data will be written.

        metric_results : TMetricResult
            The metric data to be written to the cache.
        """

        cachefile = path.join(self.cache_root_dir, filename)
        self.json_cache.write_cache(cachefile, metric_results)

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

    @abstractmethod
    def write_prediction_serializer(self, pred_and_data: CacheablePredsAndData) -> Union[TData, CacheablePredsAndData]:
        """Serializes a prediction output to be compatible with JSON format."""
        pass

    @abstractmethod
    def read_prediction_serializer(self, pred_data_cache: TData) -> CacheablePredsAndData:
        """Restores prediction data from its serialized JSON representation."""
        pass


class SimpleRICacheIC(SimpleRICacheJSON):
    """
    An implementation of the serialization functions to handle IC data structures.

    Given that we cannot infer the original subtype of a JSON serializtion of an 'ArrayLike', we have chosen
    to reconsitutue the Python objects as torch.Tensor.  If this is not compatible with an implementer's use case,
    that user will need to use/create a different `RICache` implementation.
    """

    def write_prediction_serializer(self, pred_and_data: CacheablePredsAndData) -> CacheablePredsAndData:
        """
        Serializes a prediction output to be compatible with JSON format.

        The IC data structure is already JSON-compatible, so this is just a passthrough.

        Parameters
        ----------
        pred_data_cache : CacheablePredsAndData

        Returns
        -------
        CacheablePredsAndData
        """
        return pred_and_data

    def read_prediction_serializer(
        self,
        pred_data_cache: TData,  # A list of lists
    ) -> CacheablePredsAndData:
        """
        Restores prediction data from its serialized JSON representation.

        Parameters
        ----------
        pred_data_cache : the response of json.loads(), typically a list/dict combination
            A list containing two items:
            - Lists of lists (the first list is batching) of lists representing serialized predictions.
            - Lists (batches) of 3-items lists of Image lists, list Targets, and metadata dictionaries.

        Returns
        -------
        CacheablePredsAndData, which is a tuple containing two items:
            - Batches (sequences) of sequences of predictions, and
            - Batches (sequences) of a "data" tuple, itself containing three equal length sequences:
              1) Inputs: images in Tensor (an ArrayLike) of shape (C, H, W)
              2) Targets: ground truths
              3) Metadata: dicts of [str,Any]
        """
        return (  # a 2-tuple of preds-data
            # Reconstruct predictions with double list comprehension... a list of batches...
            [  # Where each batch is a list...
                [
                    torch.as_tensor(j) for j in pred_data_cache[0][i]
                ]  # And each item must be ArrayLike so we cast back from list[float] to Tensor
                for i in range(len(pred_data_cache[0]))
            ],
            # Reconstruct list of data tuple (image_batches, target_batches, metadata_batches)
            [
                (
                    [torch.as_tensor(j) for j in pred_data_cache[1][i][0]],  # Convert image lists back to tensor
                    [
                        torch.as_tensor(j) for j in pred_data_cache[1][i][1]
                    ],  # Convert ground truth target back to tensor
                    pred_data_cache[1][i][2],  # The metadata should already come back as dicts
                )
                for i in range(len(pred_data_cache[1]))
            ],
        )


class SimpleRICacheOD(SimpleRICacheJSON):
    """
    An implementation of the serialization functions to handle OD DetectionTargets.

    Given that we cannot infer the original subtype of a JSON serializtion of an 'ArrayLike', we have chosen
    to reconsitutue the Python objects as torch.Tensor.  If this is not compatible with an implementer's use case,
    that user will need to use/create a different `RICache` implementation.
    """

    def write_prediction_serializer(
        self,
        pred_and_data: CacheablePredsAndData,
    ) -> tuple[list[list[dict[str, Any]]], list[tuple[list[Any], list[Any], list[Any]]]]:
        """
        Serializes a prediction output to be compatible with JSON format.

        Parameters
        ----------
        pred_and_data :
            CacheablePredsAndData, which is a tuple containing two items:
            - Batches (sequences) of sequences of predictions, and
            - Batches (sequences) of a "data" tuple, itself containing three equal length sequences:
              1) Inputs: images in ArrayLike shape (C, H, W)
              2) Targets: ground truths
              3) Metadata: dicts of [str,Any]
        Returns
        -------
        tuple :
            A tuple containing two lists:
            - One batch list, where each item is a list of OD DetectionTargets represented as dicts
            - Another batch list, where each item is a tuple containing three lists:
              1) Inputs: Still in ArrayLike
              2) Targets: converted to dict representations
              3) Metadata: dicts of [str,Any]
        """

        preds: Sequence[Sequence[DetectionTarget]] = pred_and_data[0]
        data: Sequence[tuple[Sequence[ArrayLike], Sequence[DetectionTarget], Sequence[dict[str, Any]]]] = pred_and_data[
            1
        ]

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
        pred_data_cache: TData,
    ) -> CacheablePredsAndData:
        """
        Restores prediction data from its serialized JSON representation.

        Parameters
        ----------
        pred_data_cache : the response of json.loads(), typically a list/dict combination
            A list containing two items:
            - Lists (batches) of lists of dictionaries representing serialized `DetectionTarget` predictions.
            - Lists (batches) of 3-items lists of Image lists, DetectionTarget dictionaries, and metadata dictionaries

        Returns
        -------
        CacheablePredsAndData, which is a tuple containing two items:
            - Batches (sequences) of sequences of predictions, and
            - Batches (sequences) of a "data" tuple, itself containing three equal length sequences:
              1) Inputs: images in Tensor (an ArrayLike) of shape (C, H, W)
              2) Targets: ground truths
              3) Metadata: dicts of [str,Any]
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
