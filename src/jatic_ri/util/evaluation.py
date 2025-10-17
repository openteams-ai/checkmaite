"""Evaluation and Prediction tool for Test Stages"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Generic, TypeAlias, TypeVar

from maite._internals.protocols.generic import DataLoader, Dataset
from maite.protocols import ArrayLike
from maite.protocols import image_classification as ic
from maite.protocols import object_detection as od

SomeInputBatchType: TypeAlias = Sequence[ic.InputType] | Sequence[od.InputType]
SomeTargetBatchType: TypeAlias = Sequence[ic.TargetType] | Sequence[od.TargetType]
SomeMetadataBatchType: TypeAlias = Sequence[ic.DatumMetadataType] | Sequence[od.DatumMetadataType]
SomeInputType: TypeAlias = ic.InputType | od.InputType
SomeTargetType: TypeAlias = ic.TargetType | od.TargetType
SomeMetadataType: TypeAlias = ic.DatumMetadataType | od.DatumMetadataType

TDataloader: TypeAlias = DataLoader[Any, Any, Any] | None

TInput = TypeVar("TInput", bound=SomeInputType)
TTarget = TypeVar("TTarget", bound=SomeTargetType)
TMetadata = TypeVar("TMetadata", bound=SomeMetadataType)
TDataset = TypeVar("TDataset", bound=Dataset)


TData = TypeVar("TData", dict, list)

# RI conventions state each value must (1) be safely cast to a float, and (2) possess <value>.numpy() method
# We have recommended future MAITE release defines a protocol for these criteria
TMetricResult = dict[str, Any]

# The data structure generally passed around by MAITE's predict tools.  This is a simplification of the type
# system built-out in maite._internals.protocols that can be applied to type hints for only this use case.
CacheablePredsAndData: TypeAlias = tuple[  # One tuple containing...
    Sequence[  # first, Sequences of batches where...
        Sequence[TTarget]
    ],  # each batch is a Sequence of predictions, and...
    Sequence[  # second, Sequences of batches where...
        tuple[  # each batch is a "data tuple" containing corresponding three more sequences...
            Sequence[ArrayLike],  # (1) Inputs: images in ArrayLike shape (C, H, W),
            Sequence[TTarget],  # (2) Targets: ground truths, and
            Sequence[dict[str, Any]],  # (3) datum-wise Metadata.
        ]
    ],
]


class RICache(ABC):
    """Abstract Class for using cache for evaluation and prediction."""

    @abstractmethod
    def read_predictions(self, filename: str) -> CacheablePredsAndData | None:
        """Reads a prediction from the cache.

        Parameters
        ----------
        filename : str
            The name of the file to read the prediction from.

        Returns
        -------
        CacheablePredsAndData | None
            The cached prediction data, or None if not found.
        """
        pass

    @abstractmethod
    def write_predictions(
        self,
        filename: str,
        prediction: CacheablePredsAndData,
    ) -> None:
        """Writes a prediction to the cache.

        Parameters
        ----------
        filename : str
            The name of the file to write the prediction to.
        prediction : CacheablePredsAndData
            The prediction data to cache.
        """
        pass

    @abstractmethod
    def read_metric(self, filename: str) -> TMetricResult | None:
        """Reads a metric from the cache.

        Parameters
        ----------
        filename : str
            The name of the file to read the metric from.

        Returns
        -------
        TMetricResult | None
            The cached metric result, or None if not found.
        """
        pass

    @abstractmethod
    def write_metric(self, filename: str, metric_results: TMetricResult) -> None:
        """Writes a metric to the cache.

        Parameters
        ----------
        filename : str
            The name of the file to write the metric to.
        metric_results : TMetricResult
            The metric results to cache.
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clears the cache dir."""
        pass


TRICache = TypeVar("TRICache", bound="RICache")
Cache_Option: TypeAlias = TRICache | None


class SimpleDataLoader(Generic[TInput, TTarget, TMetadata]):
    """
    A simple, deterministic data loader that batches data from a given dataset.

    This data loader takes a dataset and splits it into batches of a given size.
    It ensures deterministic batching, meaning the order and size of batches
    are consistent.
    """

    def __init__(self, dataset: Dataset[TInput, TTarget, TMetadata], batch_size: int) -> None:
        """Initialize the data loader.

        Parameters
        ----------
        dataset : Dataset[TInput, TTarget, TMetadata]
            The dataset from which data will be loaded, containing input,
            target, and metadata.
        batch_size : int
            The number of examples per batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[Sequence[TInput], Sequence[TTarget], Sequence[TMetadata]]]:
        """Iterate over the dataset in batches.

        This method divides the dataset into batches of size `batch_size`
        (or smaller for the last batch), collates each batch, and yields
        a tuple containing:
        - input data batch (Sequence[TInput])
        - target data batch (Sequence[TTarget])
        - metadata batch (Sequence[TMetadata])

        Yields
        ------
        tuple[Sequence[TInput], Sequence[TTarget], Sequence[TMetadata]]
            A tuple containing:
            - input_batch: A batch of input data.
            - target_batch: A batch of target data.
            - metadata_batch: A batch of metadata.
        """
        total_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        for batch_no in range(total_batches):
            batch_data_as_singles = [
                self.dataset[i]
                for i in range(batch_no * self.batch_size, (batch_no + 1) * self.batch_size)
                if i < len(self.dataset)
            ]

            batch_inputs, batch_targets, batch_metadata = self._collate(batch_data_as_singles)

            yield (batch_inputs, batch_targets, batch_metadata)

    def _collate(
        self,
        batch_data_as_singles: Iterable[tuple[TInput, TTarget, TMetadata]],
    ) -> tuple[Sequence[TInput], Sequence[TTarget], Sequence[TMetadata]]:
        """Collate a batch of data.

        Collates data from an iterable of individual data points into three
        separate batches: input data, target data, and metadata.

        Parameters
        ----------
        batch_data_as_singles : Iterable[tuple[TInput, TTarget, TMetadata]]
            An iterable of tuples, where each tuple contains an individual
            data point in the format (input_datum, target_datum, metadata_datum).
            Each datum is of type TInput, TTarget, and TMetadata, respectively.

        Returns
        -------
        tuple[Sequence[TInput], Sequence[TTarget], Sequence[TMetadata]]
            A tuple containing three sequences:
            - input_batch: A batch of input data.
            - target_batch: A batch of target data.
            - metadata_batch: A batch of metadata associated with the data.
        """
        input_batch: list[TInput] = []
        target_batch: list[TTarget] = []
        metadata_batch: list[TMetadata] = []
        for input_datum, target_datum, metadata_datum in batch_data_as_singles:
            input_batch.append(input_datum)
            target_batch.append(target_datum)
            metadata_batch.append(metadata_datum)
        return input_batch, target_batch, metadata_batch
