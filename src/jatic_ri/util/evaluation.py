"""Evaluation and Prediction tool for Test Stages"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Generic, TypeAlias, TypeVar, overload

from maite._internals.protocols.generic import DataLoader, Dataset, Metric, Model
from maite.errors import InvalidArgument
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


class EvaluationTool:
    """
    A class for evaluating machine learning models on datasets using specified metrics.

    The class handles model, dataset, and metrics, including caching mechanisms
    for predictions and metric results.
    """

    def __init__(self, ri_cache: Cache_Option[Any] = None) -> None:
        self.ri_cache = ri_cache

    def compute_metric(
        self,
        metric: Metric,
        filename: str,
        prediction: Sequence[SomeTargetBatchType],
        data: Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]],
    ) -> dict[str, Any]:
        """Compute a specified metric over predictions and batched data.

        Parameters
        ----------
        metric : Metric
            The metric object to be computed. This can be a custom metric
            function or an identifier for a predefined metric.
        filename : str
            A unique identifier for caching purposes (file name).
            Used to store or retrieve intermediate results.
        prediction : Sequence[SomeTargetBatchType]
            A sequence of prediction data, where each element is a batch of
            predictions.
        data : Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]]
            A sequence of tuples, where each tuple corresponds to a batch and
            contains:
            - Input batch
            - Target (ground truth) batch
            - Metadata batch

        Returns
        -------
        dict[str, Any]
            A dictionary where keys are metric identifiers or names, and
            values are the computed metric values for the predictions and data.
        """
        if self.ri_cache:
            cache = self.ri_cache.read_metric(filename)
            if cache is not None:
                return cache
        for i, pred in enumerate(prediction):
            metric.update(pred, data[i][1])
        metric_results = metric.compute()
        if self.ri_cache:
            self.ri_cache.write_metric(filename, metric_results)
        return metric_results

    @overload
    def predict(
        self,
        model: ic.Model,
        model_id: str,  # Remove after MAITE 0.7.1 upgrade
        dataset: ic.Dataset,
        dataset_id: str,  # Remove after MAITE 0.7.1 upgrade
        dataloader: TDataloader = None,
        batch_size: int = 1,  # Remove after MAITE 0.7.1 upgrade
        augmentation: None = None,  # To match MAITE signature and return appropriate error
        return_augmented_data: bool = False,
    ) -> tuple[
        Sequence[Sequence[ic.TargetType]],
        Sequence[tuple[Sequence[ic.InputType], Sequence[ic.TargetType], Sequence[ic.DatumMetadataType]]],
    ]: ...

    @overload
    def predict(
        self,
        model: od.Model,
        model_id: str,  # Remove after MAITE 0.7.1 upgrade
        dataset: od.Dataset,
        dataset_id: str,  # Remove after MAITE 0.7.1 upgrade
        dataloader: TDataloader = None,
        batch_size: int = 1,  # Remove after MAITE 0.7.1 upgrade
        augmentation: None = None,  # To match MAITE signature and return appropriate error
        return_augmented_data: bool = False,
    ) -> tuple[
        Sequence[Sequence[od.TargetType]],
        Sequence[tuple[Sequence[od.InputType], Sequence[od.TargetType], Sequence[od.DatumMetadataType]]],
    ]: ...

    def predict(
        self,
        model: Model,
        model_id: str,  # Remove after MAITE 0.7.1 upgrade
        dataset: Dataset[Any, Any, Any],
        dataset_id: str,  # Remove after MAITE 0.7.1 upgrade
        dataloader: TDataloader = None,
        batch_size: int = 1,  # Remove after MAITE 0.7.1 upgrade
        augmentation: None = None,  # To match MAITE signature and return appropriate error
        return_augmented_data: bool = False,
    ) -> tuple[
        Sequence[SomeTargetBatchType],
        Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]],
    ]:
        """Generate predictions using a model and dataset.

        This tool checks the cache with the provided ID before running predictions.
        If cached results are available (cache hit), it returns them.

        Note
        ----
        The "<object>_id" parameters are required because MAITE v0.6.0 does not
        mandate that types include ID metadata. MAITE v0.7.1 requires Model,
        Dataset, Metric, Datum, and Augmentation to have a metadata property
        with a required 'id' string. We will deprecate "<object>_id" in favor
        of `<object>.metadata.id` when upgrading MAITE.

        Parameters
        ----------
        model : Model
            The model to be used for generating predictions.
        model_id : str
            A unique identifier for the model.
        dataset : Dataset[Any, Any, Any]
            The dataset to generate predictions on.
        dataset_id : str
            A unique identifier for the dataset.
        dataloader : TDataloader, optional
            A dataloader that facilitates batch processing of the dataset.
            If None, a `SimpleDataLoader` is used.
        batch_size : int, default=1
            The batch size to be used for prediction.
        augmentation : None, optional
            NOT IMPLEMENTED. If provided, raises an `InvalidArgument` error.
        return_augmented_data : bool, default=False
            Set to True to return post-augmentation data as a function output.
            Note that caching this data requires significant memory.

        Returns
        -------
        tuple[
            Sequence[SomeTargetBatchType],
            Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]]
        ]
            A tuple containing two sequences:
            1. Predictions: A sequence of batches, where each batch contains
               the model's predictions.
            2. Data: A sequence of batches, where each batch is a tuple of:
               - Input data (empty if `return_augmented_data` is False)
               - Target (ground truth) data
               - Metadata

        Raises
        ------
        InvalidArgument
            If `augmentation` is not None, as this feature is not implemented.
        """
        if augmentation is not None:
            raise InvalidArgument(
                "EvaluationTool has not implemented MAITE's augumentation capability. "
                "Import maite.tasks.evaluate to use augmentations without a caching mechanism."
            )

        cache_file = f"{model_id}_{dataset_id}_{batch_size}.json"
        if self.ri_cache:
            cache = self.ri_cache.read_predictions(cache_file)
            if cache is not None:
                return cache

        if dataloader is None:
            dataloader = SimpleDataLoader(dataset=dataset, batch_size=batch_size)

        preds_batches = []
        data_batches = []

        for input_datum_batch, target_datum_batch, metadata_batch in dataloader:
            preds_batch = model(input_datum_batch)
            preds_batches.append(preds_batch)
            data_batches.append(
                (input_datum_batch if return_augmented_data else [], target_datum_batch, metadata_batch)
            )

        results = (preds_batches, data_batches)
        if self.ri_cache:
            self.ri_cache.write_predictions(cache_file, results)
        return results

    @overload
    def evaluate(
        self,
        model: ic.Model,
        model_id: str,  # Remove after MAITE 0.7.1 upgrade
        dataset: ic.Dataset,
        dataset_id: str,  # Remove after MAITE 0.7.1 upgrade
        metric: ic.Metric,
        metric_id: str,  # Remove after MAITE 0.7.1 upgrade
        batch_size: int = 1,
        dataloader: TDataloader = None,
        return_augmented_data: bool = False,
        return_preds: bool = False,
        augmentation: None = None,  # To match MAITE signature and return appropriate error
    ) -> tuple[
        dict[str, Any],
        Sequence[Sequence[ic.TargetType]],
        Sequence[tuple[Sequence[ic.InputType], Sequence[ic.TargetType], Sequence[ic.DatumMetadataType]]],
    ]: ...

    @overload
    def evaluate(
        self,
        model: od.Model,
        model_id: str,  # Remove after MAITE 0.7.1 upgrade
        dataset: od.Dataset,
        dataset_id: str,  # Remove after MAITE 0.7.1 upgrade
        metric: od.Metric,
        metric_id: str,  # Remove after MAITE 0.7.1 upgrade
        batch_size: int = 1,
        dataloader: TDataloader = None,
        return_augmented_data: bool = False,
        return_preds: bool = False,
        augmentation: None = None,  # To match MAITE signature and return appropriate error
    ) -> tuple[
        dict[str, Any],
        Sequence[Sequence[od.TargetType]],
        Sequence[tuple[Sequence[od.InputType], Sequence[od.TargetType], Sequence[od.DatumMetadataType]]],
    ]: ...

    def evaluate(
        self,
        model: Model,
        model_id: str,  # Remove after MAITE 0.7.1 upgrade
        dataset: Dataset[Any, Any, Any],
        dataset_id: str,  # Remove after MAITE 0.7.1 upgrade
        metric: Metric,
        metric_id: str,  # Remove after MAITE 0.7.1 upgrade
        batch_size: int = 1,
        dataloader: TDataloader = None,
        return_augmented_data: bool = False,
        return_preds: bool = False,
        augmentation: None = None,  # To match MAITE signature and return appropriate error
    ) -> tuple[
        dict[str, Any],
        Sequence[SomeTargetBatchType] | None,
        Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]] | None,
    ]:
        """Evaluate the model on a dataset using a specified metric.

        This function checks for the availability of the model, dataset, and
        metric, and performs the evaluation to compute the metric values.
        It utilizes caching for both predictions and metric results.

        Note
        ----
        The "<object>_id" parameters are required because MAITE v0.6.0 does not
        mandate that types include ID metadata. MAITE v0.7.1 requires Model,
        Dataset, Metric, Datum, and Augmentation to have a metadata property
        with a required 'id' string. We will deprecate "<object>_id" in favor
        of `<object>.metadata.id` when upgrading MAITE.

        Parameters
        ----------
        model : Model
            The model to be evaluated.
        model_id : str
            A unique identifier for the model.
        dataset : Dataset[Any, Any, Any]
            The dataset to evaluate the model on.
        dataset_id : str
            A unique identifier for the dataset.
        metric : Metric
            The metric to evaluate the model's performance.
        metric_id : str
            A unique identifier for the metric.
        batch_size : int, default=1
            The batch size for evaluation.
        dataloader : TDataloader, optional
            A dataloader for batching and loading the dataset. If None,
            a `SimpleDataLoader` is used.
        return_augmented_data : bool, default=False
            If True, include batches of data (from the dataset) in the
            return tuple. "Augmented" is a misnomer here as augmentations
            are not supported, but it matches the MAITE `evaluate` signature.
        return_preds : bool, default=False
            If True, include predictions in the return tuple.
        augmentation : None, optional
            NOT IMPLEMENTED. If provided, raises an `InvalidArgument` error.

        Returns
        -------
        tuple[
            dict[str, Any],
            Sequence[SomeTargetBatchType] | None,
            Sequence[tuple[SomeInputBatchType,SomeTargetBatchType,SomeMetadataBatchType]] | None
        ]
            A 3-tuple containing:
            1. Metric results: A dictionary where keys are metric identifiers
               and values are the computed metric values.
            2. Predictions: A sequence of prediction batches if `return_preds`
               is True, else None.
            3. Data: A sequence of data batches if `return_augmented_data`
               is True, else None. Each data batch is a tuple of:
               - Input data (empty if `return_augmented_data` is False and
                 not returned by `predict`)
               - Target (ground truth) data
               - Metadata

        Raises
        ------
        InvalidArgument
            If `augmentation` is not None, as this feature is not implemented.
        """
        if augmentation is not None:
            raise InvalidArgument(
                "EvaluationTool has not implemented MAITE's augumentation capability. "
                "Import maite.tasks.evaluate to use augmentations without a caching mechanism."
            )

        cache_file_metric = f"{model_id}_{dataset_id}_{metric_id}_{batch_size}.json"

        if dataloader is None:
            dataloader = SimpleDataLoader(dataset=dataset, batch_size=batch_size)

        prediction, data = self.predict(
            model=model,
            model_id=model_id,
            dataset=dataset,
            dataset_id=dataset_id,
            dataloader=dataloader,
            batch_size=batch_size,
            return_augmented_data=return_augmented_data,
        )

        metric_results = self.compute_metric(
            metric=metric, filename=cache_file_metric, prediction=prediction, data=data
        )

        return (metric_results, prediction if return_preds else None, data if return_augmented_data else None)
