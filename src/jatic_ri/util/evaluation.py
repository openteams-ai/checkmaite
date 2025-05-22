"""Evaluation and Prediction tool for Test Stages"""

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, overload

from maite._internals.protocols.generic import DataLoader, Dataset, Metric, Model
from maite.errors import InvalidArgument
from maite.protocols import image_classification as ic
from maite.protocols import object_detection as od

if TYPE_CHECKING:
    from jatic_ri.util.cache import RICache

SomeInputBatchType: TypeAlias = ic.InputBatchType | od.InputBatchType
SomeTargetBatchType: TypeAlias = ic.TargetBatchType | od.TargetBatchType
SomeMetadataBatchType: TypeAlias = ic.DatumMetadataBatchType | od.DatumMetadataBatchType
SomeInputType: TypeAlias = ic.InputType | od.InputType
SomeTargetType: TypeAlias = ic.TargetType | od.TargetType
SomeMetadataType: TypeAlias = ic.DatumMetadataType | od.DatumMetadataType

TDataloader: TypeAlias = DataLoader[Any, Any, Any] | None

TInput = TypeVar("TInput", bound=SomeInputType)
TTarget = TypeVar("TTarget", bound=SomeTargetType)
TMetadata = TypeVar("TMetadata", bound=SomeMetadataType)
TDataset = TypeVar("TDataset", bound=Dataset)

TRICache = TypeVar("TRICache", bound="RICache")
Cache_Option: TypeAlias = TRICache | None


class SimpleDataLoader(Generic[TInput, TTarget, TMetadata]):
    """
    A simple, deterministic data loader that batches data from a given dataset.

    This data loader takes a dataset and splits it into batches of a given size. It ensures deterministic
    batching, meaning the order and size of batches are consistent.
    """

    def __init__(self, dataset: Dataset[TInput, TTarget, TMetadata], batch_size: int) -> None:
        """
        Initializes the data loader with the provided dataset and batch size.

        Parameters
        ----------
        dataset : Dataset[TInput, TTarget, TMetadata]
            The dataset from which data will be loaded, containing input, target, and metadata.

        batch_size : int
            The number of examples per batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[Sequence[TInput], Sequence[TTarget], Sequence[TMetadata]]]:
        """
        Iterates over the dataset in batches, collates the data into input, target, and metadata batches,
        and yields them.

        This method divides the dataset into batches of size `batch_size` (or smaller for the last batch),
        collates each batch, and yields a tuple containing:
        - input data batch (Sequence[TInput]),
        - target data batch (Sequence[TTarget]),
        - metadata batch (Sequence[TMetadata]).

        Returns
        -------
        iterator of tuple
            An iterator that yields batches, where each batch is a tuple containing:
            - input_batch (Sequence[TInput]): A batch of input data.
            - target_batch (Sequence[TTarget]): A batch of target data.
            - metadata_batch (Sequence[TMetadata]): A batch of metadata.
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
        """
        Collates a batch of data from an iterable of individual data points into three separate batches:
        input data, target data, and metadata.

        Parameters
        ----------
        batch_data_as_singles : iterable of tuple
            An iterable of tuples, where each tuple contains an individual data point in the format
            (input_datum, target_datum, metadata_datum). Each datum is of type TInput, TTarget, and
            TMetadata, respectively.

        Returns
        -------
        tuple of sequences
            A tuple containing three sequences:
            - input_batch (Sequence[TInput]): A batch of input data.
            - target_batch (Sequence[TTarget]): A batch of target data.
            - metadata_batch (Sequence[TMetadata]): A batch of metadata associated with the data.
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

    Methods
    -------
    compute_metric
        Processes a metric over predictions and batched data, using caching if available.

    predict
        Generates predictions using a specified model and dataset,
        checking for cached results before computing predictions.

    evaluate
        Evaluates a model on a dataset using a specified metric.
        Utilizes compute_prediction and compute_metric and their caching mechanisms.
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
        """
        Compute a specified metric over predictions and batched data.

        Parameters
        ----------
        metric : Metric
            The metric object to be computed. This can be a custom metric function or an identifier
            for a predefined metric.
        cache_id : str
            A unique identifier for caching purposes. in this case a file name.
            Used to store or retrieve intermediate results.
        prediction : sequence of list of dicts
            A sequence of prediction data, where each element is a list of dictionaries representing
            predicted values.
        data : sequence of tuples
            A sequence of tuples containing three elements:
                - A list of ground truth values.
                - A list of any additional values associated with each data point.
                - A list containing metadata.

        Returns
        -------
        result : dict
            A dictionary where keys are metric identifiers or names, and values are the computed
            metric values for the predictions and data.
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
        Sequence[ic.TargetBatchType], Sequence[tuple[ic.InputBatchType, ic.TargetBatchType, ic.DatumMetadataBatchType]]
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
        Sequence[od.TargetBatchType], Sequence[tuple[od.InputBatchType, od.TargetBatchType, od.DatumMetadataBatchType]]
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
        Sequence[SomeTargetBatchType], Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]]
    ]:
        """
        Prediction tool that checks cache with the provided ID before running evaluation.
        If cache is available (hit), returns the cached results.

        Note the "<object>_id" parameters are required because MAITE v0.6.0 does not require the types to contain
        id metadata.  MAITE v0.7.1 requires Model, Dataset, Metric, Datum, and Augumentation to include a metadata
        property which has a required string 'id'.  We will deprecate "<object>_id" in favor of <object>.metadata.id
        when upgrading MAITE.


        Parameters
        ----------
        model : Model
            The model to be used for generating predictions.
        model_id : str
            A unique identifier for the model.
        dataset : Dataset
            The dataset to generate predictions on.
        dataset_id : str
            A unique identifier for the dataset.
        dataloader (Optional) : TDataloader
            A dataloader that facilitates batch processing of the dataset.
        batch_size : int
            The batch size to be used for prediction. Default is 1, meaning predictions will be
            generated one sample at a time.
        augmentation : None = None
            NOT IMPLEMENTED: only raise appropriate error if called.
        return_augmented_data : bool = False
            Set to True to return post-augmentation data as a function output. Note that caching the data requires a lot
            of memory.

        Returns
        -------
        predictions : tuple
            A tuple containing two sequences:
            - The first sequence is a list of dictionaries representing predicted values for each data point.
            - The second sequence contains tuples with three elements:
              1) A list of inputs (e.g. images). This will be empty unless return_augmented_data is set
              2) A list of targets (ground truth) for each input
              3) A list of metadata
        """
        if augmentation is not None:
            raise InvalidArgument(
                "EvaluationTool has not implemented MAITE's augumentation capability. "
                "Import maite.workflows.evaluate to use augmentations without a caching mechanism."
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
        Sequence[ic.TargetBatchType],
        Sequence[tuple[ic.InputBatchType, ic.TargetBatchType, ic.DatumMetadataBatchType]],
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
        Sequence[od.TargetBatchType],
        Sequence[tuple[od.InputBatchType, od.TargetBatchType, od.DatumMetadataBatchType]],
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
        """
        Evaluates the model on the given dataset using the specified metric.
        The function checks for the availability of the model, dataset, and metric,
        and performs the evaluation to compute the metric values.

        Note the "<object>_id" parameters are required because MAITE v0.6.0 does not require the types to contain
        id metadata.  MAITE v0.7.1 requires Model, Dataset, Metric, Datum, and Augumentation to include a metadata
        property which has a required string 'id'.  We will deprecate "<object>_id" in favor of <object>.metadata.id
        when upgrading MAITE.

        Parameters
        ----------
        model : Model
            The model to be evaluated.
        model_id : str
            A unique identifier for the model.
        dataset : Dataset
            The dataset to evaluate the model on.
        dataset_id : str
            A unique identifier for the dataset, used for managing or caching the dataset.
        dataloader (Optional): TDataloader
            A dataloader used to batch and load the dataset, enabling efficient evaluation of the model.
        metric : Metric
            The metric function or identifier used to evaluate the model's performance on the dataset.
        metric_id : str
            A unique identifier for the metric, used for managing or caching the metric computation.
        batch_size : int, optional
            The batch size to be used for evaluation. Default is 1, meaning the evaluation is performed one
            sample at a time.
        return_predictions : bool = False
            Set to True to include predictions in second element of return tuple
        return_augmented_data : bool = False
            Set to True to include the batches of data (extracted from the dataset) in the return tuple.
            Note that "augmented" is a misnomer since EvaluationTool does not support 'augmentation' inputs, but
            this matches the MAITE evaluate signature.
        augmentation : None = None
            NOT IMPLEMENTED: only raise appropriate error if called.

        Returns
        -------
        result : tuple
            A 3-tuple containing:
            - A dictionary where keys are metric identifiers and values are the computed metric values.
            - A Sequence (batches) of list of predictions IF return_preds=true, else None
            - A Sequence (batches) of tuples IF return_augmented_data=true, else None, containing:
                - A list of data (e.g. images). This will be empty unless return_augmented_data is set.
                - A list of targets (ground truth).
                - A list of metadata.
        """
        if augmentation is not None:
            raise InvalidArgument(
                "EvaluationTool has not implemented MAITE's augumentation capability. "
                "Import maite.workflows.evaluate to use augmentations without a caching mechanism."
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
