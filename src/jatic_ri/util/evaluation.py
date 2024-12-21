"""Evaluation and Prediction tool for Test Stages"""

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Generic, TypeVar, Union

from maite._internals.protocols import image_classification as ic
from maite._internals.protocols import object_detection as od
from maite._internals.protocols.generic import DataLoader, Dataset, Metric, Model
from maite._internals.utils import add_progress_bar
from typing_extensions import TypeAlias

from jatic_ri.util.cache import RICache

SomeInputBatchType: TypeAlias = Union[ic.InputBatchType, od.InputBatchType]
SomeTargetBatchType: TypeAlias = Union[ic.TargetBatchType, od.TargetBatchType]
SomeMetadataBatchType: TypeAlias = Union[ic.DatumMetadataBatchType, od.DatumMetadataBatchType]
SomeInputType: TypeAlias = Union[ic.InputType, od.InputType]
SomeTargetType: TypeAlias = Union[ic.TargetType, od.TargetType]
SomeMetadataType: TypeAlias = Union[ic.DatumMetadataType, od.DatumMetadataType]

TDataloader: TypeAlias = Union[DataLoader[Any, Any, Any], None]

TInput = TypeVar("TInput", bound=SomeInputType)
TTarget = TypeVar("TTarget", bound=SomeTargetType)
TMetadata = TypeVar("TMetadata", bound=SomeMetadataType)
TDataset = TypeVar("TDataset", bound=Dataset)
TMetric = TypeVar("TMetric", bound=Metric)
TModel = TypeVar("TModel", bound=Model)

TRICache = TypeVar("TRICache", bound=RICache)
Cache_Option: TypeAlias = Union[TRICache, None]


class SimpleDataLoader(Generic[TInput, TTarget, TMetadata]):
    """Simple dataloader from maite in case none is provided. Deterministic dataloader."""

    def __init__(self, dataset: Dataset[TInput, TTarget, TMetadata], batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[Sequence[TInput], Sequence[TTarget], Sequence[TMetadata]]]:
        """Iterate over first batch_size examples from dataset, collate them, and yield result."""
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
        Describe how to create a tuple of
        (InputBatchType, TargetBatchType, DatumMetadataBatchType)
        from an iterator of tuples of
        (InputType, TargetType, DatumMetadataType)
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
    """Class for Finding Cache from Maite objects."""

    def __init__(self, ri_cache: Cache_Option[Any] = None) -> None:
        self.ri_cache = ri_cache

    def compute_predictions(
        self,
        model: TModel,
        model_id: str,
        dataset: Dataset[Any, Any, Any],
        dataset_id: str,
        dataloader: TDataloader,
        batch_size: int = 1,
    ) -> tuple[
        Sequence[SomeTargetBatchType],
        Sequence[tuple[SomeInputBatchType, SomeTargetBatchType, SomeMetadataBatchType]],
    ]:
        """A prediction tool that checks cache with the ID's before running evaluate. If cache hits, return cache."""
        cache_file = f"{model_id}_{dataset_id}_{batch_size}.json"
        if self.ri_cache:
            cache = self.ri_cache.read_predictions(cache_file)
            if cache is not None:
                return cache

        if dataloader is None:
            dataloader = SimpleDataLoader(dataset=dataset, batch_size=batch_size)

        preds_batches = []
        data_batches = []

        for input_datum_batch, target_datum_batch, metadata_batch in add_progress_bar(dataloader):
            preds_batch = model(input_datum_batch)
            preds_batches.append(preds_batch)
            data_batches.append((input_datum_batch, target_datum_batch, metadata_batch))

        results = (preds_batches, data_batches)
        if self.ri_cache:
            self.ri_cache.write_predictions(cache_file, results)
        return results
