from typing import TypeAlias, TypeVar

import maite.protocols.generic as gen
from dataeval.data import Indices, Limit, Operation, View
from dataeval.data import split_dataset as dataeval_split_dataset
from maite.protocols import DatasetMetadata

TInput = TypeVar("TInput")
TTarget = TypeVar("TTarget")
TDatumMetadata = TypeVar("TDatumMetadata")

TrainDataset: TypeAlias = gen.Dataset[TInput, TTarget, TDatumMetadata]
ValidationDataset: TypeAlias = gen.Dataset[TInput, TTarget, TDatumMetadata]
TestDataset: TypeAlias = gen.Dataset[TInput, TTarget, TDatumMetadata]

TrainValidationFolds: TypeAlias = list[
    tuple[
        TrainDataset[TInput, TTarget, TDatumMetadata],
        ValidationDataset[TInput, TTarget, TDatumMetadata],
    ]
]
TrainValidationFoldsWithTest: TypeAlias = tuple[
    TrainValidationFolds[TInput, TTarget, TDatumMetadata],
    TestDataset[TInput, TTarget, TDatumMetadata],
]
SplitDatasetOutput: TypeAlias = (
    TrainValidationFolds[TInput, TTarget, TDatumMetadata]
    | TrainValidationFoldsWithTest[TInput, TTarget, TDatumMetadata]
)


# TODO(CheckMAITE): Remove this wrapper once ``dataeval.data.View`` is
# typed as MAITE-compatible (i.e., it satisfies ``maite.protocols.generic.Dataset``
# with MAITE ``DatasetMetadata``). Today, ``View`` follows DataEval protocol
# metadata typing, which causes pyright protocol mismatches in our MAITE-typed API.
class _MaiteViewDataset(gen.Dataset[TInput, TTarget, TDatumMetadata]):
    """MAITE Dataset wrapper around ``dataeval.data.View``."""

    def __init__(
        self,
        dataset: gen.Dataset[TInput, TTarget, TDatumMetadata],
        indices: list[int] | None = None,
        operations: list[Operation] | None = None,
    ) -> None:
        if (indices and operations) or not (indices or operations):
            raise ValueError("Only indices or operations should be provided")
        ops = Indices(indices) if indices is not None else operations
        self._view = View(dataset, operations=ops)
        self.metadata: DatasetMetadata = dataset.metadata

    def __getitem__(self, ind: int) -> tuple[TInput, TTarget, TDatumMetadata]:
        return self._view[ind]

    def __len__(self) -> int:
        return len(self._view)


def split_dataset(
    dataset: gen.Dataset[TInput, TTarget, TDatumMetadata],
    num_folds: int = 1,
    stratify: bool = True,
    split_on: list[str] | None = None,
    test_frac: float = 0.0,
    val_frac: float = 0.0,
) -> SplitDatasetOutput[TInput, TTarget, TDatumMetadata]:
    """Split a dataset into train/val or train/val/test subsets.

    Parameters
    ----------
    dataset
        Input dataset
    num_folds
        Number of train/val folds. If equal to 1, val_frac must be greater than 0.0
    stratify
        If true, dataset is split such that the class distribution
        of the entire dataset is preserved within each train/val partition,
        which is generally recommended.
    split_on
        Keys of the metadata dictionary upon which to group the dataset.
        A grouped partition is divided such that no group is present within
        both the training and validation set. Split_on groups should be selected
        to mitigate validation bias
    test_frac
        Fraction of data to be optionally held out for test set. If set to 0.0,
        only train and val subset are returned.
    val_frac
        Fraction of training data to be set aside for validation in the case
        where a single train/val split is desired

    Returns
    -------
    subsets
        SplitDatasetOutput instance.

    """
    split_defs = dataeval_split_dataset(
        dataset,
        num_folds=num_folds,
        stratify=stratify,
        split_on=split_on,
        test_frac=test_frac,
        val_frac=val_frac,
    )
    output: TrainValidationFolds[TInput, TTarget, TDatumMetadata] = []
    test_indices = split_defs.test
    for fold in split_defs.folds:
        train_indices = fold.train.tolist()
        val_indices = fold.val.tolist()
        train_dataset = _MaiteViewDataset(dataset, train_indices)
        val_dataset = _MaiteViewDataset(dataset, val_indices)
        output.append((train_dataset, val_dataset))
    if len(test_indices) > 0:
        test_dataset = _MaiteViewDataset(dataset, test_indices.tolist())
        return output, test_dataset
    return output


def make_subset(
    dataset: gen.Dataset[TInput, TTarget, TDatumMetadata],
    num_samples: int,
    stratify: bool = True,
) -> gen.Dataset[TInput, TTarget, TDatumMetadata]:
    """Make a subset of the dataset

    Parameters
    ----------
    dataset
        Input dataset
    num_samples:
        Number of samples in the output subset
    stratify:
        If true, dataset is split such that the class distribution
        of the entire dataset is preserved within each train/val partition,
        which is generally recommended.

    Returns
    -------
    subset
        gen.Dataset[TInput, TTarget, TDatumMetadata] instance
    """
    if not stratify:
        operation = Limit(num_samples)
    else:
        val_frac = (len(dataset) - num_samples + 1) / len(dataset)
        split_defs = dataeval_split_dataset(
            dataset,
            num_folds=1,
            stratify=stratify,
            test_frac=0.0,
            val_frac=val_frac,
        )
        indices = split_defs.folds[0].train.tolist()
        if len(indices) > num_samples:
            indices = indices[:num_samples]
        elif len(indices) < num_samples:
            others = split_defs.folds[0].val.tolist()
            num_remaining = num_samples - len(indices)
            indices = indices + others[:num_remaining]
        operation = Indices(indices)
    return _MaiteViewDataset(dataset, operations=[operation])
