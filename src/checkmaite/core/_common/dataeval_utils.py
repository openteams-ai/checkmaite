from typing import TypeAlias, TypeVar

import maite.protocols.generic as gen
from dataeval.selection import Indices, Select
from dataeval.utils.data import split_dataset as dataeval_split_dataset
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


# TODO(checkmaite): Remove this wrapper once ``dataeval.selection.Select`` is
# typed as MAITE-compatible (i.e., it satisfies ``maite.protocols.generic.Dataset``
# with MAITE ``DatasetMetadata``). Today, ``Select`` follows DataEval protocol
# metadata typing, which causes pyright protocol mismatches in our MAITE-typed API.
class _MaiteSelectDataset(gen.Dataset[TInput, TTarget, TDatumMetadata]):
    """MAITE Dataset wrapper around ``dataeval.selection.Select``."""

    def __init__(
        self,
        dataset: gen.Dataset[TInput, TTarget, TDatumMetadata],
        indices: list[int],
    ) -> None:
        self._select = Select(dataset, selections=Indices(indices))
        self.metadata: DatasetMetadata = dataset.metadata

    def __getitem__(self, ind: int) -> tuple[TInput, TTarget, TDatumMetadata]:
        return self._select[ind]

    def __len__(self) -> int:
        return len(self._select)


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
        Either a list of the train/val subsets or a tuple of list of the train/val
        subsets and the test subset. The size of list of the train/val subsets is
        equal to num_folds.

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
        train_dataset = _MaiteSelectDataset(dataset, train_indices)
        val_dataset = _MaiteSelectDataset(dataset, val_indices)
        output.append((train_dataset, val_dataset))
    if len(test_indices) > 0:
        test_dataset = _MaiteSelectDataset(dataset, test_indices.tolist())
        return output, test_dataset
    return output
