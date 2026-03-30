import pytest
from maite.protocols import Dataset

from checkmaite.core._common.dataeval_utils import split_dataset


@pytest.mark.parametrize(
    ("num_folds", "stratify", "test_frac", "val_frac"),
    [
        (1, False, 0.0, 0.2),
        (1, False, 0.1, 0.1),
        (2, False, 0.1, 0.0),
        (2, False, 0.0, 0.0),
        (1, True, 0.0, 0.1),
    ],
)
def test_split_dataset(
    fake_ic_dataset_default,
    num_folds,
    stratify,
    test_frac,
    val_frac,
):
    output = split_dataset(
        fake_ic_dataset_default,
        num_folds,
        stratify=stratify,
        test_frac=test_frac,
        val_frac=val_frac,
    )

    trainval_set = output
    if test_frac > 0:
        assert isinstance(output, tuple)
        assert len(output) == 2
        trainval_set = output[0]
        test_set = output[1]
        assert isinstance(test_set, Dataset)

    assert isinstance(trainval_set, list)
    assert len(trainval_set) == num_folds

    for train_set, val_set in trainval_set:
        assert isinstance(train_set, Dataset)
        assert isinstance(val_set, Dataset)
