import pytest

from checkmaite.core._common._knn import _knn_lowerbound


@pytest.mark.parametrize(
    ("upper", "num_classes", "k", "expected"),
    [
        (0.0, 2, 7, 0.0),
        (0.5, 2, 2, 0.25),
        (0.5, 2, 4, 0.5 / (1 + (1 / 2))),
    ],
)
def test_knn_lowerbound_simple_binary_branches(upper: float, num_classes: int, k: int, expected: float) -> None:
    assert _knn_lowerbound(upper, num_classes, k) == pytest.approx(expected)


def test_knn_lowerbound_large_k_binary_branch() -> None:
    assert 0.0 < _knn_lowerbound(0.5, 2, 7) < 0.5
