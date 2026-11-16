import numpy as np
import pytest
import torch

from checkmaite.core.metric_keys import numeric_value, resolve_metric_key, undefined_to_none


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.5, 0.5),
        (2, 2.0),
        (np.float32(0.25), 0.25),
        (torch.tensor(0.75), 0.75),
        (torch.tensor([0.125]), 0.125),
    ],
)
def test_numeric_value_accepts_single_numbers(value, expected):
    assert numeric_value(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, True, np.bool_(False), "0.5", b"1", torch.tensor([0.1, 0.2]), [0.5], {"a": 1}])
def test_numeric_value_rejects_everything_else(value):
    assert numeric_value(value) is None


def test_the_config_key_wins_over_a_single_number():
    assert resolve_metric_key({"map": 0.3, "map_50": 0.5}, "map_50") == "map_50"


def test_a_metric_with_one_number_needs_no_key():
    assert resolve_metric_key({"accuracy": torch.tensor(0.9), "classes": [0, 1]}, None) == "accuracy"


def test_several_numbers_and_no_key_raise_and_list_the_keys():
    with pytest.raises(ValueError, match=r"several numeric results \('map', 'map_50'\).*Set metric_key"):
        resolve_metric_key({"map": 0.3, "map_50": 0.5}, None)


def test_several_numbers_and_no_key_give_none_when_not_required():
    assert resolve_metric_key({"map": 0.3, "map_50": 0.5}, None, required=False) is None


def test_no_numbers_raise_when_required():
    with pytest.raises(ValueError, match="no numeric results"):
        resolve_metric_key({"label": "cat"}, None)


def test_a_missing_key_raises_and_lists_the_keys():
    with pytest.raises(ValueError, match=r"'map_75' is not among the metric's results\. Numeric results: 'map'"):
        resolve_metric_key({"map": 0.3}, "map_75")


def test_a_key_that_is_not_a_single_number_raises():
    with pytest.raises(ValueError, match="'map_per_class' is not a single number"):
        resolve_metric_key({"map": 0.3, "map_per_class": torch.tensor([0.1, 0.2])}, "map_per_class")


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_an_undefined_key_raises(value):
    with pytest.raises(ValueError, match="'mota' is undefined for this evaluation"):
        resolve_metric_key({"mota": value, "motp": 0.5}, "mota")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), torch.tensor(float("nan"))])
def test_an_undefined_only_number_raises_like_a_named_one(value):
    with pytest.raises(ValueError, match="'score' is undefined for this evaluation"):
        resolve_metric_key({"score": value}, None)


def test_an_undefined_only_number_gives_none_when_not_required():
    assert resolve_metric_key({"score": float("nan")}, None, required=False) is None


def test_results_that_are_all_undefined_say_so():
    with pytest.raises(ValueError, match="'score' is undefined for this evaluation"):
        resolve_metric_key({"score": None, "label": "cat"}, None)


@pytest.mark.parametrize(("value", "expected"), [(float("nan"), None), (float("-inf"), None), (0.5, 0.5), ("x", "x")])
def test_undefined_to_none(value, expected):
    assert undefined_to_none(value) == expected


@pytest.mark.parametrize("value", [None, float("nan")])
def test_an_undefined_named_key_is_accepted_when_allowed(value):
    assert resolve_metric_key({"precision": value, "recall": 0.5}, "precision", allow_undefined=True) == "precision"


def test_an_undefined_only_number_is_accepted_when_allowed():
    assert resolve_metric_key({"precision": float("nan")}, None, allow_undefined=True) == "precision"


def test_a_missing_key_still_raises_when_undefined_is_allowed():
    with pytest.raises(ValueError, match="is not among"):
        resolve_metric_key({"recall": 0.5}, "precision", allow_undefined=True)
