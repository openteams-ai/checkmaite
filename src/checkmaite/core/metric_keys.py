"""Choose which of a metric's results a capability reports as its single number.

A MAITE metric's ``compute()`` returns a dictionary, and some metrics return
several numbers, for example a COCO mAP metric returns ``map``, ``map_50``,
``map_75`` and more. A capability that needs one number, such as the value
NRTK Robustness plots, gets the key from its own config, following the same
convention as Lightning's ``monitor`` or Hugging Face's
``metric_for_best_model``. When the config does not name a key and the metric
returns exactly one number, that number is used.
"""

import math
from collections.abc import Mapping
from typing import Any, Literal, overload

import numpy as np

__all__ = ["numeric_value", "resolve_metric_key", "undefined_to_none"]


def numeric_value(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` if it is not a single number.

    Tensors and arrays with one element, NumPy scalars and Python numbers are
    numbers. Booleans, strings, ``None`` and anything with several elements
    are not.
    """
    if value is None or isinstance(value, (bool, np.bool_, str, bytes)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return None


@overload
def resolve_metric_key(
    results: Mapping[str, Any], metric_key: str | None, *, required: Literal[True] = ..., allow_undefined: bool = ...
) -> str: ...


@overload
def resolve_metric_key(
    results: Mapping[str, Any], metric_key: str | None, *, required: bool, allow_undefined: bool = ...
) -> str | None: ...


def resolve_metric_key(
    results: Mapping[str, Any],
    metric_key: str | None,
    *,
    required: bool = True,
    allow_undefined: bool = False,
) -> str | None:
    """Return the key of the result a capability should report.

    Parameters
    ----------
    results
        A metric's results, as returned by its ``compute()``.
    metric_key
        The key named in the capability's config, or ``None`` if the config
        does not name one.
    required
        If ``True``, raise when no key can be chosen. If ``False``, return
        ``None`` instead.
    allow_undefined
        If ``True``, accept a key whose value is undefined (NaN, infinite or
        ``None``) for these results, for a capability that reports the key over
        several evaluations and only needs it to exist in the first.

    Returns
    -------
    str | None
        ``metric_key`` if it is set; otherwise the key of the only numeric
        result; otherwise ``None`` when ``required`` is ``False``.

    Raises
    ------
    ValueError
        If ``metric_key`` is not a numeric result with a finite value, or if
        ``required`` is ``True`` and the results hold several numbers or none.
    """
    numeric = {key: number for key, value in results.items() if (number := numeric_value(value)) is not None}
    available = ", ".join(repr(key) for key in numeric) or "none"

    if metric_key is not None:
        _check_named_key(results, numeric, metric_key, available, allow_undefined=allow_undefined)
        return metric_key

    if len(numeric) == 1:
        # The only number is checked like a named one: an undefined value cannot be reported.
        only_key, only_value = next(iter(numeric.items()))
        if allow_undefined or math.isfinite(only_value):
            return only_key
        if not required:
            return None
        raise _undefined(only_key)
    if not required:
        return None
    if not numeric:
        undefined = [key for key, value in results.items() if value is None]
        if undefined:
            raise _undefined(", ".join(repr(key) for key in undefined), already_quoted=True)
        raise ValueError("The metric returned no numeric results, so there is no value to report.")
    raise ValueError(
        f"The metric returned several numeric results ({available}). Set metric_key in the capability's config to "
        "choose the one to report."
    )


def _check_named_key(
    results: Mapping[str, Any],
    numeric: Mapping[str, float],
    metric_key: str,
    available: str,
    *,
    allow_undefined: bool,
) -> None:
    """Raise if ``metric_key`` is not a finite, single-number result."""
    if metric_key not in results:
        raise ValueError(f"metric_key {metric_key!r} is not among the metric's results. Numeric results: {available}.")
    # ``None`` is how CheckMAITE records an undefined value, such as NaN.
    if results[metric_key] is not None and metric_key not in numeric:
        raise ValueError(f"metric_key {metric_key!r} is not a single number. Numeric results: {available}.")
    if allow_undefined:
        return
    if results[metric_key] is None or not math.isfinite(numeric[metric_key]):
        raise _undefined(metric_key)


def _undefined(key: str, *, already_quoted: bool = False) -> ValueError:
    name = key if already_quoted else repr(key)
    return ValueError(
        f"The metric result {name} is undefined for this evaluation (not a finite number), "
        "for example because a ratio's denominator is zero."
    )


def undefined_to_none(value: Any) -> Any:
    """Map a non-finite number to ``None``, the marker for an undefined metric value.

    A metric can be undefined for an evaluation, for example a ratio whose
    denominator is zero. JSON has no NaN or infinity, so the caches and run files
    would write them as ``null`` and a cached rerun would differ from a fresh run;
    normalizing them to ``None`` up front keeps both identical.
    """
    number = numeric_value(value)
    return None if number is not None and not math.isfinite(number) else value
