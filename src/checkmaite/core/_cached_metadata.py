"""Validation for values persisted by cached tasks."""

import math
from typing import Any, cast

from typing_extensions import TypedDict

from checkmaite.core._cache import binary_de_serializer

_MAX_CACHE_NESTING_DEPTH = 64


class CachedDatumMetadata(TypedDict, extra_items=Any):
    """MAITE datum metadata with a required ID and arbitrary extension fields."""

    id: int | str


def _require_utf8(value: str, *, location: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise TypeError(f"{location} is not UTF-8 encodable.") from error


def _is_strict_cache_scalar(value: Any, *, location: str) -> bool:
    if value is None or type(value) is bool:
        return True
    if type(value) is int:
        if not -(2**63) <= value < 2**63:
            raise TypeError(f"{location} contains an integer outside the strict signed 64-bit range.")
        return True
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(f"{location} contains a non-finite float.")
        return True
    if type(value) is str:
        _require_utf8(value, location=location)
        return True
    return binary_de_serializer.supports(value)


def validate_cache_value(value: Any, *, location: str = "value") -> None:
    """Require task data with an explicitly lossless cache representation."""
    _validate_cache_value(
        value,
        location=location,
        depth=0,
        active_container_ids=set(),
        validated_container_ids=set(),
    )


def validate_metadata_value(value: Any, *, location: str = "metadata", strict: bool = True) -> None:
    """Validate metadata according to strict or compatibility cache policy."""
    if strict:
        if isinstance(value, dict) and "id" in value and type(value["id"]) not in (int, str):
            raise TypeError(f"{location}.id must be an exact int or str in strict mode.")
        validate_cache_value(value, location=location)
    else:
        return


def _enter_container(
    value: Any,
    *,
    location: str,
    depth: int,
    active_container_ids: set[int],
) -> None:
    if depth >= _MAX_CACHE_NESTING_DEPTH:
        raise TypeError(f"{location} exceeds the maximum supported cache nesting depth of {_MAX_CACHE_NESTING_DEPTH}.")
    if id(value) in active_container_ids:
        raise TypeError(f"{location} contains a cyclic reference.")
    active_container_ids.add(id(value))


def _validate_cache_value(
    value: Any,
    *,
    location: str,
    depth: int,
    active_container_ids: set[int],
    validated_container_ids: set[int],
) -> None:
    if _is_strict_cache_scalar(value, location=location):
        return
    if type(value) not in (list, dict):
        raise TypeError(
            f"{location} contains unsupported strict cache value of type "
            f"{type(value).__module__}.{type(value).__qualname__}."
        )

    if id(value) in validated_container_ids:
        return
    _enter_container(value, location=location, depth=depth, active_container_ids=active_container_ids)
    try:
        if type(value) is list:
            items = enumerate(cast(list[Any], value))
        else:
            dictionary = cast(dict[Any, Any], value)
            for key in dictionary:
                if type(key) is not str:
                    raise TypeError(f"{location} contains non-string dictionary key {key!r}.")
                _require_utf8(key, location=f"{location} dictionary key")
            items = dictionary.items()
        for key, item in items:
            child_location = f"{location}[{key}]" if type(value) is list else f"{location}.{key}"
            _validate_cache_value(
                item,
                location=child_location,
                depth=depth + 1,
                active_container_ids=active_container_ids,
                validated_container_ids=validated_container_ids,
            )
    finally:
        active_container_ids.remove(id(value))
    validated_container_ids.add(id(value))
