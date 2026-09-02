import dataclasses
import math
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import polars as pl
import pytest
import torch
from pydantic import BaseModel
from torchvision import tv_tensors

from checkmaite.core._cached_metadata import validate_metadata_value


def test_validate_metadata_value_accepts_json_and_registered_binary_values():
    validate_metadata_value(
        {
            "id": "datum",
            "weight": 0.5,
            "flags": [True, None],
            "attributes": {"camera": "left"},
            "mask": np.asarray([True, False]),
        }
    )


@pytest.mark.parametrize(
    "value",
    [
        Path("image.png"),
        (1, 2),
        math.nan,
        {1: "class"},
        PIL.Image.new("RGB", (2, 2)),
        pd.DataFrame({"score": [0.5]}),
        pl.DataFrame({"score": [0.5]}),
        tv_tensors.Image(torch.zeros((3, 2, 2))),
    ],
)
def test_validate_metadata_value_rejects_values_that_cannot_round_trip(value):
    with pytest.raises(TypeError):
        validate_metadata_value({"id": "datum", "value": value})


def test_strict_values_reject_structured_objects_without_explicit_reconstruction():
    @dataclasses.dataclass
    class Statistics:
        count: int

    class Result(BaseModel):
        count: int

    for value in (Statistics(count=1), Result(count=1)):
        with pytest.raises(TypeError, match="unsupported strict cache value"):
            validate_metadata_value({"id": "datum", "value": value})


def test_strict_values_reject_out_of_range_integers():
    validate_metadata_value({"id": "datum", "value": 2**63 - 1})
    for value in (2**63, -(2**63) - 1):
        with pytest.raises(TypeError, match="signed 64-bit range"):
            validate_metadata_value({"id": "datum", "value": value})


def test_metadata_id_strictness_follows_serialization_policy():
    metadata = {"id": np.int64(1)}

    validate_metadata_value(metadata, strict=False)
    with pytest.raises(TypeError, match="exact int or str"):
        validate_metadata_value(metadata, strict=True)


def test_validate_metadata_value_rejects_lossy_numpy_state():
    nested_metadata_dtype = np.dtype([("value", np.dtype("i4", metadata={"unit": "pixels"}))])
    subarray_metadata_dtype = np.dtype((np.dtype("i4", metadata={"unit": "pixels"}), (2,)))
    aligned_dtype = np.dtype([("small", "u1"), ("large", "u4")], align=True)
    nested_aligned_dtype = np.dtype([("nested", aligned_dtype)])
    overlapping_dtype = np.dtype(
        {"names": ["first", "second"], "formats": ["u4", "u4"], "offsets": [0, 2], "itemsize": 6}
    )
    values = [
        np.asarray([object()], dtype=object),
        np.zeros(1, dtype=nested_metadata_dtype),
        np.zeros(1, dtype=subarray_metadata_dtype),
        np.zeros(1, dtype=aligned_dtype),
        np.zeros(1, dtype=nested_aligned_dtype),
        np.zeros(1, dtype=overlapping_dtype),
    ]
    for value in values:
        with pytest.raises(TypeError):
            validate_metadata_value({"id": "datum", "value": value})


def test_validate_metadata_value_accepts_noncanonical_numpy_storage():
    transposed = np.arange(6).reshape(2, 3).T
    sliced = np.arange(6)[::2]
    fortran = np.asfortranarray(np.arange(6).reshape(2, 3))
    read_only = np.arange(3)
    read_only.flags.writeable = False

    for value in (transposed, sliced, fortran, read_only):
        validate_metadata_value({"id": "datum", "value": value})


def test_compatibility_metadata_allows_lossy_and_reference_shaped_values():
    class ReferenceString(str, Enum):
        VALUE = "binary+numpy://00000000-0000-0000-0000-000000000000"

    values = (
        Path("image.png"),
        "binary+numpy://not-a-uuid",
        ("binary+numpy://00000000-0000-0000-0000-000000000000",),
        {"binary+numpy://00000000-0000-0000-0000-000000000000"},
        frozenset({ReferenceString.VALUE}),
    )
    for value in values:
        validate_metadata_value({"id": "datum", "value": value}, strict=False)


def test_validate_metadata_value_rejects_cyclic_containers():
    cyclic_list = []
    cyclic_list.append(cyclic_list)
    with pytest.raises(TypeError, match="cyclic reference"):
        validate_metadata_value({"id": "datum", "value": cyclic_list})

    cyclic_dict = {}
    cyclic_dict["self"] = cyclic_dict
    with pytest.raises(TypeError, match="cyclic reference"):
        validate_metadata_value({"id": "datum", "value": cyclic_dict})


def test_validate_metadata_value_rejects_excessive_nesting():
    value = None
    for _ in range(65):
        value = {"nested": value}

    with pytest.raises(TypeError, match="maximum supported cache nesting depth"):
        validate_metadata_value(value)


def test_validate_metadata_value_accepts_shared_noncyclic_containers():
    shared = [1, 2]
    validate_metadata_value({"id": "datum", "first": shared, "second": shared})
