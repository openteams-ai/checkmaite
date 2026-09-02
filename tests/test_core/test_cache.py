import builtins
import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import PIL.Image
import polars as pl
import pytest
import torch
from pydantic import Field, field_serializer
from pydantic_core import PydanticSerializationError
from torchvision import tv_tensors

from checkmaite.core._cache import (
    PydanticCache,
    _BinaryCache,
    _BinaryDeSerializeConfig,
    _BinaryDeSerializer,
    binary_de_serializer,
)
from checkmaite.core.capability_core import CapabilityOutputsBase

_REFERENCE = "binary+numpy://00000000-0000-0000-0000-000000000000"


class ReferenceOutput(CapabilityOutputsBase):
    value: Any


class AliasedOutput(CapabilityOutputsBase):
    value: int = Field(alias="aliased_value")


class SerializedReferenceOutput(CapabilityOutputsBase):
    value: str

    @field_serializer("value")
    def serialize_value(self, _):
        return _REFERENCE


def test_binary_codec_selection_prefers_exact_types():
    class Parent:
        pass

    class Child(Parent):
        pass

    serializer = _BinaryDeSerializer(
        _BinaryDeSerializeConfig(
            name="parent",
            cls=Parent,
            serialize=lambda _: b"parent",
            deserialize=lambda _: Parent(),
            lossless_subclasses=True,
        ),
        _BinaryDeSerializeConfig(
            name="child",
            cls=Child,
            serialize=lambda _: b"child",
            deserialize=lambda _: Child(),
        ),
    )

    value = Child()
    reference = serializer.serialize(value)

    assert not serializer.supports(value)
    assert reference.startswith("binary+child://")
    assert type(serializer.deserialize(reference)) is Child


def test_binary_codec_names_are_valid_and_unique():
    config = _BinaryDeSerializeConfig(
        name="value",
        cls=int,
        serialize=lambda _: b"value",
        deserialize=lambda _: 1,
    )
    with pytest.raises(ValueError, match="word characters"):
        _BinaryDeSerializer(dataclasses.replace(config, name="not-valid"))
    with pytest.raises(ValueError, match="unique"):
        _BinaryDeSerializer(config, config)


def test_strict_admission_uses_the_publication_codec():
    unsafe = _BinaryDeSerializeConfig(
        name="unsafe",
        cls=int,
        serialize=lambda _: b"unsafe",
        deserialize=lambda _: 1,
    )
    safe = _BinaryDeSerializeConfig(
        name="safe",
        cls=int,
        serialize=lambda _: b"safe",
        deserialize=lambda _: 1,
        strict_safe=lambda _: True,
    )
    serializer = _BinaryDeSerializer(unsafe, safe)

    assert not serializer.supports(1)
    assert serializer.serialize(1).startswith("binary+unsafe://")


def test_custom_codecs_must_explicitly_opt_into_strict_mode():
    class Value:
        pass

    serializer = _BinaryDeSerializer(
        _BinaryDeSerializeConfig(
            name="value",
            cls=Value,
            serialize=lambda _: b"value",
            deserialize=lambda _: Value(),
            strict_safe=lambda _: True,
        )
    )

    assert serializer.supports(Value())


def test_numpy_scalar_subclasses_are_not_strict_safe():
    class CustomFloat(np.float64):
        pass

    assert not binary_de_serializer.supports(CustomFloat(1.0))


def test_numpy_scalars_include_booleans():
    value = np.bool_(True)

    assert binary_de_serializer.supports(value)
    restored = binary_de_serializer.deserialize(binary_de_serializer.serialize(value))
    assert type(restored) is np.bool_
    assert restored


def test_strict_numpy_scalar_admission_requires_exact_type_round_trip():
    values = (np.longlong(1), np.ulonglong(1), np.longdouble(1), np.clongdouble(1))

    for value in values:
        restored = binary_de_serializer.deserialize(binary_de_serializer.serialize(value))
        assert binary_de_serializer.supports(value) is (type(restored) is type(value))


def test_strict_numpy_preserves_nonfinite_binary_values():
    value = np.asarray([np.nan, np.inf], dtype=np.float32)

    assert binary_de_serializer.supports(value)
    restored = binary_de_serializer.deserialize(binary_de_serializer.serialize(value))
    np.testing.assert_equal(restored, value)


def test_wide_numpy_headers_round_trip_from_trusted_sidecars():
    dtype = np.dtype([(f"scientific_field_{index:04d}", "u1") for index in range(1000)])
    value = np.zeros(1, dtype=dtype)

    assert binary_de_serializer.supports(value)
    restored = binary_de_serializer.deserialize(binary_de_serializer.serialize(value))
    assert restored.dtype == dtype


def test_unsupported_numpy_values_are_not_sent_to_the_binary_codec():
    value = np.asarray([object()], dtype=object)

    assert not binary_de_serializer.supports(value)
    assert binary_de_serializer.serialize(value) is value


def test_strict_torch_rejects_autograd_and_non_strided_layouts():
    assert not binary_de_serializer.supports(torch.ones(1, requires_grad=True))

    tensor_with_grad = torch.ones(1)
    tensor_with_grad.grad = torch.ones(1)
    assert not binary_de_serializer.supports(tensor_with_grad)

    sparse_tensor = torch.sparse_coo_tensor([[0]], [1.0], (1,), check_invariants=True)
    assert not binary_de_serializer.supports(sparse_tensor)


def test_pil_subclasses_remain_supported_for_outputs_but_not_metadata(tmp_path):
    path = tmp_path / "animated.gif"
    frames = [PIL.Image.new("RGB", (2, 2), color=color) for color in ("red", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:], format="GIF")

    with PIL.Image.open(path) as image:
        assert type(image) is not PIL.Image.Image
        assert not binary_de_serializer.supports(image)
        reference = binary_de_serializer.serialize(image)
        assert reference.startswith("binary+pil_image://")
        assert isinstance(binary_de_serializer.deserialize(reference), PIL.Image.Image)


@pytest.mark.parametrize(
    ("value", "protocol"),
    [
        (pd.DataFrame({"score": [0.5]}), "pandas_df"),
        (pl.DataFrame({"score": [0.5]}), "polars_df"),
        (tv_tensors.Image(torch.zeros((3, 2, 2))), "torch_tensor"),
    ],
)
def test_main_output_codecs_remain_available_without_claiming_metadata_losslessness(value, protocol):
    assert not binary_de_serializer.supports(value)

    reference = binary_de_serializer.serialize(value)
    if protocol == "pandas_df" and reference is value:
        pytest.skip("A pandas Parquet engine is not installed")

    assert reference.startswith(f"binary+{protocol}://")
    assert type(binary_de_serializer.deserialize(reference)) is type(value)


def test_binary_cache_deletes_partial_destination_after_failed_write(tmp_path, monkeypatch):
    class TempBinaryCache(_BinaryCache):
        def path(self, key: str) -> Path:
            return tmp_path / key

    destination = tmp_path / "entry"
    real_open = builtins.open

    class FailingFile:
        def __enter__(self):
            self.file = real_open(destination, "xb")
            return self

        def write(self, value):
            self.file.write(value[:1])
            raise OSError("disk full")

        def __exit__(self, *_):
            self.file.close()

    monkeypatch.setattr(builtins, "open", lambda *_, **__: FailingFile())

    with pytest.raises(OSError, match="disk full"):
        TempBinaryCache().set("entry", b"payload")

    assert not destination.exists()


def test_binary_cache_does_not_delete_preexisting_destination(tmp_path):
    class TempBinaryCache(_BinaryCache):
        def path(self, key: str) -> Path:
            return tmp_path / key

    destination = tmp_path / "entry"
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        TempBinaryCache().set("entry", b"replacement")

    assert destination.read_bytes() == b"existing"


def test_reference_escaping_is_in_place_and_rejects_invalid_base64():
    nested = ["binary+numpy://00000000-0000-0000-0000-000000000000"]
    payload = {"nested": nested}

    escaped = binary_de_serializer.escape_user_references(payload)

    assert escaped is payload
    assert payload["nested"] is nested
    with pytest.raises(ValueError, match="Invalid escaped cache string"):
        binary_de_serializer.deserialize("checkmaite+escaped-string://%%%")


def test_pydantic_cache_escapes_user_reference_strings(tmp_path):
    class OutputCache(PydanticCache[ReferenceOutput]):
        def path(self, key: str) -> Path:
            return tmp_path / key

    cache = OutputCache()
    cache.set("output", ReferenceOutput(value={_REFERENCE}))

    assert cache.get("output").value == [_REFERENCE]


def test_pydantic_cache_escapes_reference_strings_from_field_serializers(tmp_path):
    class OutputCache(PydanticCache[SerializedReferenceOutput]):
        def path(self, key: str) -> Path:
            return tmp_path / key

    cache = OutputCache()
    cache.set("output", SerializedReferenceOutput(value="original"))

    assert cache.get("output").value == _REFERENCE


def test_pydantic_cache_round_trips_aliases(tmp_path):
    class OutputCache(PydanticCache[CapabilityOutputsBase]):
        def path(self, key: str) -> Path:
            return tmp_path / key

    cache = OutputCache()
    cache.set("alias", AliasedOutput(aliased_value=1))

    assert cache.get("alias") == AliasedOutput(aliased_value=1)


def test_pydantic_cache_rejects_local_models(tmp_path):
    class OutputCache(PydanticCache[CapabilityOutputsBase]):
        def path(self, key: str) -> Path:
            return tmp_path / key

    class LocalOutput(CapabilityOutputsBase):
        value: int

    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        assert not OutputCache().try_set("local", LocalOutput(value=1))


def test_pydantic_cache_publication_does_not_read_binary_sidecars(tmp_path, monkeypatch):
    class OutputCache(PydanticCache[ReferenceOutput]):
        def path(self, key: str) -> Path:
            return tmp_path / key

    def fail_if_deserialized(value):
        raise AssertionError(f"Publication unexpectedly deserialized {value!r}")

    monkeypatch.setattr(binary_de_serializer, "deserialize", fail_if_deserialized)

    cache = OutputCache()
    cache.set("output", ReferenceOutput(value=np.asarray([1, 2, 3])))

    assert cache.path("output").is_file()


def test_pydantic_cache_treats_legacy_envelopes_as_misses(tmp_path):
    class OutputCache(PydanticCache[ReferenceOutput]):
        def path(self, key: str) -> Path:
            return tmp_path / key

    cache = OutputCache()
    envelope = json.loads(cache.serialize(ReferenceOutput(value="literal")))
    assert envelope["cache_schema_version"] == 1
    envelope.pop("cache_schema_version")
    cache.path("legacy").write_text(json.dumps(envelope))

    with pytest.warns(UserWarning, match="Ignoring invalid cache entry"):
        assert cache.get("legacy") is None


def test_pydantic_cache_removes_binary_sidecars_after_failed_publication(tmp_path):
    class Output(CapabilityOutputsBase):
        values: list[Any]

    class OutputCache(PydanticCache[Output]):
        def path(self, key: str) -> Path:
            return tmp_path / key

    class Unsupported:
        pass

    with pytest.raises(PydanticSerializationError):
        OutputCache().set("output", Output(values=[np.asarray([1]), Unsupported()]))

    assert not (tmp_path / "output").exists()
    assert list((tmp_path / "binary").glob("*")) == []


def test_optional_cache_warnings_do_not_become_computation_errors(tmp_path):
    class Output(CapabilityOutputsBase):
        values: list[Any]

    class OutputCache(PydanticCache[Output]):
        def path(self, key: str) -> Path:
            return tmp_path / key

    class Unsupported:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not OutputCache().try_set("output", Output(values=[Unsupported()]))


def test_pydantic_cache_can_skip_unsupported_values(tmp_path):
    class Output(CapabilityOutputsBase):
        values: list[Any]

    class OutputCache(PydanticCache[Output]):
        def path(self, key: str) -> Path:
            return tmp_path / key

    class Unsupported:
        pass

    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        assert not OutputCache().try_set("output", Output(values=[Unsupported()]))
    assert not (tmp_path / "output").exists()
