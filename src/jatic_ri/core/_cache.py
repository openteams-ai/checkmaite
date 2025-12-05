import abc
import dataclasses
import io
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
import PIL.Image
import pydantic
import torch

from jatic_ri import cache_path

__all__ = ["Cache", "binary_cache", "binary_de_serializer", "PydanticCache"]


T = TypeVar("T")


class Cache(abc.ABC, Generic[T]):
    """Abstract base class for a key-value cache that stores serialized data."""

    @abc.abstractmethod
    def path(self, key: str) -> Path:
        """Get the file path for a given cache key.

        Parameters
        ----------
        key : str
            The cache key.

        Returns
        -------
        pathlib.Path
            The path where the data for the key is or should be stored.
        """
        ...

    @abc.abstractmethod
    def serialize(self, value: T) -> bytes:
        """Serialize the value into bytes.

        Parameters
        ----------
        value : T
            The value to serialize.

        Returns
        -------
        bytes
            The serialized byte representation of the value.
        """
        ...

    @abc.abstractmethod
    def deserialize(self, b: bytes) -> T:
        """Deserialize bytes into the original value.

        Parameters
        ----------
        b : bytes
            The bytes to deserialize.

        Returns
        -------
        T
            The deserialized value.
        """
        ...

    def set(self, key: str, value: T) -> None:
        """Set a value in the cache for a given key.

        The value is serialized and written to the path determined by the key.

        Parameters
        ----------
        key : str
            The cache key.
        value : T
            The value to store.
        """
        with open(self.path(key), "wb") as f:
            f.write(self.serialize(value))

    def get(self, key: str) -> T | None:
        """Get a value from the cache for a given key.

        If the key exists, its data is read, deserialized, and returned.

        Parameters
        ----------
        key : str
            The cache key.

        Returns
        -------
        T | None
            The deserialized value if the key exists, otherwise None.
        """
        if (p := self.path(key)).is_file():
            with open(p, "rb") as f:
                return self.deserialize(f.read())
        else:
            return None


class _BinaryCache(Cache[bytes]):
    """A cache for storing raw binary data."""

    def path(self, key: str) -> Path:
        """Get the file path for a given binary cache key.

        Creates the 'binary' cache subdirectory if it doesn't exist.

        Parameters
        ----------
        key : str
            The cache key.

        Returns
        -------
        pathlib.Path
            The path where the binary data for the key is stored.
        """
        d = cache_path() / "binary"
        d.mkdir(parents=True, exist_ok=True)
        return d / key

    def serialize(self, value: bytes) -> bytes:
        """Serialize a byte value (identity operation).

        Parameters
        ----------
        value : bytes
            The byte value to serialize.

        Returns
        -------
        bytes
            The input byte value.
        """
        return value

    def deserialize(self, b: bytes) -> bytes:
        """Deserialize bytes (identity operation).

        Parameters
        ----------
        b : bytes
            The bytes to deserialize.

        Returns
        -------
        bytes
            The input bytes.
        """
        return b


binary_cache = _BinaryCache()


@dataclasses.dataclass
class _BinaryDeSerializeConfig(Generic[T]):
    """Configuration for serializing/deserializing a specific type to/from binary."""

    name: str
    cls: type[T]
    serialize: Callable[[T], bytes]
    deserialize: Callable[[bytes], T]


def _serialize_numpy(v: np.ndarray | np.number) -> bytes:
    with io.BytesIO() as b:
        np.save(b, v, allow_pickle=False)
        return b.getvalue()


def _deserialize_numpy(v: bytes) -> np.ndarray:
    with io.BytesIO(v) as b:
        return np.load(b, allow_pickle=False)


def _deserialize_numpy_number(v: bytes) -> np.number:
    # numpy.load, used in _deserialize_numpy, deserializes numpy.number into a 0d numpy.ndarray
    a = _deserialize_numpy(v)
    return a.dtype.type(a)


def _serialize_pil_image(v: PIL.Image.Image) -> bytes:
    with io.BytesIO() as b:
        v.save(b, format="png")
        return b.getvalue()


def _deserialize_pil_image(v: bytes) -> PIL.Image.Image:
    with io.BytesIO(v) as b:
        image = PIL.Image.open(b)
        image.load()
        return image


def _serialize_pandas_df(v: pd.DataFrame) -> bytes:
    with io.BytesIO() as b:
        v.to_parquet(b)
        return b.getvalue()


def _deserialize_pandas_df(v: bytes) -> pd.DataFrame:
    with io.BytesIO(v) as b:
        return pd.read_parquet(b)


def _serialize_torch_tensor(v: torch.Tensor) -> bytes:
    with io.BytesIO() as b:
        torch.save(v, b)
        return b.getvalue()


def _deserialize_torch_tensor(v: bytes) -> torch.Tensor:
    with io.BytesIO(v) as b:
        return torch.load(b)


class _BinaryDeSerializer:
    """Handles serialization of specific types to binary cache and deserialization from it."""

    def __init__(self, *configs: _BinaryDeSerializeConfig) -> None:
        """Initialize with configurations for supported types.

        Parameters
        ----------
        *configs : _BinaryDeSerializeConfig
            Variable number of configurations, one for each type to be handled.
        """
        self.configs = configs
        self._binary_pattern: re.Pattern[str] = re.compile(r"binary\+(?P<name>\w+)://(?P<key>.*)")

    def register(self, config: _BinaryDeSerializeConfig) -> None:
        """Register a new type configuration for serialization/deserialization.

        Parameters
        ----------
        config : _BinaryDeSerializeConfig
            The configuration to register.
        """
        self.configs += (config,)

    def serialize(self, v: Any) -> Any:
        """Serialize a value.

        If the value's type matches a configured type, it's serialized to the
        binary cache and a string reference is returned. Otherwise, the value
        is returned unchanged.

        Parameters
        ----------
        v : Any
            The value to serialize.

        Returns
        -------
        Any
            A string reference if serialized to binary cache, or the original value.
        """
        for config in self.configs:
            if isinstance(v, config.cls):
                break
        else:
            return v

        key = str(uuid.uuid4())
        binary_cache.set(key, config.serialize(v))

        return f"binary+{config.name}://{key}"

    def deserialize(self, v: Any) -> Any:
        """Deserialize a value.

        If the value is a string reference matching the binary cache pattern,
        it's deserialized from the binary cache. Otherwise, the value is
        returned unchanged.

        Parameters
        ----------
        v : Any
            The value to deserialize.

        Returns
        -------
        Any
            The deserialized object if `v` was a binary cache reference,
            or the original value.

        Raises
        ------
        ValueError
            If the reference protocol is unknown or the key is not in the binary cache.
        """
        if not isinstance(v, str):
            return v

        match = self._binary_pattern.match(v)
        if not match:
            return v

        try:
            config = next(c for c in self.configs if c.name == match["name"])
        except StopIteration:
            raise ValueError(f"Unknown deserialization protocol {match['name']}") from None

        v = binary_cache.get(match["key"])
        if v is None:
            raise ValueError
        return config.deserialize(v)


binary_de_serializer = _BinaryDeSerializer(
    _BinaryDeSerializeConfig(
        name="numpy",
        cls=np.ndarray,
        serialize=_serialize_numpy,
        deserialize=_deserialize_numpy,
    ),
    _BinaryDeSerializeConfig(
        name="numpy_number",
        cls=np.number,
        serialize=_serialize_numpy,
        deserialize=_deserialize_numpy_number,
    ),
    _BinaryDeSerializeConfig(
        name="pil_image",
        cls=PIL.Image.Image,
        serialize=_serialize_pil_image,
        deserialize=_deserialize_pil_image,
    ),
    _BinaryDeSerializeConfig(
        name="pandas_df",
        cls=pd.DataFrame,
        serialize=_serialize_pandas_df,
        deserialize=_deserialize_pandas_df,
    ),
    _BinaryDeSerializeConfig(
        name="torch_tensor",
        cls=torch.Tensor,
        serialize=_serialize_torch_tensor,
        deserialize=_deserialize_torch_tensor,
    ),
)

TModel = TypeVar("TModel", bound=pydantic.BaseModel)


class _SerializableModel(pydantic.BaseModel, Generic[TModel]):
    """
    A plain pydantic.BaseModel.model_dump(...) will only dump the fields. Thus, one needs to know the specific model
    class before deserializing. Since we cannot impose this requirement on a generic cache for pydantic.BaseModels,
    we serialize the model class alongside the fields.
    """

    cls: pydantic.ImportString[type[TModel]]
    dumped_fields: dict[str, Any]

    @classmethod
    def from_model(cls, model: TModel) -> "_SerializableModel[TModel]":
        """Create a _SerializableModel instance from a Pydantic model.

        Parameters
        ----------
        model : TModel
            The Pydantic model instance to serialize.

        Returns
        -------
        _SerializableModel[TModel]
            A serializable representation containing the model's class and dumped fields.
        """
        return _SerializableModel(cls=type(model), dumped_fields=model.model_dump(mode="json"))

    def to_model(self) -> TModel:
        """Convert this _SerializableModel back to its original Pydantic model type.

        Returns
        -------
        TModel
            The deserialized Pydantic model instance.
        """
        return self.cls.model_validate(self.dumped_fields)


class PydanticCache(Cache[TModel]):
    """A cache for Pydantic models.

    Serializes models to JSON, including their type information for robust deserialization.
    """

    def serialize(self, value: TModel) -> bytes:
        """Serialize a Pydantic model to JSON bytes.

        Parameters
        ----------
        value : TModel
            The Pydantic model instance to serialize.

        Returns
        -------
        bytes
            The JSON representation of the model, encoded to bytes.
        """
        return _SerializableModel.from_model(value).model_dump_json().encode()

    def deserialize(self, b: bytes) -> TModel:
        """Deserialize bytes (JSON) back into a Pydantic model.

        Parameters
        ----------
        b : bytes
            The JSON bytes to deserialize.

        Returns
        -------
        TModel
            The deserialized Pydantic model instance.
        """
        return _SerializableModel.model_validate_json(b).to_model()
