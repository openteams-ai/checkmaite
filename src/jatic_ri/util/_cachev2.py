from __future__ import annotations

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
    @abc.abstractmethod
    def path(self, key: str) -> Path: ...

    @abc.abstractmethod
    def serialize(self, value: T) -> bytes: ...

    @abc.abstractmethod
    def deserialize(self, b: bytes) -> T: ...

    def set(self, key: str, value: T) -> None:
        with open(self.path(key), "wb") as f:
            f.write(self.serialize(value))

    def get(self, key: str) -> T | None:
        if (p := self.path(key)).is_file():
            with open(p, "rb") as f:
                return self.deserialize(f.read())
        else:
            return None


class _BinaryCache(Cache[bytes]):
    def path(self, key: str) -> Path:
        d = cache_path() / "binary"
        d.mkdir(parents=True, exist_ok=True)
        return d / key

    def serialize(self, value: bytes) -> bytes:
        return value

    def deserialize(self, b: bytes) -> bytes:
        return b


binary_cache = _BinaryCache()


@dataclasses.dataclass
class _BinaryDeSerializeConfig(Generic[T]):
    name: str
    cls: type[T] | tuple[type, ...]
    serialize: Callable[[T], bytes]
    deserialize: Callable[[bytes], T]


def _serialize_numpy(v: np.ndarray | np.number) -> bytes:
    with io.BytesIO() as b:
        np.save(b, v, allow_pickle=False)
        return b.getvalue()


def _deserialize_numpy(v: bytes) -> np.ndarray | np.number:
    with io.BytesIO(v) as b:
        return np.load(b, allow_pickle=False)


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
    def __init__(self, *configs: _BinaryDeSerializeConfig) -> None:
        self.configs = configs
        self._binary_pattern: re.Pattern[str] = re.compile(r"binary\+(?P<name>\w+)://(?P<key>.*)")

    def serialize(self, v: Any) -> Any:
        for config in self.configs:
            if isinstance(v, config.cls):
                break
        else:
            return v

        key = str(uuid.uuid4())
        binary_cache.set(key, config.serialize(v))

        return f"binary+{config.name}://{key}"

    def deserialize(self, v: Any) -> Any:
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
        cls=(np.ndarray, np.number),
        serialize=_serialize_numpy,
        deserialize=_deserialize_numpy,
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
    def from_model(cls, model: TModel) -> _SerializableModel[TModel]:
        return _SerializableModel(cls=type(model), dumped_fields=model.model_dump(mode="json"))

    def to_model(self) -> TModel:
        return self.cls.model_validate(self.dumped_fields)


class PydanticCache(Cache[TModel]):
    def serialize(self, value: TModel) -> bytes:
        return _SerializableModel.from_model(value).model_dump_json().encode()

    def deserialize(self, b: bytes) -> TModel:
        return _SerializableModel.model_validate_json(b).to_model()
