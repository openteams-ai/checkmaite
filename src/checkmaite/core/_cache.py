import abc
import base64
import contextvars
import dataclasses
import functools
import importlib.util
import io
import logging
import os
import re
import uuid
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast

import numpy as np
import pandas as pd
import PIL.Image
import polars as pl
import pydantic
import torch

from checkmaite import cache_path

__all__ = ["Cache", "binary_cache", "binary_de_serializer", "PydanticCache"]


T = TypeVar("T")


def warn_optional_cache_failure(message: str, *, stacklevel: int) -> None:
    """Report an optional cache failure without honoring warning-as-error filters."""
    try:
        warnings.warn(message, stacklevel=stacklevel)
    except Warning:
        logging.getLogger(__name__).warning(message)


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
        serialized = self.serialize(value)
        destination = self.path(key)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(temporary, "xb") as f:
                f.write(serialized)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def contains(self, key: str) -> bool:
        """Return whether a cache entry exists without deserializing it."""
        return self.path(key).is_file()

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
    """A cache for immutable binary data addressed by unique UUID keys."""

    def set(self, key: str, value: bytes) -> None:
        """Write a new immutable binary entry and delete it if writing fails."""
        destination = self.path(key)
        created = False
        try:
            with open(destination, "xb") as f:
                created = True
                f.write(self.serialize(value))
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise

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


def _supports_all_values(_: Any) -> bool:
    return True


def _supports_no_values(_: Any) -> bool:
    return False


@dataclasses.dataclass
class _BinaryDeSerializeConfig(Generic[T]):
    """Configuration for serializing/deserializing a specific type to/from binary."""

    name: str
    cls: type[T]
    serialize: Callable[[T], bytes]
    deserialize: Callable[[bytes], T]
    lossless_subclasses: bool = False
    # Use for constraints on whether the representation codec can handle a value.
    supports_value: Callable[[T], bool] = _supports_all_values
    # Compatibility mode retains main's best-effort subclass support, while
    # strict task caching admits only values satisfying this predicate.
    strict_safe: Callable[[T], bool] = _supports_no_values


def _serialize_numpy(v: np.ndarray | np.generic) -> bytes:
    with io.BytesIO() as b:
        np.save(b, v, allow_pickle=False)
        return b.getvalue()


def _deserialize_numpy(v: bytes) -> np.ndarray:
    with io.BytesIO(v) as b:
        # Supported NumPy runtimes accept max_header_size, but older stubs omit it.
        load = cast(Any, np.load)
        return load(b, allow_pickle=False, max_header_size=len(v))


def _deserialize_numpy_number(v: bytes) -> np.generic:
    # numpy.load, used in _deserialize_numpy, deserializes numpy scalars into a 0d numpy.ndarray
    a = _deserialize_numpy(v)
    return a.dtype.type(a)


def _supports_numpy_array(v: np.ndarray) -> bool:
    return not v.dtype.hasobject


def _dtype_has_unsupported_state(dtype: np.dtype, *, seen: set[int] | None = None) -> bool:
    seen = set() if seen is None else seen
    if id(dtype) in seen:
        return False
    seen.add(id(dtype))
    if dtype.metadata is not None or dtype.isalignedstruct:
        return True
    if dtype.subdtype is not None and _dtype_has_unsupported_state(dtype.subdtype[0], seen=seen):
        return True
    return bool(dtype.fields) and any(
        _dtype_has_unsupported_state(field[0], seen=seen) for field in dtype.fields.values()
    )


@functools.lru_cache(maxsize=256)
def _dtype_round_trips(dtype: np.dtype) -> bool:
    try:
        restored = _deserialize_numpy(_serialize_numpy(np.empty(0, dtype=dtype)))
    except (TypeError, ValueError):
        return False
    return restored.dtype == dtype


@functools.lru_cache(maxsize=256)
def _numpy_scalar_type_round_trips(scalar_type: type[np.generic], dtype: np.dtype) -> bool:
    try:
        value = np.zeros((), dtype=dtype)[()]
        restored = _deserialize_numpy_number(_serialize_numpy(value))
    except (TypeError, ValueError):
        return False
    return type(value) is scalar_type and type(restored) is scalar_type


def _supports_numpy_strict(v: np.ndarray | np.generic) -> bool:
    dtype = v.dtype
    if dtype.hasobject or _dtype_has_unsupported_state(dtype) or not _dtype_round_trips(dtype):
        return False
    if isinstance(v, np.ndarray):
        return type(v) is np.ndarray
    return type(v) is dtype.type and _numpy_scalar_type_round_trips(type(v), dtype)


def _supports_torch_strict(v: torch.Tensor) -> bool:
    return (
        type(v) is torch.Tensor
        and v.device.type == "cpu"
        and v.layout is torch.strided
        and not v.requires_grad
        and v.grad is None
    )


def _serialize_pil_image(v: PIL.Image.Image) -> bytes:
    with io.BytesIO() as b:
        v.save(b, format="png")
        return b.getvalue()


def _deserialize_pil_image(v: bytes) -> PIL.Image.Image:
    with io.BytesIO(v) as b:
        image = PIL.Image.open(b)
        image.load()
        return image.copy()


def _supports_pandas_dataframe(_: pd.DataFrame) -> bool:
    return importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None


def _serialize_pandas_df(v: pd.DataFrame) -> bytes:
    with io.BytesIO() as b:
        v.to_parquet(b)
        return b.getvalue()


def _deserialize_pandas_df(v: bytes) -> pd.DataFrame:
    with io.BytesIO(v) as b:
        return pd.read_parquet(b)


def _serialize_polars_df(v: pl.DataFrame) -> bytes:
    with io.BytesIO() as b:
        v.write_parquet(b)
        return b.getvalue()


def _deserialize_polars_df(v: bytes) -> pl.DataFrame:
    with io.BytesIO(v) as b:
        return pl.read_parquet(b)


def _serialize_torch_tensor(v: torch.Tensor) -> bytes:
    with io.BytesIO() as b:
        torch.save(v, b)
        return b.getvalue()


def _deserialize_torch_tensor(v: bytes) -> torch.Tensor:
    with io.BytesIO(v) as b:
        # Cache files are locally generated and may contain Tensor subclasses such
        # as torchvision TVTensors, which weights-only loading does not support.
        return torch.load(b, weights_only=False)


class _BinaryDeSerializer:
    """Handles serialization of specific types to binary cache and deserialization from it."""

    def __init__(self, *configs: _BinaryDeSerializeConfig) -> None:
        """Initialize with configurations for supported types.

        Parameters
        ----------
        *configs : _BinaryDeSerializeConfig
            Variable number of configurations, one for each type to be handled.
        """
        self._validate_configs(configs)
        self.configs = configs
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        self._binary_pattern: re.Pattern[str] = re.compile(rf"binary\+(?P<name>\w+)://(?P<key>{uuid_pattern})")
        self._transaction_keys: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
            "binary_cache_transaction_keys", default=None
        )

    @staticmethod
    def _validate_configs(configs: Sequence[_BinaryDeSerializeConfig]) -> None:
        names = [config.name for config in configs]
        if any(re.fullmatch(r"\w+", name) is None for name in names):
            raise ValueError("Binary codec names may contain only word characters.")
        if len(names) != len(set(names)):
            raise ValueError("Binary codec names must be unique.")

    def _publication_config(self, value: Any) -> _BinaryDeSerializeConfig | None:
        exact_matches = (config for config in self.configs if type(value) is config.cls)
        subclass_matches = (config for config in self.configs if isinstance(value, config.cls))
        return next((config for config in (*exact_matches, *subclass_matches) if config.supports_value(value)), None)

    def _matching_config(self, value: Any, *, strict: bool) -> _BinaryDeSerializeConfig | None:
        config = self._publication_config(value)
        if config is None or not strict:
            return config
        if type(value) is not config.cls and not config.lossless_subclasses:
            return None
        return config if config.strict_safe(value) else None

    def supports(self, value: Any) -> bool:
        """Return whether a value is safe for strict task caching."""
        return self._matching_config(value, strict=True) is not None

    @contextmanager
    def rollback_binary_writes_on_error(self) -> Iterator[None]:
        """Remove binary sidecars if publication of their parent artifact fails."""
        active_keys = self._transaction_keys.get()
        if active_keys is not None:
            yield
            return

        keys: set[str] = set()
        token = self._transaction_keys.set(keys)
        try:
            yield
        except BaseException:
            for key in keys:
                binary_cache.path(key).unlink(missing_ok=True)
            raise
        finally:
            self._transaction_keys.reset(token)

    def is_reference(self, value: Any) -> bool:
        """Return whether a string would be interpreted as a binary reference."""
        return isinstance(value, str) and self._binary_pattern.fullmatch(value) is not None

    def escape_user_references(self, value: Any) -> Any:
        """Escape user strings that overlap Checkmaite's serialized reference syntax."""
        if isinstance(value, str):
            transaction_keys = self._transaction_keys.get() or []
            match = self._binary_pattern.fullmatch(value)
            generated = match is not None and match["key"] in transaction_keys
            if generated:
                return value
            if value.startswith("checkmaite+escaped-string://") or match is not None:
                encoded = base64.urlsafe_b64encode(value.encode()).decode()
                return f"checkmaite+escaped-string://{encoded}"
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = self.escape_user_references(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                value[key] = self.escape_user_references(item)
        return value

    def register(self, config: _BinaryDeSerializeConfig) -> None:
        """Register a new type configuration for serialization/deserialization.

        Parameters
        ----------
        config : _BinaryDeSerializeConfig
            The configuration to register.
        """
        self._validate_configs((*self.configs, config))
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
        config = self._matching_config(v, strict=False)
        if config is None:
            return v

        key = str(uuid.uuid4())
        binary_cache.set(key, config.serialize(v))
        transaction_keys = self._transaction_keys.get()
        if transaction_keys is not None:
            transaction_keys.add(key)

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
        escape_prefix = "checkmaite+escaped-string://"
        if v.startswith(escape_prefix):
            try:
                encoded = v.removeprefix(escape_prefix)
                return base64.b64decode(encoded, altchars=b"-_", validate=True).decode()
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError("Invalid escaped cache string.") from error

        match = self._binary_pattern.fullmatch(v)
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
        supports_value=_supports_numpy_array,
        strict_safe=_supports_numpy_strict,
    ),
    _BinaryDeSerializeConfig(
        name="numpy_number",
        cls=np.generic,
        serialize=_serialize_numpy,
        deserialize=_deserialize_numpy_number,
        lossless_subclasses=True,
        strict_safe=_supports_numpy_strict,
    ),
    _BinaryDeSerializeConfig(
        name="pil_image",
        cls=PIL.Image.Image,
        serialize=_serialize_pil_image,
        deserialize=_deserialize_pil_image,
        strict_safe=_supports_no_values,
    ),
    _BinaryDeSerializeConfig(
        name="pandas_df",
        cls=pd.DataFrame,
        serialize=_serialize_pandas_df,
        deserialize=_deserialize_pandas_df,
        supports_value=_supports_pandas_dataframe,
        strict_safe=_supports_no_values,
    ),
    _BinaryDeSerializeConfig(
        name="polars_df",
        cls=pl.DataFrame,
        serialize=_serialize_polars_df,
        deserialize=_deserialize_polars_df,
        strict_safe=_supports_no_values,
    ),
    _BinaryDeSerializeConfig(
        name="torch_tensor",
        cls=torch.Tensor,
        serialize=_serialize_torch_tensor,
        deserialize=_deserialize_torch_tensor,
        strict_safe=_supports_torch_strict,
    ),
)

TModel = TypeVar("TModel", bound=pydantic.BaseModel)


def _deserialize_dumped_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [_deserialize_dumped_fields(item) for item in value]
    if isinstance(value, dict):
        return {key: _deserialize_dumped_fields(item) for key, item in value.items()}
    return binary_de_serializer.deserialize(value)


class _SerializableModel(pydantic.BaseModel, Generic[TModel]):
    """
    A plain pydantic.BaseModel.model_dump(...) will only dump the fields. Thus, one needs to know the specific model
    class before deserializing. Since we cannot impose this requirement on a generic cache for pydantic.BaseModels,
    we serialize the model class alongside the fields.
    """

    cache_schema_version: Literal[1]
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
        dumped_fields = model.model_dump(mode="json", by_alias=True, round_trip=True)
        binary_de_serializer.escape_user_references(dumped_fields)
        return _SerializableModel(cache_schema_version=1, cls=type(model), dumped_fields=dumped_fields)

    def to_model(self) -> TModel:
        """Convert this _SerializableModel back to its original Pydantic model type.

        Returns
        -------
        TModel
            The deserialized Pydantic model instance.
        """
        return self.cls.model_validate(_deserialize_dumped_fields(self.dumped_fields))


class PydanticCache(Cache[TModel]):
    """A cache for Pydantic models.

    Serializes models to JSON, including their type information for robust deserialization.
    """

    def get(self, key: str) -> TModel | None:
        """Treat stale or malformed Pydantic entries as cache misses."""
        try:
            return super().get(key)
        except Exception as error:  # noqa: BLE001 - cache reads are optional
            warn_optional_cache_failure(f"Ignoring invalid cache entry {key!r}: {error}", stacklevel=2)
            return None

    def try_set(self, key: str, value: TModel) -> bool:
        """Publish when serializable, without failing an otherwise successful operation."""
        try:
            self.set(key, value)
        except Exception as error:  # noqa: BLE001 - caching is optional after completed work
            warn_optional_cache_failure(
                f"Cache publication is disabled for this value: {error}",
                stacklevel=2,
            )
            return False
        return True

    def set(self, key: str, value: TModel) -> None:
        """Atomically publish a model or remove any binary sidecars written before failure."""
        with binary_de_serializer.rollback_binary_writes_on_error():
            super().set(key, value)

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
        with binary_de_serializer.rollback_binary_writes_on_error():
            serialized = _SerializableModel.from_model(value).model_dump_json().encode()
            envelope = _SerializableModel.model_validate_json(serialized)
            if envelope.cls is not type(value):
                raise TypeError("The cached model class does not have a stable import path.")
            return serialized

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
