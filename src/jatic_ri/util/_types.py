import contextlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Generic, TypeVar

import matplotlib.figure
import pandas as pd
import PIL.Image
import pydantic
import pyspark.sql
import torch
from pydantic import BeforeValidator, PlainSerializer
from pydantic_core import core_schema
from smqtk_core import Plugfigurable
from smqtk_core.configuration import from_config_dict

from jatic_ri._common.models import set_device


def _to_image(v: Any) -> Any:
    """Convert input to a PIL Image if possible.

    Attempts to open various input types (path string, Path object,
    buffered IO base, matplotlib Figure) as a PIL Image.

    Parameters
    ----------
    v : Any
        The input value to convert.

    Returns
    -------
    Any
        A PIL.Image.Image object if conversion is successful,
        otherwise the original input `v`.
    """
    if isinstance(v, (str, Path, io.BufferedIOBase)):
        with contextlib.suppress(Exception):
            v = PIL.Image.open(v)  # pyright: ignore[reportArgumentType]
            v.load()
    elif isinstance(v, matplotlib.figure.Figure):
        with io.BytesIO() as b:
            v.savefig(b)
            b.seek(0)
            v = PIL.Image.open(b)
            v.load()

    return v


Image = Annotated[PIL.Image.Image, BeforeValidator(_to_image)]


def _to_pandas(v: Any) -> Any:
    """Convert input to a pandas DataFrame if possible.

    Specifically handles conversion from PySpark DataFrame to pandas DataFrame.

    Parameters
    ----------
    v : Any
        The input value to convert.

    Returns
    -------
    Any
        A pandas.DataFrame object if conversion is successful (i.e., input was
        a PySpark DataFrame), otherwise the original input `v`.
    """
    if isinstance(v, pyspark.sql.DataFrame):
        v = v.toPandas()

    return v


DataFrame = Annotated[pd.DataFrame, BeforeValidator(_to_pandas)]

Device = Annotated[torch.device, BeforeValidator(set_device), PlainSerializer(str)]

TPlugfigurable = TypeVar("TPlugfigurable", bound=Plugfigurable)


class _PlugfigurableAnnotationBase(Generic[TPlugfigurable]):
    """Base class for creating Pydantic-compatible annotations for Plugfigurable types.

    This class provides the Pydantic core schema logic to enable serialization
    and deserialization of `smqtk_core.Plugfigurable` instances using their
    configuration dictionaries.
    """

    _PLUGFIGURABLE_TYPE: type[TPlugfigurable]

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Generate the Pydantic core schema for the Plugfigurable type.

        This schema allows Pydantic to:
        1. Validate and deserialize from a configuration dictionary to an instance
           of the Plugfigurable type (via `from_config_dict` using the inner
           `from_config` helper).
        2. Serialize an instance of the Plugfigurable type to its configuration
           dictionary (via `get_config` using the inner `to_config` helper).

        Parameters
        ----------
        _source_type : Any
            The source type being annotated.
        _handler : pydantic.GetCoreSchemaHandler
            Pydantic's schema handler.

        Returns
        -------
        core_schema.CoreSchema
            The Pydantic core schema for handling the Plugfigurable type.
        """

        def from_config(value: Mapping[str, Any]) -> TPlugfigurable:
            return from_config_dict(dict(value), cls._PLUGFIGURABLE_TYPE.get_impls())

        def to_config(value: TPlugfigurable) -> dict[str, Any]:
            type_str = f"{type(value).__module__}.{type(value).__name__}"
            return {
                "type": type_str,
                type_str: value.get_config(),
            }

        from_config_dict_schema = core_schema.chain_schema(
            [
                core_schema.dict_schema(
                    keys_schema=core_schema.str_schema(),
                    values_schema=core_schema.any_schema(),
                ),
                core_schema.no_info_plain_validator_function(from_config),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_config_dict_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(cls._PLUGFIGURABLE_TYPE),
                    from_config_dict_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(to_config),
        )


if TYPE_CHECKING:
    DeSerializablePlugfigurable = Annotated[TPlugfigurable, ...]
else:

    class DeSerializablePlugfigurable:
        """A generic type annotation helper for `Plugfigurable` objects.

        This class enables the use of `DeSerializablePlugfigurable[MyPlugfigurableType]`
        as a Pydantic field type. It dynamically creates an annotation that
        instructs Pydantic how to serialize/deserialize `MyPlugfigurableType`
        instances using their configuration dictionaries.

        This is primarily used when `TYPE_CHECKING` is false, providing the runtime
        implementation for Pydantic.
        """

        @classmethod
        def __class_getitem__(cls, t: type[TPlugfigurable]) -> type[TPlugfigurable]:
            """Create a Pydantic-compatible annotated type for a Plugfigurable.

            When `DeSerializablePlugfigurable[SomePlugfigurableClass]` is used,
            this method is called with `SomePlugfigurableClass`. It returns
            an `Annotated` type that wraps `SomePlugfigurableClass` with
            a custom Pydantic handling logic provided by a dynamically generated
            subclass of `_PlugfigurableAnnotationBase`.

            Parameters
            ----------
            t : type[TPlugfigurable]
                The specific `Plugfigurable` type to be annotated.

            Returns
            -------
            type[TPlugfigurable]
                An `Annotated` type suitable for Pydantic model fields.
            """
            return Annotated[
                t, type(f"_{t.__name__}Annotation", (_PlugfigurableAnnotationBase,), {"_PLUGFIGURABLE_TYPE": t})
            ]
