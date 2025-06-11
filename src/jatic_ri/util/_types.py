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
    if isinstance(v, pyspark.sql.DataFrame):
        v = v.toPandas()

    return v


DataFrame = Annotated[pd.DataFrame, BeforeValidator(_to_pandas)]

Device = Annotated[torch.device, BeforeValidator(set_device), PlainSerializer(str)]

TPlugfigurable = TypeVar("TPlugfigurable", bound=Plugfigurable)


class _PlugfigurableAnnotationBase(Generic[TPlugfigurable]):
    _PLUGFIGURABLE_TYPE: type[TPlugfigurable]

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def from_config(value: Mapping[str, Any]) -> TPlugfigurable:
            return from_config_dict(dict(value), cls._PLUGFIGURABLE_TYPE.get_impls())

        def to_config(value: TPlugfigurable) -> dict[str, Any]:
            return {
                "type": (t := f"{type(value).__module__}.{type(value).__name__}"),
                t: value.get_config(),
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
        @classmethod
        def __class_getitem__(cls, t: type[TPlugfigurable]) -> type[TPlugfigurable]:
            return Annotated[
                t, type(f"_{t.__name__}Annotation", (_PlugfigurableAnnotationBase,), {"_PLUGFIGURABLE_TYPE": t})
            ]
