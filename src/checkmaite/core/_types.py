import contextlib
import io
from pathlib import Path
from typing import Annotated, Any, Literal

import matplotlib.figure
import pandas as pd
import PIL.Image
import pydantic
import pyspark.sql
import torch
from pydantic import BeforeValidator, PlainSerializer

from checkmaite.core._utils import set_device


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
    if isinstance(v, bytes):
        v = io.BytesIO(v)
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


class TorchvisionModelSpec(pydantic.BaseModel):
    source: Literal["torchvision"] = "torchvision"

    name: str = pydantic.Field(
        default="efficientnet_b0",
        description="Torchvision model name, e.g. 'resnet18', 'efficientnet_b0', 'convnext_tiny'.",
    )

    weights: str = pydantic.Field(
        default="DEFAULT",
        description="Which torchvision weights preset to use. Usually 'DEFAULT'.",
    )


# placeholder for future ModelSpec types
ModelSpec = Annotated[TorchvisionModelSpec, pydantic.Field(discriminator="source")]
