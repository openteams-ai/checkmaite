from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Annotated, Any

import matplotlib.figure
import pandas as pd
import PIL.Image
import pyspark.sql
import torch
from pydantic import BeforeValidator, PlainSerializer

from jatic_ri._common.models import set_device


def _to_image(v: Any) -> Any:
    if isinstance(v, (str, Path, io.BufferedIOBase)):
        with contextlib.suppress(Exception):
            v = PIL.Image.open(v)  # type: ignore[reportArgumentType]
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
