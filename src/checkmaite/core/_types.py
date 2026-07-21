import contextlib
import io
from pathlib import Path
from typing import Annotated, Any, Literal

import matplotlib.figure
import pandas as pd
import PIL.Image
import pydantic
import torch
from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema

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


DataFrame = pd.DataFrame

Device = Annotated[
    torch.device,
    BeforeValidator(set_device),
    PlainSerializer(str),
    WithJsonSchema(
        {
            "type": "string",
            "description": "Torch device string such as 'cpu', 'cuda', 'cuda:0', or 'mps'.",
            "examples": ["cpu", "cuda", "mps"],
        }
    ),
]


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
