from __future__ import annotations

import importlib
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from maite.protocols import ArrayLike

if TYPE_CHECKING:
    from torch import nn


def load_torchvision_constructor(model_name: str, supported_torchvision_models: dict[str, str]) -> object:
    try:
        return getattr(
            importlib.import_module("torchvision.models.detection"),
            supported_torchvision_models[model_name],
        )
    except Exception as e:
        raise ImportError(
            f"There was an error importing {supported_torchvision_models[model_name]} from torchvision.models.detection"
        ) from e


def set_device(device: str | None | torch.device) -> torch.device:
    """
    Determines the appropriate `torch.device` based on the provided input.

    If `device` is None, it selects the best available option:
    - `"cuda"` if a CUDA-capable GPU is available.
    - `"mps"` if running on macOS with an Apple Metal backend.
    - `"cpu"` otherwise.

    If `device` is provided as a string, it must be a valid PyTorch device identifier
    such as `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, etc.
    For a complete list of valid device strings, see:
    https://pytorch.org/docs/stable/tensor_attributes.html#torch-device
    """

    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def get_index2label_from_model_config(
    config_path: str, model_config: dict[str, Any], index2label_key: str
) -> dict[int, str]:
    if index2label_key not in model_config:
        raise FileNotFoundError(f"The config_file at {config_path} is missing a {index2label_key} key.")
    if isinstance(model_config[index2label_key], list):
        return dict(enumerate(model_config[index2label_key]))
    if isinstance(model_config[index2label_key], dict):
        return {int(key): val for key, val in model_config[index2label_key].items()}
    raise TypeError(f"index2label should be provided as a dict or list, not {type(model_config[index2label_key])}")


def get_default_index2label(
    torchvision_weights_constructor: Any,
) -> dict[int, str]:
    default = torchvision_weights_constructor.DEFAULT
    return dict(enumerate(default.meta["categories"]))


def maybe_download_weights(
    model: Any,
    torchvision_weights_constructor: Any,
    device: torch.device,
    **kwargs: Any,
) -> nn.Module:
    # if weights not already in cache, they are downloaded here
    default = torchvision_weights_constructor.DEFAULT
    return model(weights=default, **kwargs).to(device)


def validate_input_batch(input_batch: Sequence[ArrayLike]) -> None:
    total_channels, orig_img_height, orig_img_width = np.asarray(input_batch[0]).shape
    # channels can be used as a proxy to confirm CHW-ordering
    if not (1 <= total_channels <= 4):
        raise ValueError(
            f"Input data must follow CHW-ordering, current shape: {total_channels, orig_img_height, orig_img_width}"
        )
    for val in input_batch:
        # required to convert to array to appease type-checker...
        npy_array = np.asarray(val)  # creates view, not copy
        if npy_array.shape != (total_channels, orig_img_height, orig_img_width):
            raise ValueError(
                f"All input images currently required to have identical shape, {npy_array.shape} "
                f"not equal to {(total_channels, orig_img_height, orig_img_width)}. Please "
                "contact the RI-team if your use-case requires unevenly shaped images."
            )


def to_torch_batch(input_batch: Sequence[ArrayLike], device: torch.device) -> torch.Tensor:
    # we are not writing to the underlying array in this method and hence we suppress this warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors.",
            category=UserWarning,
        )
        return torch.stack([torch.as_tensor(obj, device=device) for obj in input_batch])
