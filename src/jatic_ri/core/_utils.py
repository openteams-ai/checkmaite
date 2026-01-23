import functools
import hashlib
import importlib
import json
import logging
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import numpy as np
import torch
from maite.protocols import ArrayLike

if TYPE_CHECKING:
    from torch import nn

P = ParamSpec("P")
R = TypeVar("R")


class MissingDependencyWarning(UserWarning):
    """Optional dependency missing for a code path."""


class CountAndDrop(logging.Filter):
    def __init__(self, predicate: Callable[[logging.LogRecord], bool]) -> None:
        super().__init__()
        self.predicate = predicate
        self.count = 0
        self.first = None

    def filter(self, record: logging.LogRecord) -> bool:
        if self.predicate(record):
            self.count += 1
            if self.first is None:
                self.first = record.getMessage()
            return False
        return True


@contextmanager
def squash_repeated_warnings(logger_prefix: str, match: Callable[[logging.LogRecord], bool]) -> Iterator[CountAndDrop]:
    """
    Temporarily suppress repeated warnings emitted under `logger_prefix` (e.g. "dataeval"),
    counting how many were suppressed and a sample message that was emitted.
    """
    lg = logging.getLogger(logger_prefix)
    filt = CountAndDrop(match)

    lg.addFilter(filt)

    try:
        yield filt

    finally:
        lg.removeFilter(filt)


def set_device(device: str | None | torch.device) -> torch.device:
    """Determine the appropriate `torch.device` based on the provided input.

    If `device` is None, it selects the best available option: "cuda" if a
    CUDA-capable GPU is available, "mps" if running on macOS with an Apple
    Metal backend, or "cpu" otherwise. If `device` is provided as a string,
    it must be a valid PyTorch device identifier such as "cpu", "cuda",
    "cuda:0", "mps", etc. For a complete list of valid device strings, see:
    https://pytorch.org/docs/stable/tensor_attributes.html#torch-device

    Parameters
    ----------
    device : str or None or torch.device
        The device to use. Can be a string (e.g., "cuda", "cpu"),
        a `torch.device` object, or None to auto-detect.

    Returns
    -------
    torch.device
        The selected `torch.device` object.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def get_index2label_from_model_config(
    config_path: str | Path, model_config: dict[str, Any], index2label_key: str
) -> dict[int, str]:
    """Extract index-to-label mapping from a model configuration.

    Parameters
    ----------
    config_path : str
        Path to the configuration file (used for error messages).
    model_config : dict[str, Any]
        The model configuration dictionary.
    index2label_key : str
        The key in `model_config` that holds the index-to-label mapping.

    Returns
    -------
    dict[int, str]
        A dictionary mapping class indices to label names.

    Raises
    ------
    FileNotFoundError
        If `index2label_key` is not found in `model_config`.
    TypeError
        If the value associated with `index2label_key` is not a list or
        dict.
    """
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
    """Get the default index-to-label mapping from TorchVision weights.

    Parameters
    ----------
    torchvision_weights_constructor : Any
        The TorchVision weights constructor object (e.g.,
        `FasterRCNN_ResNet50_FPN_Weights`).

    Returns
    -------
    dict[int, str]
        A dictionary mapping class indices to label names.
    """
    default = torchvision_weights_constructor.DEFAULT
    return dict(enumerate(default.meta["categories"]))


def maybe_download_weights(
    model: Any,
    torchvision_weights_constructor: Any,
    device: torch.device,
    **kwargs: Any,
) -> "nn.Module":
    """Load a model with default TorchVision weights, downloading if necessary.

    Parameters
    ----------
    model : Any
        The model class (e.g.,
        `torchvision.models.detection.fasterrcnn_resnet50_fpn`).
    torchvision_weights_constructor : Any
        The TorchVision weights constructor (e.g.,
        `FasterRCNN_ResNet50_FPN_Weights`).
    device : torch.device
        The device to move the model to.
    **kwargs : Any
        Additional keyword arguments to pass to the model constructor.

    Returns
    -------
    nn.Module
        The instantiated model with loaded weights, moved to the specified
        device.
    """
    # if weights not already in cache, they are downloaded here
    default = torchvision_weights_constructor.DEFAULT
    return model(weights=default, **kwargs).to(device)


def validate_input_batch(input_batch: Sequence[ArrayLike]) -> None:
    """Validate the format and consistency of an input batch of images.

    Checks for CHW ordering and consistent shapes across images in the batch.

    Parameters
    ----------
    input_batch : Sequence[ArrayLike]
        A sequence of image-like arrays.

    Raises
    ------
    ValueError
        If input data is not CHW-ordered or if images in the batch have
        inconsistent shapes.
    """
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
    """Convert a sequence of array-like images to a PyTorch tensor batch.

    Parameters
    ----------
    input_batch : Sequence[ArrayLike]
        A sequence of image-like arrays.
    device : torch.device
        The device to move the resulting tensor to.

    Returns
    -------
    torch.Tensor
        A PyTorch tensor representing the batch of images.
    """
    # we are not writing to the underlying array in this method and hence we
    # suppress this warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors.",
            category=UserWarning,
        )
        return torch.stack([torch.as_tensor(obj, device=device) for obj in input_batch])


def id_hash(**kwargs: Any) -> str:
    """Generate a consistent hash from keyword arguments.

    Parameters
    ----------
    **kwargs : Any
        Key-value pairs to include in the hash generation

    Returns
    -------
    str
        First 8 characters of the SHA-256 hash of the JSON-serialized kwargs
    """
    return hashlib.sha256(json.dumps(kwargs, default=str, sort_keys=True).encode()).hexdigest()[:8]


def deprecated(*, replacement: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function as deprecated, emitting a DeprecationWarning on call."""

    def deco(func: Callable[P, R]) -> Callable[P, R]:
        msg = f"'{func.__qualname__}' is deprecated."
        if replacement:
            msg += f" Use '{replacement}' instead."

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return deco


def requires_optional_dependency(
    module_name: str,
    *,
    install_hint: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Require an optional dependency; if missing, raise ImportError with an install hint."""

    def deco(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                importlib.import_module(module_name)
            except ImportError:
                hint = f"\nInstall: {install_hint}" if install_hint else ""
                msg = (
                    f"'{func.__qualname__}' requires optional dependency '{module_name}', "
                    "which is not installed."
                    f"{hint}"
                )
                raise ImportError(msg) from None
            return func(*args, **kwargs)

        return wrapper

    return deco
