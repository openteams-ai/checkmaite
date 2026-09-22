import functools
import hashlib
import importlib
import json
import logging
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import torch

if TYPE_CHECKING:
    from torch import nn

P = ParamSpec("P")
R = TypeVar("R")

CHECKMAITE_PLUGINS_UNSUPPORTED_INSTALL_HINT = (
    "pip install 'checkmaite-plugins[unsupported] @ "
    "git+https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite-plugins.git@main'"
)


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
                    f"'{func.__qualname__}' requires optional dependency '{module_name}', which is not installed.{hint}"
                )
                raise ImportError(msg) from None
            return func(*args, **kwargs)

        return wrapper

    return deco
