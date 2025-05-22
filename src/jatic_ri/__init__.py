"""JATIC Reference Implementation package"""

import importlib.metadata as importlib_metadata
import logging
import os
import warnings
from pathlib import Path

__version__ = importlib_metadata.version("jatic_ri")

PACKAGE_DIR = Path(os.path.dirname(__file__)).resolve()


class _CachePath:
    """
    Utility class to globally get and set a cache path.

    Calling an instance of this (`cache_path` below)
    - without arguments, will return the current cache path.
    - with a path, will create the directory and set the path for future invocations.
    """

    def __init__(self, p: str | Path) -> None:
        self.__p: Path
        self._p = p

    @property
    def _p(self) -> Path:
        return self.__p

    @_p.setter
    def _p(self, p: str | Path) -> None:
        self.__p = Path(p).expanduser().resolve()
        self.__p.mkdir(parents=True, exist_ok=True)

    def __call__(self, p: str | Path | None = None) -> Path:
        if p is not None:
            self._p = p
        return self._p


cache_path = _CachePath(Path.home() / ".cache" / "jatic-ri")

# setup loger to print to stdout and to file (`runtime.log`)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)-7.7s] [%(module)s] %(message)s")
root_logger = logging.getLogger()

file_handler = logging.FileHandler(f"{PACKAGE_DIR}/runtime.log")
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/326
# must be set before torch is imported!
user_mps_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is not None
if not user_mps_fallback:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


import torch  # noqa: E402

if not user_mps_fallback and torch.mps.is_available():
    warnings.warn(
        "MPS fallback has been enabled. "
        "Please set the environment variable PYTORCH_ENABLE_MPS_FALLBACK=0 "
        "to prevent CPU fallback.",
        stacklevel=2,
    )
