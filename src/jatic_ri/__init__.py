###############################
# PyTorch MPS fallback handling
#
# This block must run before torch
# is imported anywhere in the process.
# Keep it at the top of the module.
###############################
import os
import sys
import warnings


def _configure_torch_mps_fallback() -> None:
    mps_env_var = "PYTORCH_ENABLE_MPS_FALLBACK"

    torch_imported = "torch" in sys.modules
    user_set_fallback = mps_env_var in os.environ

    if not user_set_fallback:
        os.environ[mps_env_var] = "1"

    import torch

    # Only warn if fallback wasn't user-configured *and* MPS is relevant
    if not user_set_fallback and torch.backends.mps.is_available():
        if torch_imported:
            warnings.warn(
                "torch was imported before jatic_ri and "
                f"{mps_env_var} was not set. Changing it now may have no effect. "
                f"Set {mps_env_var} before importing torch/jatic_ri to avoid this warning.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                "Enabled PyTorch MPS CPU fallback by default "
                f"({mps_env_var}=1). "
                f"Set {mps_env_var}=0 to disable CPU fallback.",
                stacklevel=2,
            )


_configure_torch_mps_fallback()

del _configure_torch_mps_fallback

###############################
# library init
###############################

import importlib.metadata as importlib_metadata  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

__version__ = importlib_metadata.version("jatic_ri")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
cache_path.__doc__ = """
Get or set the global cache path used by jatic_ri.

- Called with no arguments, returns the current cache path.
- Called with a path, creates the directory (if needed) and sets it as the new cache path.

Example
-------
>>> from jatic_ri import cache_path
>>> cache_path()
PosixPath('/home/user/.cache/jatic-ri')
>>> cache_path("~/my-cache")
PosixPath('/home/user/my-cache')
"""

from jatic_ri.core import cached_tasks, capability_core, image_classification, object_detection, report  # noqa: E402
from jatic_ri.core._cache import binary_de_serializer  # noqa: E402

__all__ = [
    "__version__",
    "cache_path",
    "image_classification",
    "object_detection",
    "cached_tasks",
    "capability_core",
    "binary_de_serializer",
    "report",
]
