"""JATIC Reference Implementation package"""

import importlib.metadata as importlib_metadata
import os
from pathlib import Path

__version__ = importlib_metadata.version("jatic_ri")

PACKAGE_DIR = Path(os.path.dirname(__file__)).resolve()
CACHE_DIR = Path(os.path.dirname(__file__)).resolve()

DEFAULT_CACHE_ROOT = ".tscache"
