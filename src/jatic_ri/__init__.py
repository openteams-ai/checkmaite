"""JATIC Reference Implementation package"""

import importlib.metadata as importlib_metadata
import logging
import os
from pathlib import Path

__version__ = importlib_metadata.version("jatic_ri")

PACKAGE_DIR = Path(os.path.dirname(__file__)).resolve()
CACHE_DIR = Path(os.path.dirname(__file__)).resolve()

DEFAULT_CACHE_ROOT = ".tscache"

# setup loger to print to stdout and to file (`runtime.log`)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)-7.7s] [%(module)s] %(message)s")
root_logger = logging.getLogger()

file_handler = logging.FileHandler(f"{PACKAGE_DIR}/runtime.log")
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)
