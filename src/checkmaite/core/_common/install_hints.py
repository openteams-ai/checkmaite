"""Point modelmaite's optional-dependency errors at checkmaite's extras."""

import re
from collections.abc import Iterator
from contextlib import contextmanager

# modelmaite's install hints name its own extras; checkmaite forwards these
# under the same names, so users of checkmaite should install checkmaite's.
_MODELMAITE_EXTRA = re.compile(r"modelmaite\[(visdrone|onnx|onnx-cuda)\]")


@contextmanager
def checkmaite_install_hints() -> Iterator[None]:
    """Re-raise modelmaite's missing-extra ImportErrors with a ``checkmaite[...]`` hint."""
    try:
        yield
    except ImportError as err:
        message = str(err)
        rewritten = _MODELMAITE_EXTRA.sub(r"checkmaite[\1]", message)
        if rewritten == message:
            raise
        raise ImportError(rewritten) from err
