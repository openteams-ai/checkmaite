"""Shared helpers for the datamaite-backed dataset factories.

Two policies live here because they must not drift between the object-detection
and image-classification factories:

* **Path anchoring.** CheckMAITE's loader API has always interpreted an
  ``ann_file``/``ann_dir`` override relative to the caller's working directory.
  datamaite deliberately anchors a relative override to the *dataset root*
  instead. Local overrides are therefore resolved against the working directory
  and forwarded as ``file://`` URIs, which datamaite always reads from the local
  filesystem -- a bare absolute path would be looked up on a remote root's
  backend instead. Remote overrides pass through untouched, and path objects
  are never stringified, so their filesystem options (credentials, endpoints)
  reach datamaite intact.
* **Fail-loud loading (best effort).** datamaite's loaders are best-effort by
  contract: a source record they cannot parse is dropped with a warning and
  loading continues. CheckMAITE is a T&E tool, so silently converting a
  malformed annotation into background can corrupt metrics. Recognized
  datamaite row-rejection warnings emitted while a dataset loads are collected
  and promoted to a typed error. This matches datamaite's human-readable log
  text, so it is a best-effort guard rather than a guarantee of complete source
  integrity: a loss datamaite reports in a new phrasing, or does not report at
  all, still gets through. A structured diagnostics API upstream is tracked
  separately.
"""

import logging
import re
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from upath import UPath

__all__ = [
    "DATASET_PROVENANCE_METADATA_KEYS",
    "DatasetSourceError",
    "collect_source_warnings",
    "datamaite_root",
    "enforce_source_integrity",
    "is_remote",
    "to_caller_anchored_path",
]

# datamaite's loaders all log through this logger tree.
_DATAMAITE_LOGGER = "datamaite"

# Warnings that report lost source data, matched against datamaite 0.5.0's
# message text. Most row rejections start with "Skipping" (VisDrone's orphan
# annotation trails it instead); whole-file and batch losses use "Dropping",
# "Could not read", or "Could not list"; and two paths keep the record but
# strip its ground truth.
_REJECTION_PATTERNS = (
    re.compile(r"^Skipping\b"),
    re.compile(r"; skipping$"),
    re.compile(r"^Dropping\b"),
    re.compile(r"^Could not (?:read|list)\b"),
    re.compile(r"; it will carry no label$"),
    re.compile(r"; leaving them unlabelled\b"),
)

# ...except these, which report a deliberate policy rather than unusable data:
# traversal guards drop files that are not part of the dataset, and a duplicate
# id is recovered from deterministically ("keeping the first").
_TOLERATED_PATTERNS = (
    re.compile(r"^Skipping symlinked\b"),
    re.compile(r"^Skipping image escaping\b"),
    re.compile(r"^Skipping duplicate\b"),
)

# Upper bound on how many individual rejections are quoted in the error; the
# count is always exact.
_MAX_REPORTED = 10


# datamaite surfaces every source attribute it preserved as datum metadata,
# including its own parsing provenance, and dataeval expands a list-valued datum
# metadata key into a per-detection bias factor. Provenance is useful for tracing
# a datum back to its source row, but it is not a property of the data under
# test: ``yolo_bbox`` is a re-encoding of the target itself and ``source_line``
# is a file offset, so neither is a meaningful bias factor. Genuine annotation
# attributes (VisDrone ``truncation``/``occlusion``/``visdrone_score``, COCO
# ``annotations[]`` extras) are deliberately *not* in this set — they are exactly
# the per-object factors this migration set out to surface.
DATASET_PROVENANCE_METADATA_KEYS = (
    "annotation_file",
    "label_file",
    "source_file_name",
    "source_format",
    "source_line",
    "variant",
    "yolo_bbox",
)


class DatasetSourceError(ValueError):
    """datamaite rejected source records while loading a dataset.

    Raised instead of returning a dataset whose ground truth is quietly
    incomplete. The message aggregates datamaite's own diagnostics, which carry
    ``file:line`` for row-oriented formats (YOLO, VisDrone).
    """


def is_remote(path: "str | Path | UPath") -> bool:
    """True for URL-shaped paths (``s3://``, ``gs://``, ``memory://``, ...)."""
    return "://" in str(path)


def datamaite_root(root: "str | Path | UPath") -> "str | Path":
    """Pass a dataset root to datamaite unchanged, whatever its type.

    datamaite's loaders accept a configured ``UPath`` at runtime and keep its
    filesystem options, but annotate ``root`` as ``str | Path``; under
    universal-pathlib >= 0.3 a remote ``UPath`` is not a ``pathlib.Path``
    subclass. The cast only reconciles the annotation. Converting the value
    (for example with ``str()``) would drop the options this exists to keep.
    """
    return cast("str | Path", root)


def to_caller_anchored_path(path: "str | Path | UPath") -> "str | Path | UPath":
    """Anchor a local override to the caller's working directory.

    A local override comes back as a ``file://`` URI of its absolute path.
    datamaite resolves ``file://`` to the local filesystem whatever the dataset
    root's backend, whereas a bare absolute path is looked up on a remote root's
    backend -- a local ``ann_dir`` under a remote root would silently match no
    labels at all.

    Remote overrides are returned as given: a remote path object keeps its own
    filesystem options, and a remote URL string inherits the dataset root's.
    """
    if not isinstance(path, str) and is_remote(path):
        return path
    if is_remote(path):
        return str(path)
    return Path(str(path)).expanduser().resolve().as_uri()


@contextmanager
def collect_source_warnings() -> Iterator[list[str]]:
    """Collect the warnings datamaite logs while loading inside the block.

    The handler is attached for the duration of the block only, and the
    logger's own level and handlers are restored afterwards, so this never
    changes what an application sees on its own logging configuration.
    """
    messages: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Collector(level=logging.WARNING)
    logger = logging.getLogger(_DATAMAITE_LOGGER)
    previous_level, previous_disabled = logger.level, logger.disabled
    logger.addHandler(handler)
    # A quieter application logging config must not turn a fail-loud load back
    # into a silent one.
    if not logger.isEnabledFor(logging.WARNING):
        logger.setLevel(logging.WARNING)
    logger.disabled = False
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.disabled = previous_disabled


def _rejections(messages: Sequence[str]) -> list[str]:
    return [
        message
        for message in messages
        if any(pattern.search(message) for pattern in _REJECTION_PATTERNS)
        and not any(pattern.search(message) for pattern in _TOLERATED_PATTERNS)
    ]


def enforce_source_integrity(messages: Sequence[str], *, source: str, strict: bool) -> None:
    """Fail (or warn) when datamaite dropped source records during a load.

    Args:
        messages: Warnings collected by :func:`collect_source_warnings`.
        source: What was being loaded, for the diagnostic.
        strict: Raise when records were rejected. When ``False`` the same
            diagnostic is emitted as a :class:`UserWarning` and the partially
            parsed dataset is returned.

    Raises:
        DatasetSourceError: If ``strict`` and any source record was rejected.
    """
    rejected = _rejections(messages)
    if not rejected:
        return

    quoted = rejected[:_MAX_REPORTED]
    detail = "\n".join(f"  - {message}" for message in quoted)
    if len(rejected) > len(quoted):
        detail += f"\n  - ... and {len(rejected) - len(quoted)} more"
    summary = (
        f"{len(rejected)} source record(s) were rejected while loading {source}. "
        f"Loading them as background would silently corrupt evaluation metrics:\n{detail}"
    )
    if not strict:
        warnings.warn(summary, stacklevel=3)
        return
    raise DatasetSourceError(
        f"{summary}\nFix the dataset, or pass strict_annotations=False to load the parseable records anyway."
    )
