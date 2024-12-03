"""Module for vendoring types from newer MAITE releases that may not yet be supported by all JATIC packages."""

from __future__ import annotations

from typing import TypedDict

from typing_extensions import NotRequired, ReadOnly, Required


# This code should be removed once MAITE 0.7.x is allowed.
class DatasetMetadata(TypedDict):
    """
    Metadata associated with a Dataset object.

    Attributes
    ----------
    id : str
        Identifier for a single Dataset instance
    index2label : NotRequired[ReadOnly[dict[int, str]]]
        Mapping from integer labels to corresponding string descriptions
    """

    id: Required[ReadOnly[str]]
    index2label: NotRequired[ReadOnly[dict[int, str]]]


# This code should be removed once MAITE 0.7.x is allowed.
class DatumMetadata(TypedDict):
    """
    Metadata associated with a single datum.

    Attributes
    ----------
    id : int|str
        Identifier for a single datum
    """

    id: Required[ReadOnly[int | str]]
