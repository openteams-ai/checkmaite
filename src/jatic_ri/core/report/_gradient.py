"""
DEPRECATED: Gradient-based reporting utilities.
These functions are deprecated and will be removed in a future release.
Use Markdown-based reporting instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

_GRADIENT_INSTALL_HINT = "pip install jatic-ri[unsupported]"


def _missing_gradient_error() -> ImportError:
    return ImportError(
        "Gradient-based reporting is deprecated and is not installed by default.\n\n"
        "Please migrate to Markdown reports (collect_md_report), or install:\n"
        f"  {_GRADIENT_INSTALL_HINT}"
    )


class _MissingGradientMeta(type):
    """Stub types that fail lazily with a helpful error."""

    def __getattr__(cls, _name: str) -> Any:
        raise _missing_gradient_error()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        raise _missing_gradient_error()


# Static typing (pyright/mypy)
if TYPE_CHECKING:  # # pragma: no cover
    from typing import TypeAlias

    HAS_GRADIENT: bool

    class SubText:
        def __init__(self, text: Any = None, **kwargs: Any) -> None: ...

    class Text:
        def __init__(self, text: Any = None, **kwargs: Any) -> None: ...

    class GradientImage:
        def __init__(self, src: str | Path, **kwargs: Any) -> None: ...

    Item: TypeAlias = Any

    class ItemByNarrowText:
        ArgKeys: Any

    class SectionByItem:
        ArgKeys: Any

    class SectionByStackedItems:
        ArgKeys: Any

    class TableText:
        ArgKeys: Any

    class TwoItem:
        ArgKeys: Any

    def create_deck(*args: Any, **kwargs: Any) -> Any: ...  # noqa: ARG001
    def parse_lines(*args: Any, **kwargs: Any) -> Any: ...  # noqa: ARG001

else:
    try:  # pragma: no cover
        from gradient import SubText, create_deck, parse_lines
        from gradient.slide_deck.shapes import Item, Text
        from gradient.slide_deck.shapes.image_shapes import GradientImage
        from gradient.templates_and_layouts.generic_layouts import (
            ItemByNarrowText,
            SectionByItem,
            SectionByStackedItems,
            TableText,
            TwoItem,
        )

        HAS_GRADIENT = True
    except ImportError:
        HAS_GRADIENT = False

        class SubText(metaclass=_MissingGradientMeta): ...

        class Text(metaclass=_MissingGradientMeta): ...

        class Item(metaclass=_MissingGradientMeta): ...

        class GradientImage(metaclass=_MissingGradientMeta): ...

        class ItemByNarrowText(metaclass=_MissingGradientMeta): ...

        class SectionByItem(metaclass=_MissingGradientMeta): ...

        class SectionByStackedItems(metaclass=_MissingGradientMeta): ...

        class TableText(metaclass=_MissingGradientMeta): ...

        class TwoItem(metaclass=_MissingGradientMeta): ...

        def create_deck(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
            raise _missing_gradient_error()

        def parse_lines(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
            raise _missing_gradient_error()


def create_section_by_item_slide(
    deck: str,
    title: str,
    heading: str,
    text: list[str] | list[list[SubText]],
    table: pd.DataFrame | None = None,
    image_path: Path | None = None,
) -> dict[str, Any]:  # pragma: no cover
    """
    Fills in SectionByItem Gradient template with values

    The content slide automatically uses a half-text/half-data layout.
    This template can only display either a table or an image. If both are supplied,
    the function will display only the image. If both a table and image are
    needed, then use the create_text_table_data_slide function.

    Parameters
    ----------
    deck : str
        Title of the gradient deck
    title : str
        Title at the top of the content slide
    heading : str
        Subheading of the content slide
    text : str
        Main text content
    table : pd.DataFrame or None, default None
        DataFrame containing table values to display
    image_path : Path or None, default None
        Path to saved image for display

    Returns
    -------
    dict[str, Any]
        Dictionary with Gradient template arguments
    """
    if table is not None and image_path is not None:
        raise ValueError("Both table and image_path cannot be provided.")

    content = [Text(t, fontsize=16) for t in text]

    template: dict[str, Any] = {
        "deck": deck,
        "layout_name": "SectionByItem",
        "layout_arguments": {
            SectionByItem.ArgKeys.TITLE.value: title,
            # Text arguments
            SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
            SectionByItem.ArgKeys.LINE_SECTION_HALF.value: True,
            SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
        },
    }

    if table is not None:
        template["layout_arguments"].update({SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: table})
    elif image_path is not None:
        template["layout_arguments"].update({SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: image_path})

    return template


def create_section_by_stacked_items_slide(
    deck: str,
    title: str,
    heading: str,
    text: list[str],
    table: pd.DataFrame,
    image_path: Path,
) -> dict[str, Any]:  # pragma: no cover
    """
    Fills in SectionByStackedItems Gradient template with values

    The content slide automatically uses a half-text/half-data layout.
    This template can display both a table and an image. Both a table
    and an image are required for this function. If only a table or
    an image are needed, then use the create_text_data_slide function.

    Parameters
    ----------
    deck : str
        Title of the gradient deck
    title : str
        Title at the top of the content slide
    heading : str
        Subheading of the content slide
    text : str
        Main text content
    table : pd.DataFrame or None, default None
        DataFrame contianing table values to display
    image_path : Path
        Path to saved image for display

    Returns
    -------
    dict[str, Any]
        Dictionary with Gradient template arguments
    """
    content = [Text(t, fontsize=16) for t in text]

    return {
        "deck": deck,
        "layout_name": "SectionByStackedItems",
        "layout_arguments": {
            SectionByStackedItems.ArgKeys.TITLE.value: title,
            # Text arguments
            SectionByStackedItems.ArgKeys.LINE_SECTION_HEADING.value: heading,
            SectionByStackedItems.ArgKeys.LINE_SECTION_BODY.value: content,
            SectionByStackedItems.ArgKeys.LINE_SECTION_HALF.value: True,
            # DataFrame arguments
            SectionByStackedItems.ArgKeys.ITEM_SECTION_TABLE.value: table,
            # Image arguments
            SectionByStackedItems.ArgKeys.ITEM_SECTION_BOTTOM.value: image_path,
        },
    }


def create_table_text_slide(
    deck: str,
    title: str,
    text: Text,
    data: pd.DataFrame,
) -> dict[str, Any]:  # pragma: no cover
    """
    Fills a **TableText** Gradient slide with a block of narrative text and a
    tabular view of data.
    """
    return {
        "deck": deck,
        "layout_name": "TableText",
        "layout_arguments": {
            TableText.ArgKeys.TITLE.value: title,
            TableText.ArgKeys.TEXT.value: text,
            TableText.ArgKeys.TABLE.value: data,
        },
    }


def create_section_by_item_slide_extra_caption(
    deck: str,
    title: str,
    heading: Text,
    content: list[Text],
    body_value: pd.DataFrame | Path,
) -> dict[str, Any]:  # pragma: no cover
    """
    Populates a **SectionByItem** Gradient slide that combines a heading,
    multiline body text, and a single “item” (either a table *or* an image).
    """
    return {
        "deck": deck,
        "layout_name": "SectionByItem",
        "layout_arguments": {
            SectionByItem.ArgKeys.TITLE.value: title,
            SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
            SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
            SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: body_value,
        },
    }


def create_item_by_narrow_text_slide(
    deck: str,
    title: str,
    content: list[Text],
    body_value: pd.DataFrame | Path,
) -> dict[str, Any]:  # pragma: no cover
    """
    Populates a **ItemByNarrowText** Gradient slide that combines a title,
    multiline body text, and a single “item” (either a table *or* an image).
    """
    return {
        "deck": deck,
        "layout_name": "ItemByNarrowText",
        "layout_arguments": {
            ItemByNarrowText.ArgKeys.TITLE.value: title,
            ItemByNarrowText.ArgKeys.TEXT.value: content,
            ItemByNarrowText.ArgKeys.ITEM.value: body_value,
        },
    }


def create_two_item_text_slide(
    deck: str,
    title: str,
    left_item: Item,
    right_item: Item,
) -> dict[str, Any]:  # pragma: no cover
    """
    Populates a **TwoItem** Gradient slide that combines a title,
    and two items.
    """
    return {
        "deck": deck,
        "layout_name": "TwoItem",
        "layout_arguments": {
            TwoItem.ArgKeys.TITLE.value: title,
            TwoItem.ArgKeys.LEFT_ITEM.value: left_item,
            TwoItem.ArgKeys.RIGHT_ITEM.value: right_item,
        },
    }
