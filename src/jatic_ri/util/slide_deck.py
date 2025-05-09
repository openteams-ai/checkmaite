from __future__ import annotations  # noqa: D100

from pathlib import Path
from typing import Any

import pandas as pd
from gradient.slide_deck.shapes import Text
from gradient.templates_and_layouts.generic_layouts.section_by_item import SectionByItem
from gradient.templates_and_layouts.generic_layouts.section_by_stacked_items import SectionByStackedItems


def create_section_by_item_slide(
    deck: str,
    title: str,
    heading: str,
    text: list[str],
    table: pd.DataFrame | None = None,
    image_path: Path | None = None,
) -> dict[str, Any]:
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
    if all([table is not None, image_path is not None]):
        raise ValueError("Both table and image_path cannot be provided.")

    content = [Text(t, fontsize=16) for t in text]

    template = {
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
) -> dict[str, Any]:
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
