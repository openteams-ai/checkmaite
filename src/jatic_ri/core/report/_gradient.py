from pathlib import Path
from typing import Any

import pandas as pd
from gradient.slide_deck.shapes import Item, SubText, Text
from gradient.templates_and_layouts.generic_layouts import (
    ItemByNarrowText,
    SectionByItem,
    TableText,
    TwoItem,
)
from gradient.templates_and_layouts.generic_layouts.section_by_stacked_items import (
    SectionByStackedItems,
)


def create_section_by_item_slide(
    deck: str,
    title: str,
    heading: str,
    text: list[str] | list[list[SubText]],
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


def create_table_text_slide(
    deck: str,
    title: str,
    text: Text,
    data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Fills a **TableText** Gradient slide with a block of narrative text and a
    tabular view of data.

    Parameters
    ----------
    deck : str
        Name of the Gradient deck this slide belongs to.
    title : str
        Title that appears in the slide header.
    text : Text
        Text object (e.g., `gradient.Text`) providing the explanatory
        paragraph(s) that accompany the table.
    data : pd.DataFrame
        DataFrame to render as a table on the slide.

    Returns
    -------
    dict[str, Any]
        Dictionary with Gradient template arguments.
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
) -> dict[str, Any]:
    """
    Populates a **SectionByItem** Gradient slide that combines a heading,
    multiline body text, and a single “item” (either a table *or* an image).

    Parameters
    ----------
    deck : str
        Name of the Gradient deck this slide will live in.
    title : str
        Slide-level title shown in the header.
    heading : Text
        Sub-heading placed above the body text block.
    content : list[Text]
        Sequence of `gradient.Text` objects that form the main narrative body.
    body_value : pd.DataFrame or pathlib.Path
        • If a `pd.DataFrame` is supplied, it is rendered as a table.
        • If a `Path` is supplied, the referenced file is inserted as an
          image (typically PNG/JPEG).

    Returns
    -------
    dict[str, Any]
        Dictionary with Gradient template arguments.
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
) -> dict[str, Any]:
    """
    Populates a **ItemByNarrowText** Gradient slide that combines a title,
    multiline body text, and a single “item” (either a table *or* an image).

    Parameters
    ----------
    deck : str
        Name of the Gradient deck this slide will live in.
    title : str
        Slide-level title shown in the header.
    content : list[Text]
        Sequence of `gradient.Text` objects that form the main narrative body.
    body_value : pd.DataFrame or pathlib.Path
        • If a `pd.DataFrame` is supplied, it is rendered as a table.
        • If a `Path` is supplied, the referenced file is inserted as an
          image (typically PNG/JPEG).

    Returns
    -------
    dict[str, Any]
        Dictionary with Gradient template arguments.
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
) -> dict[str, Any]:
    """
    Populates a **TwoItem** Gradient slide that combines a title,
    and two items.

    Parameters
    ----------
    deck : str
        Name of the Gradient deck this slide will live in.
    title : str
        Slide-level title shown in the header.
    left_item : Item
        Left item to display on the slide.
    right_item : Item
        Right item to display on the slide.

    Returns
    -------
    dict[str, Any]
        Dictionary with Gradient template arguments.
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
