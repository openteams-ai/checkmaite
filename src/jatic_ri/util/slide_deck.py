from __future__ import annotations  # noqa: D100

from typing import Any

import pandas as pd
from gradient.slide_deck.shapes import Text
from gradient.templates_and_layouts.generic_layouts.text_data import TextData


def create_text_data_slide(
    deck: str,
    title: str,
    heading: str,
    text: list[str],
    table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fills in TextData Gradient template with values

    The content slide automatically uses a half-text/half-table layout

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

    Returns
    -------
    dict[str, Any]
        Dictionary with Gradient template arguments
    """

    content = [Text(t, fontsize=16) for t in text]

    template = {
        "deck": deck,
        "layout_name": "TextData",
        "layout_arguments": {
            TextData.ArgKeys.TITLE.value: title,
            # Text arguments
            TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
            TextData.ArgKeys.TEXT_COLUMN_HALF.value: True,
            TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
        },
    }

    if table is not None:
        template["layout_arguments"].update({TextData.ArgKeys.DATA_COLUMN_TABLE.value: table})

    return template
