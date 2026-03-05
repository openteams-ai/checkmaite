"""Survivor Image Classification Panel App.

This file is just a pointer to the OD Survivor panel, as the two workflows are identical, and we
would like to reduce duplicate code.
"""

from typing import Any

from checkmaite.ui._common.base_app import DEFAULT_STYLING, AppStyling
from checkmaite.ui.configuration_pages.object_detection.survivor_app import SurvivorAppOD

__all__ = ["SurvivorAppIC"]


class SurvivorAppIC(SurvivorAppOD):
    """Survivor Image Classification Panel App."""

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, Any]) -> None:
        super().__init__(styles, **params)
