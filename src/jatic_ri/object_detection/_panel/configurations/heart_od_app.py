"""Module for configuring attack settings in HeartApp."""

import os
import sys

import panel as pn
from bokeh.resources import INLINE

from jatic_ri._common._panel.configurations.base_app import DEFAULT_STYLING, AppStyling

# Local imports
from jatic_ri._common._panel.configurations.heart_app_common import HeartBaseApp


class HeartODApp(HeartBaseApp):
    """App for building Heart Object Detection Panel"""

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]) -> None:
        super().__init__(styles, **params)
        self.task = "object_detection"


if __name__ == "__main__":  # pragma: no cover
    sd: HeartODApp = HeartODApp()
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        os.makedirs("artifacts", exist_ok=True)
        sd.panel().save(os.path.join("artifacts", "heart_od_app.html"), resources=INLINE)
    elif len(sys.argv) > 1:
        msg = f"Got unexpected flag: {sys.argv[1]}"
        sys.stderr.write(msg)
        sys.exit(1)
    else:
        pn.serve(sd.panel(), host="127.0.0.1", port=5008)
