"""Module for configuring attack settings in HeartApp."""

import os
import sys

import panel as pn
from bokeh.resources import INLINE

from checkmaite.ui._common.base_app import DEFAULT_STYLING, AppStyling

# Local imports
from checkmaite.ui._common.heart_app_common import HeartBaseApp


class HeartODApp(HeartBaseApp):
    """App for building Heart Object Detection Panel.

    Parameters
    ----------
    styles : AppStyling, optional
        Styling configuration, by default DEFAULT_STYLING.
    **params : dict[str, object]
        Additional parameters for the `param.Parameterized` base class.

    """

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]) -> None:
        super().__init__(styles, **params)
        self.task = "object_detection"


if __name__ == "__main__":  # pragma: no cover
    sd: HeartODApp = HeartODApp()
    app = sd.panel()
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        os.makedirs("artifacts", exist_ok=True)
        app.save(os.path.join("artifacts", "heart_od_app.html"), resources=INLINE)
    elif len(sys.argv) > 1:
        msg = f"Got unexpected flag: {sys.argv[1]}"
        sys.stderr.write(msg)
        sys.exit(1)
    else:
        # special adaption to ensure the poetry blocks execution when the server is running
        server = pn.serve(app, address="localhost", port=5008, show=True, threaded=True)
        server.join()
