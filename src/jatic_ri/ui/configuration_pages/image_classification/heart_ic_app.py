"""Module for configuring attack settings in HeartApp."""

import os
import sys

import panel as pn
from bokeh.resources import INLINE

from jatic_ri.ui._common.base_app import DEFAULT_STYLING, AppStyling

# Local imports
from jatic_ri.ui._common.heart_app_common import HeartBaseApp


class HeartICApp(HeartBaseApp):
    """App for building Heart Image Classification Panel.

    Parameters
    ----------
    styles : AppStyling, optional
        Styling configuration, by default DEFAULT_STYLING.
    **params : dict[str, object]
        Additional parameters for the `param.Parameterized` base class.

    """

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]) -> None:
        super().__init__(styles, **params)
        self.task = "image_classification"


if __name__ == "__main__":  # pragma: no cover
    sd: HeartICApp = HeartICApp()
    app = sd.panel()
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        os.makedirs("artifacts", exist_ok=True)
        app.save(os.path.join("artifacts", "heart_ic_app.html"), resources=INLINE)
    elif len(sys.argv) > 1:
        msg = f"Got unexpected flag: {sys.argv[1]}"
        sys.stderr.write(msg)
        sys.exit(1)
    else:
        # special adaption to ensure the poetry blocks execution when the server is running
        server = pn.serve(app, address="localhost", port=5008, show=True, threaded=True)
        server.join()
