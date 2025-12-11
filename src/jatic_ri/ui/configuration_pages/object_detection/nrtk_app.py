"""
This module contains the NRTKApp class, which is an implemntation of NRTKBaseApp.
It is able to configure and create NRTKTestStages for consumption

Run with `--ci` flag to save the app as html instead of serving it.
"""

# Python generic imports

import os
import sys

# Panel app imports
import panel as pn
import param
from bokeh.resources import INLINE

from jatic_ri.ui._common.base_app import DEFAULT_STYLING, AppStyling
from jatic_ri.ui._common.nrtk_app_common import NRTKBaseApp


class NRTKAppOD(NRTKBaseApp):
    """App for building NRTKTestStages.

    Attributes
    ----------
    title : param.String
        The title of the application.
    """

    title = param.String(default="Configure Natural Robustness Testing")

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]) -> None:
        super().__init__(styles, **params)

    def _run_export(self) -> None:
        """Collect all configurations into the output_test_stages dictionary.

        This dictionary is shared across app pages. If no configurations are
        found, a status message is emitted.
        """
        if len(self.test_stages) == 0:
            self.status_source.emit("No configurations found. Press Add Test Stage to add a configuration.")
            return
        for idx, stage in enumerate(self.test_stages):
            stage_name = stage["name"]
            self.output_test_stages[f"{self.__class__.__name__}_{idx}"] = {
                "TYPE": "NRTKTestStage",
                "CONFIG": {
                    "name": f"natural_robustness_{stage_name}",
                    "perturber_factory": stage["factory"],
                },
            }


if __name__ == "__main__":  # pragma: no cover
    sd: NRTKAppOD = NRTKAppOD()
    app = sd.panel()
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        os.makedirs("artifacts", exist_ok=True)
        app.save(os.path.join("artifacts", "nrtk_app.html"), resources=INLINE)
    elif len(sys.argv) > 1:
        msg = f"Got unexpected flag: {sys.argv[1]}"
        sys.stderr.write(msg)
        sys.exit(1)
    else:
        # special adaption to ensure the poetry blocks execution when the server is running
        server = pn.serve(app, address="localhost", port=5008, show=True, threaded=True)
        server.join()
