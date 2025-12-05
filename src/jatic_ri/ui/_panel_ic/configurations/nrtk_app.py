"""NRTKApp for Image Classification.

This module contains the NRTKAppIC class, an implementation of NRTKBaseApp
for image classification. It configures and creates NRTKTestStages.

Run with the ``--ci`` flag to save the app as HTML instead of serving it.
"""

# Python generic imports

import os
import sys

# Panel app imports
import param
from bokeh.resources import INLINE

from jatic_ri.ui._panel._common.base_app import DEFAULT_STYLING, AppStyling
from jatic_ri.ui._panel._common.nrtk_app_common import NRTKBaseApp


class NRTKAppIC(NRTKBaseApp):
    """App for building NRTKTestStages for image classification.

    Parameters
    ----------
    styles : AppStyling, optional
        Styling configuration, by default DEFAULT_STYLING.
    **params : dict[str, object]
        Additional parameters for the `param.Parameterized` base class.

    Attributes
    ----------
    title : param.String
        The title of the application.
    """

    title = param.String(default="Configure Natural Robustness Testing")

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]) -> None:
        super().__init__(styles, **params)

    def _run_export(self) -> None:
        """Collect all configurations into a dictionary.

        This dictionary is shared across application pages.
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
    import panel as pn

    sd: NRTKAppIC = NRTKAppIC()
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        os.makedirs("artifacts", exist_ok=True)
        sd.panel().save(os.path.join("artifacts", "nrtk_app.html"), resources=INLINE)
    elif len(sys.argv) > 1:
        msg = f"Got unexpected flag: {sys.argv[1]}"
        sys.stderr(msg)
        sys.exit(1)
    else:
        pn.serve(sd.panel(), host="127.0.0.1", port=5008)
