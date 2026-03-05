"""Module for Object Detection Survivor panel app.

Run with ``--ci`` flag to save the app as html instead of serving it.
"""

import os
import re
import sys
from typing import Any

import panel as pn
import param
from bokeh.resources import INLINE

from checkmaite.ui._common.base_app import DEFAULT_STYLING, AppStyling, BaseApp


class SurvivorAppOD(BaseApp):
    """Survivor panel app.

    Creates a GUI interface that allows the user to enter and export
    SurvivorConfig values.

    Attributes
    ----------
    title : param.String
        Settings pane title.
    otb_threshold : param.Number
        Threshold for on the bubble data. Defaults to 0.5.
    easy_hard_threshold : param.Number
        Threshold for easy/hard data. Defaults to 0.5.
    similarity_strategy : param.ObjectSelector
        The behavior to convert the scores (currently in data scoring).
        Default: "Exact"
    round_precision : param.Integer
        Precision for rounding. Defaults to 2.
    bins : param.String
        Bins to use when `similarity_strategy` is set to "Binned".
        Defaults to "0, 0.25, 0.5, 1.0"
    """

    title: param.String = param.String(default="Survivor - Evaluate the impact of images on T&E results")

    otb_threshold: param.Number = param.Number(default=0.5, bounds=(0, 1.0), label="On The Bubble Threshold")
    easy_hard_threshold: param.Number = param.Number(default=0.5, label="Easy/Hard Threshold")
    similarity_strategy: param.ObjectSelector = param.ObjectSelector(
        objects=["Exact", "Rounded", "Binned"],
        default="Exact",
        label="Similarity Strategy",
    )
    round_precision: param.Integer = param.Integer(default=2, label="Decimal Places", bounds=(0, 10))
    bins: param.String = param.String(default="0, 0.25, 0.5, 1.0", label="Bins")
    # Background parsed bins string gets updated every time there is a new input
    _bins: int | list[float] = [0, 0.25, 0.5, 1.0]

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, Any]) -> None:
        super().__init__(styles, **params)

    @param.depends("bins", watch=True)
    def _parse_bin_string(self) -> None:
        """Parse ``self.bins`` and set ``self._bins`` to output.

        Will look for either a single integer or a comma separated list of
        integers/floats.
        """
        just_number = re.compile(r"^\d+$")
        list_of_numbers = re.compile(r"^(\d+(\.\d*)?,(\s)?)*(\d+(\.\d*)?)$")
        self.status_source.emit("Waiting for input...")
        if just_number.match(self.bins):
            self._bins = int(self.bins)
        elif list_of_numbers.match(self.bins):
            self._bins = [float(bin_edge) for bin_edge in self.bins.split(",")]
        else:
            self.status_source.emit("Invalid Bins argument")

    def _run_export(self) -> None:
        """Export a dictionary representation of the SurvivorConfig.

        The configuration is entered by the user. The output is exported to the
        ``self.output_tests_stages`` dictionary under the
        "survivor_test_stage" key.
        """
        conversion_type_mapping = {
            "Exact": "original",
            "Binned": "binned",
            "Rounded": "rounded",
        }
        conversion_args = {}

        # Only add arg if "Rounded"
        if self.similarity_strategy == "Rounded":
            conversion_args["decimals_to_round"] = self.round_precision

        # Only add arg if "Binned"
        if self.similarity_strategy == "Binned":
            self._parse_bin_string()  # self._bins will only be set to a list or int
            if isinstance(self._bins, list):
                conversion_args["bin_edges"] = self._bins
            else:
                conversion_args["num_bins"] = self._bins

        config = {
            "metric_column": "metric",
            "conversion_type": conversion_type_mapping[self.similarity_strategy],
            "otb_threshold": self.otb_threshold,
            "easy_hard_threshold": self.easy_hard_threshold,
        }
        if conversion_args:
            config.update({"conversion_args": conversion_args})

        self.output_test_stages["survivor_test_stage"] = {"TYPE": "SurvivorTestStage", "CONFIG": config}

    def settings_pane(self) -> pn.Column:
        """Return the settings pane view.

        Returns
        -------
        pn.Column
            The settings pane.
        """
        return pn.Column(
            self.view_title,
            pn.widgets.FloatInput.from_param(
                self.param.otb_threshold,
                width=self.styles.widget_width,
                styles=self.styles.style_text_body1,
                description="Upper threshold of model agreement for data to be considered 'On the Bubble'.",
                format="0.00",
            ),
            pn.widgets.FloatInput.from_param(
                self.param.easy_hard_threshold,
                width=self.styles.widget_width,
                styles=self.styles.style_text_body1,
                description="Threshold of model score for data to be considered 'Easy' or 'Hard'.",
                format="0.00",
            ),
            pn.widgets.Select.from_param(
                self.param.similarity_strategy,
                name="Similarity Strategy",
                width=self.styles.widget_width,
                styles=self.styles.style_text_body1,
                description="Strategy to use to discretize model metrics.",
            ),
            width=self.styles.app_width,
        )

    @param.depends("similarity_strategy")
    def similarity_option_pane(self) -> pn.Row | pn.viewable.Viewable:
        """Return the view of optional similarity strategy options.

        This view changes with changes to similarity threshold.

        Returns
        -------
        pn.Row | pn.viewable.Viewable
            The similarity option pane.
        """
        # if binned is selected, display the bins widget
        if self.similarity_strategy == "Binned":
            self._parse_bin_string()
            return pn.widgets.TextInput.from_param(
                self.param.bins,
                width=self.styles.widget_width,
                styles=self.styles.style_text_body1,
                margin=(0, 40),  # (vert, horiz) margins for visual offset,
                description="Edges of the bins to sort model metrics into. "
                "Should all be within metric range i.e (0-1), (1-100)",
            )
        self.status_source.emit("Waiting for input...")
        # if rounded is selected, display the round precision widget
        if self.similarity_strategy == "Rounded":
            return pn.widgets.IntInput.from_param(
                self.param.round_precision,
                width=self.styles.widget_width,
                styles=self.styles.style_text_body1,
                margin=(0, 40),  # (vert, horiz) margins for visual offset
                description="Number of decimal places to round model metrics to.",
            )
        return pn.Row()  # i.e. empty element

    def panel(self) -> pn.Column:
        """Return the high-level panel app.

        Returns
        -------
        pn.Column
            The main panel application column.
        """
        return pn.Column(
            self.view_header,
            self.settings_pane,
            self.similarity_option_pane,
            self.view_status_bar,
            pn.Spacer(height=24),
            pn.Row(pn.HSpacer(), self.next_button),
            width=self.styles.app_width,
            styles={"background": self.styles.color_main_bg},
        )


if __name__ == "__main__":  # pragma: no cover
    sd: SurvivorAppOD = SurvivorAppOD()
    app = sd.panel()
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        os.makedirs("artifacts", exist_ok=True)
        app.save(os.path.join("artifacts", "survivor_app.html"), resources=INLINE)
    elif len(sys.argv) > 1:
        msg = f"Got unexpected flag: {sys.argv[1]}"
        sys.stderr.write(msg)
        sys.exit(1)
    else:
        # special adaption to ensure the poetry blocks execution when the server is running
        server = pn.serve(app, address="localhost", port=5008, show=True, threaded=True)
        server.join()
