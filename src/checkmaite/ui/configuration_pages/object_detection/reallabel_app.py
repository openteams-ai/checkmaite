"""Module for Object Detection RealLabel panel app.

Notes
-----
Run with ``--ci`` flag to save the app as html instead of serving it.

"""

import os
import sys
from typing import Any

import panel as pn
import param
import reallabel
from bokeh.resources import INLINE

from checkmaite.ui._common.base_app import DEFAULT_STYLING, AppStyling, BaseApp


class RealLabelApp(BaseApp):
    """RealLabel panel App.

    Creates a GUI interface that allows the user to enter and export
    RealLabelConfig values.

    In this app, Reallabel is not doing any confidence calibration. Instead we
    tell RealLabel that the "score" column should be treated as pre-calibrated
    by giving that column name to `RealLabelConfig.calibrated_confidence_column`.
    This is in alignment with reallabel.MAITERealLabel as it populates MAITE
    ODT scores into the `score` column by default.

    Parameters
    ----------
    title : param.String
        Settings pane title.
    iou_threshold : param.Number
        Threshold for IoU calculation. Defaults to 0.5.
    likely_missed_min_confidence : param.Number
        The aggregated confidence value above which a model ensemble
        ground truth disagreement will be interpreted as a potentially missing
        label. Defaults to 0.5.
    likely_wrong_max_confidence : param.Number
        The aggregated confidence value below which a model ensemble
        ground truth disagreement will be interpreted as a likely either
        extraneous or incorrect. Defaults to 0.5.
    class_agnostic : param.Boolean
        Set to True for when you only care about the LOCATIONS of
        bounding boxes (and not their labels). Defaults to True.
    run_with_ground_truth : param.Boolean
        Set to False when you want to run RealLabel on an unlabeled dataset.
        Defaults to True.
    styles : AppStyling, optional
        Styling configuration, by default DEFAULT_STYLING.
    **params : dict[str, Any]
        Additional parameters for the `param.Parameterized` base class.

    """

    title: param.String = param.String(default="RealLabel - Identify potential errors in ground truth")

    iou_threshold: param.Number = param.Number(default=0.5, bounds=(0, 1), label="IoU Threshold")
    likely_missed_min_confidence: param.Number = param.Number(
        default=0.5, bounds=(0, 1), label="Minimum Confidence Threshold"
    )
    likely_wrong_max_confidence: param.Number = param.Number(
        default=0.5, bounds=(0, 1), label="Maximum Confidence Threshold"
    )
    class_agnostic: param.Boolean = param.Boolean(default=True, label="Class Agnostic")
    run_with_ground_truth: param.Boolean = param.Boolean(default=True, label="Run with Ground Truth")

    # special parameter for dynamically setting the next stage
    # in this case, its set by output from previous stage
    next_parameter = param.Selector(
        default="Configure SurvivorOD", objects=["Configure SurvivorOD", "DatasetAnalysisDashboard"]
    )

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, Any]) -> None:
        super().__init__(styles, **params)

    def _run_export(self) -> None:
        """Export RealLabelConfig.

        Exports a dictionary representation of the RealLabelConfig entered
        by the user. The output is exported to the
        `self.output_tests_stages` dictionary under the
        "reallabel_test_stage" key.
        """

        # Just giving the option to call any RealLabel output
        additional_outputs = [
            # VERBOSE_OUTPUT throws an error in 0.5.0.  Try again after updating.
            e.value
            for e in reallabel.RealLabelOutput
            if e not in {reallabel.RealLabelOutput.VERBOSE_OUTPUT}
        ]

        self.output_test_stages["reallabel_test_stage"] = {
            "TYPE": "RealLabelTestStage",
            "CONFIG": {
                "additional_outputs": additional_outputs,
                "run_with_ground_truth": self.run_with_ground_truth,
                "deduplication_iou_threshold": self.iou_threshold,
                "threshold_max_aggregated_confidence_fp": self.likely_wrong_max_confidence,
                "threshold_min_aggregated_confidence_fn": self.likely_missed_min_confidence,
                "use_thresholds": True,
                "class_agnostic": self.class_agnostic,
                "run_confidence_calibration": False,
                "column_names": {
                    "unique_identifier_columns": [
                        "id"
                    ],  # assumes individual image metadata contains the key "id", which is a maite requirement
                    "calibrated_confidence_column": "score",  # see note in class doc string
                },
            },
        }

    def settings_pane(self) -> pn.Column:
        """Return the settings pane view.

        Returns
        -------
        pn.Column
            The settings pane view.
        """
        return pn.Column(
            self.view_title,
            pn.widgets.FloatInput.from_param(
                self.param.iou_threshold,
                width=self.styles.widget_width,
                styles=self.styles.style_text_body1,
                description="Intersection over union threshold to use for NMS deduplication algorithm.",
            ),
            pn.Row(
                pn.widgets.StaticText(
                    value=self.param.class_agnostic.label,
                    styles=self.styles.style_text_body1,
                ),
                pn.widgets.TooltipIcon(
                    value="Whether to consider inference classes in the RealLabel run",
                    styles=self.styles.style_text_body1,
                ),
            ),
            pn.widgets.Switch.from_param(
                self.param.class_agnostic,
                stylesheets=[self.styles.css_switch],
            ),
            pn.Row(
                pn.widgets.StaticText(
                    value=self.param.run_with_ground_truth.label,
                    styles=self.styles.style_text_body1,
                ),
                pn.widgets.TooltipIcon(
                    value="Whether to use ground truth as part of the RealLabel run",
                    styles=self.styles.style_text_body1,
                ),
            ),
            pn.widgets.Switch.from_param(
                self.param.run_with_ground_truth,
                stylesheets=[self.styles.css_switch],
            ),
            pn.widgets.FloatInput.from_param(
                self.param.likely_missed_min_confidence,
                width=self.styles.widget_width,
                styles=self.styles.style_text_body1,
                description="The aggregated confidence value above which a model ensemble ground truth "
                "disagreement will be interpreted as a potentially missing label.",
                step=0.01,
                format="0.00",
            ),
            pn.widgets.FloatInput.from_param(
                self.param.likely_wrong_max_confidence,
                width=self.styles.widget_width,
                styles=self.styles.style_text_body1,
                description="The aggregated confidence value below which a model ensemble ground truth "
                "disagreement will be interpreted as a likely either extraneous or incorrect.",
                step=0.01,
                format="*.00",
            ),
            width=self.styles.app_width,
        )

    def panel(self) -> pn.Column:
        """Return the high-level panel app view.

        Returns
        -------
        pn.Column
            The high-level panel app view.
        """
        return pn.Column(
            self.view_header,
            self.settings_pane,
            self.view_status_bar,
            pn.Spacer(height=24),
            pn.Row(pn.HSpacer(), self.next_button),
            width=self.styles.app_width,
            styles={"background": self.styles.color_main_bg},
        )


if __name__ == "__main__":  # pragma: no cover
    sd: RealLabelApp = RealLabelApp()
    app = sd.panel()
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        os.makedirs("artifacts", exist_ok=True)
        app.save(os.path.join("artifacts", "real_label_app.html"), resources=INLINE)
    elif len(sys.argv) > 1:
        msg = f"Got unexpected flag: {sys.argv[1]}"
        sys.stderr.write(msg)
        sys.exit(1)
    else:
        # special adaption to ensure the poetry blocks execution when the server is running
        server = pn.serve(app, address="localhost", port=5008, show=True, threaded=True)
        server.join()
