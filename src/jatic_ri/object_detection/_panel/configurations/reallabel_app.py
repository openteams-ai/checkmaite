"""Module for Object Detection RealLabel panel app.

Run with `--ci` flag to save the app as html instead of serving it.
"""

import os
import sys
from typing import Any

import panel as pn
import param
from bokeh.resources import INLINE

from jatic_ri.object_detection._panel.configurations.base_app import BaseApp

pn.extension()


class RealLabelApp(BaseApp):
    """RealLabel panel App. Creates a GUI interface that allows the user to enter and export RealLabelConfig values.

    Attributes:
        title (param.String): settings pane title.
        iou_threshold (param.Number): threshold for IoU calculation. Defaults to 0.5.
        likely_missed_min_confidence (param.Number): The aggregated confidence value above which a model ensemble
            ground truth disagreement will be interpreted as a potentially missing label.
            Defaults to 0.5.
        likely_wrong_max_confidence (param.Number): The aggregated confidence value below which a model ensemble
            ground truth disagreement will be interpreted as a likely either extraneous or incorrect.
            Defaults to 0.5.
        class_agnostic (param.Boolean): Set to True for when you only care about the LOCATIONS of
            bounding boxes (and not their labels). Defaults to True.
        run_with_ground_truth (param.Boolean): Set to True when you want to run RealLabel with ground truth data.
            Defaults to False.
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
    run_with_ground_truth: param.Boolean = param.Boolean(default=False, label="Run with Ground Truth")

    def __init__(self, **params: dict[str, Any]) -> None:
        super().__init__(**params)

    def _run_export(self) -> None:
        """Exports a dictionary representation of the RealLabelConfig entered by the user.

        The output is exported to the `self.output_tests_stages` dictionary under the "reallabel_test_stage" key.
        """

        # Just giving the option to call any RealLabel output
        additional_outputs = [
            "results",
            "verbose_df",
            "classification_disagreements_df",
            "sequence_priority_score_df",
            "wanrs_df",
            "sequence_priority_score_balanced_df",
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
            },
        }

    def settings_pane(self) -> pn.Column:
        """View of settings"""
        return pn.Column(
            self.view_title,
            pn.widgets.FloatInput.from_param(
                self.param.iou_threshold,
                width=self.widget_width,
                stylesheets=[self.widget_stylesheet, self.info_button_style],
                description="Intersection over union threshold to use for NMS deduplication algorithm.",
            ),
            pn.Row(
                pn.widgets.StaticText(
                    value=self.param.class_agnostic.label,
                    stylesheets=[self.text_color_styling],
                ),
                pn.widgets.TooltipIcon(
                    value="Whether to consider inference classes in the RealLabel run",
                    stylesheets=[self.text_color_styling, self.info_button_style],
                ),
            ),
            pn.widgets.Switch.from_param(self.param.class_agnostic),
            pn.Row(
                pn.widgets.StaticText(
                    value=self.param.run_with_ground_truth.label,
                    stylesheets=[self.text_color_styling],
                ),
                pn.widgets.TooltipIcon(
                    value="Whether to use ground truth as part of the RealLabel run",
                    stylesheets=[self.text_color_styling, self.info_button_style],
                ),
            ),
            pn.widgets.Switch.from_param(self.param.run_with_ground_truth),
            pn.widgets.FloatInput.from_param(
                self.param.likely_missed_min_confidence,
                width=self.widget_width,
                stylesheets=[self.widget_stylesheet, self.info_button_style],
                description="The aggregated confidence value above which a model ensemble ground truth "
                "disagreement will be interpreted as a potentially missing label.",
                step=0.01,
                format="0.00",
            ),
            pn.widgets.FloatInput.from_param(
                self.param.likely_wrong_max_confidence,
                width=self.widget_width,
                stylesheets=[self.widget_stylesheet, self.info_button_style],
                description="The aggregated confidence value below which a model ensemble ground truth "
                "disagreement will be interpreted as a likely either extraneous or incorrect.",
                step=0.01,
                format="*.00",
            ),
            width=self.page_width,
        )

    def panel(self) -> pn.Column:
        """High level panel app"""
        return pn.Column(
            self.settings_pane,
            pn.Row(pn.layout.HSpacer(), self.export_button),
            self.view_status_bar,
            width=self.page_width,
            styles={"background": self.color_dark_blue},
        )


if __name__ == "__main__":  # pragma: no cover
    sd: RealLabelApp = RealLabelApp()
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        os.makedirs("artifacts", exist_ok=True)
        sd.panel().save(os.path.join("artifacts", "real_label_app.html"), resources=INLINE)
    elif len(sys.argv) > 1:
        msg = f"Got unexpected flag: {sys.argv[1]}"
        sys.stderr(msg)
        sys.exit(1)
    else:
        pn.serve(sd.panel(), host="127.0.0.1", port=5008)
