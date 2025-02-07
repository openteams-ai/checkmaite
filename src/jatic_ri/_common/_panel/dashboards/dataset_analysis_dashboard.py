"""dataset analysis dashboard"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn

from jatic_ri._common._panel.dashboards.base_dashboard import BaseDashboard

logger = logging.getLogger()


class DatasetAnalysisDashboard(BaseDashboard):
    """Dataset Analysis Dashboard"""

    def __init__(self, **params: dict[str, Any]) -> None:
        super().__init__(**params)
        self.title = "Dataset Analysis Testbed"

        self.run_analysis_button.on_click(self._run_button_callback)
        self.run_analysis_button.disabled = True

        # NOTE: column names defined here must match the `new_data` columns in _run_button_callback
        self.tabulator_widths = {
            "Gradient Report": 50,
            "Dataset": 50,
            "Model(s)": 50,
            "Metric": 50,
            "Threshold": 50,
        }
        self.results_df = pd.DataFrame(columns=list(self.tabulator_widths.keys()))

        # always visualize dataset_2
        self.dataset_2_visible = True

    def _run_button_callback(self, event) -> None:  # noqa: ANN001, ARG002  # pragma: no cover
        """This function runs when `run_dataset_button` is clicked"""
        # =========================
        # Load model(s)
        success = self.load_models_from_widgets()
        # if there was an issue loading, stop execution but don't error
        if not success:
            msg = "Models failed to load from widget info"
            logger.error(msg)
            self.status_text = msg
            return

        model_names = "-".join(list(self.loaded_models.keys())).replace(" ", "_")

        # =========================
        # Load dataset(s)
        success = self.load_datasets_from_widgets()
        # if there was an issue loading, stop execution but don't error
        if not success:
            msg = "Datasets failed to load from widget info"
            logger.error(msg)
            self.status_text = msg
            return

        # =========================
        # Load metric
        success = self.load_metric_from_widget()
        # if there was an issue loading, stop execution but don't error
        if not success:
            msg = "Metric failed to load from widget info"
            logger.error(msg)
            self.status_text = msg
            return

        # =========================
        # Run analysis
        report_link = self._run_all_tests()

        # =========================
        # Update reports table
        # prepare new row of data for the results table
        # NOTE: column names defined here must match the `tabulator_widths` columns in init

        new_data = {
            "Model(s)": model_names,
            "Metric": self.metric_selector.value,
            "Threshold": self.threshold,
            "Gradient Report": report_link,
        }

        new_data["Dataset"] = (
            "-".join(list(self.loaded_datasets.keys())).replace(" ", "_")
            if self.dataset_2_visible
            else self.dataset_1_selector.value.replace(" ", "_")
        )

        # add the new run information to the dataframe
        self.results_df = pd.concat([self.results_df, pd.DataFrame(new_data, index=[0])], ignore_index=True)

    def view_test_subject_row(self) -> pn.Column:
        """View of the subject under test widgets"""

        section_title = "2. Define Dataset under test"
        return pn.Column(
            pn.Spacer(height=20),
            pn.Row(
                # the text on the left:
                pn.Column(
                    pn.pane.Markdown(
                        section_title,
                        styles=self.style_text_h3,
                        stylesheets=[self.css_paragraph],
                    ),
                    pn.Row(
                        pn.Spacer(width=12),  # padding to align this with title text above
                        pn.pane.Markdown(
                            "Select models to analyze.",
                            styles=self.style_text_body2,
                            stylesheets=[self.css_paragraph],
                            width=395,
                        ),
                    ),
                ),
                pn.Spacer(width=124),  # padding between left and right content
                pn.Column(
                    pn.Spacer(height=14),  # padding above white box
                    # the white box on the right:
                    pn.Column(
                        pn.Spacer(height=14),
                        pn.Row(pn.Spacer(width=12), self._view_dataset_1_selectors),
                        pn.Spacer(height=18),
                        styles={
                            "background": self.color_white,
                            "border-color": self.color_blue_300,
                            "border-width": "thin",
                            "border-style": "solid",
                            "border-radius": "5px",
                        },
                        sizing_mode="stretch_width",
                    ),
                    pn.Spacer(height=14),  # padding below white box
                ),
                pn.Spacer(width=23),  # padding on the right edge of the app
            ),
            pn.Spacer(height=20),  # this controls the padding at the bottom of the whole section
        )

    def view_input_artifacts_row(self) -> pn.Column:
        """View of input artifacts row which includes the dataset(s)
        and the evaluation metric
        """
        view_model_1 = pn.Column(
            pn.Row(
                self.model_widgets["Model 1 type"]["model_selector"],
                self.model_widgets["Model 1 type"]["remove_button"],
            ),
            pn.Row(
                pn.Spacer(width=self.width_subwidget_offset),
                self.model_widgets["Model 1 type"]["model_weights_path"],
                # self.model_widgets["Model 1 type"]["tooltip"],
            ),
        )

        model_widget_section = pn.Column(
            view_model_1,
            self.view_model_widget_pairs,
            sizing_mode="stretch_width",
        )

        # only add the add_model option for multi-model cases
        if self.multi_model_visible:
            model_widget_section.append(self.add_model)

        return pn.Column(
            pn.Spacer(height=20),  # top padding for the overall section
            pn.Row(
                # section text on left:
                pn.Column(
                    pn.pane.Markdown(
                        "3. Define test input artifacts",
                        styles=self.style_text_h3,
                        stylesheets=[self.css_paragraph],
                    ),
                    pn.Row(
                        pn.Spacer(width=12),  # padding to align this with title text above
                        pn.pane.Markdown(
                            "Set metric type and comparison dataset and/or model type and location",
                            styles=self.style_text_body2,
                            stylesheets=[self.css_paragraph],
                            width=395,
                        ),
                    ),
                ),
                pn.Spacer(width=124),
                # white box on the right:
                pn.Row(
                    pn.Spacer(width=12),  # padding on left side of white box
                    pn.Column(
                        pn.Spacer(height=18),
                        self._view_dataset_2_selectors,
                        model_widget_section,
                        self.metric_selector,
                        pn.Spacer(height=18),
                    ),
                    styles={
                        "background": self.color_white,
                        "border-color": self.color_blue_300,
                        "border-width": "thin",
                        "border-style": "solid",
                        "border-radius": "5px",
                    },
                    sizing_mode="stretch_width",
                ),
                pn.Spacer(width=23),
            ),
            pn.Spacer(height=20),  # padding at the bottom of the overall section
        )


if __name__ == "__main__":  # pragma: no cover
    """Instantiate a deployable version of the app"""
    app = DatasetAnalysisDashboard(
        task="object_detection",
        output_dir=Path(".").resolve(),
    )
    app.panel().servable()
