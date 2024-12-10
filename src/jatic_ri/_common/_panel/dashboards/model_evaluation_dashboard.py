"""Model Evaluation Object Detection Dashboard"""

from typing import Any

import pandas as pd
import panel as pn

from jatic_ri._common._panel.dashboards.base_dashboard import BaseDashboard


class ModelEvaluationDashboard(BaseDashboard):
    """Model Evaluation Dashboard
    A panel app for running ME from configuration files.
    """

    def __init__(self, **params: dict[str, Any]) -> None:
        super().__init__(**params)
        self.title = "Model Evaluation Dashboard"
        self.tabulator_widths = {
            "Gradient Report": 200,
            "Model(s)": 200,
            "Dataset": 200,
            "Metric": 200,
            "Threshold": 200,
        }
        self.results_df = pd.DataFrame(columns=list(self.tabulator_widths.keys()))

        self.run_analysis_button.on_click(self._run_button_callback)

        # ensure we don't visualize dataset_2
        self.dataset_2_visible = False
        # ensure we don't visualize multiple models (no add model button)
        self.multi_model_visible = False

    def _run_button_callback(self, event) -> None:  # noqa: ANN001, ARG002 # pragma: no cover
        """This function runs when `run_model_button` is clicked
        Call sequence:
        * Call _run_all_tests from the baseclass which:
          * call `run` and `collect_report_consumables` on all stages
          * generate the report
        * create a new row in the visualized table which includes output summary

        """
        # =========================
        # Load model(s)
        success = self.load_models_from_widgets()
        # if there was an issue loading, stop execution but don't error
        if not success:
            return

        model_names = "-".join(list(self.loaded_models.keys())).replace(" ", "_")

        # =========================
        # Load dataset(s)
        success = self.load_datasets_from_widgets()
        # if there was an issue loading, stop execution but don't error
        if not success:
            return

        # =========================
        # Run analysis
        report_link = self._run_all_tests()

        # =========================
        # Update reports table
        # prepare new row of data for the results table
        new_data = {
            "Model(s)": model_names,
            "Dataset": self.dataset_1_selector.value,
            "Metric": self.metric,
            "Threshold": self.threshold,
            "Gradient Report": report_link,
        }

        # add the new run information to the dataframe
        self.results_df = pd.concat([self.results_df, pd.DataFrame(new_data, index=[0])], ignore_index=True)

    def panel(self) -> pn.Column:
        """View of the entire dashboard"""
        return pn.Column(
            self.view_title,
            self.view_config_input,
            pn.Column(
                pn.Row(
                    self.input_model_pane,
                    self.view_dataset_1_selectors,
                ),
                self.view_threshold_metric,
            ),
            pn.Row(
                pn.layout.Spacer(width=10),
                pn.WidgetBox(
                    "### Results",
                    self.view_df_tabulator,
                    pn.layout.Spacer(height=10),  # add a little buffer at the bottom of the results area
                    styles={"background": self.color_light_gray},
                ),
                pn.layout.Spacer(width=10),
                sizing_mode="stretch_width",
            ),
            pn.layout.Spacer(height=50),
            pn.Row(pn.layout.HSpacer(), self.run_analysis_button, sizing_mode="stretch_width"),
            self.view_status_bar,
            styles={"background": self.color_dark_blue},
            sizing_mode="stretch_width",
        )


if __name__ == "__main__":  # pragma: no cover
    """Instantiate a deployable version of the app"""
    app = ModelEvaluationDashboard(
        task="object_detection",
        output_dir=".",
    )

    app.panel().servable()
