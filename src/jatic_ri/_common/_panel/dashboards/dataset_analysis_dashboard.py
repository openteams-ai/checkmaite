"""dataset analysis dashboard"""

from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn

from jatic_ri._common._panel.dashboards.base_dashboard import BaseDashboard


class DatasetAnalysisDashboard(BaseDashboard):
    """Dataset Analysis Dashboard"""

    def __init__(self, **params: dict[str, Any]) -> None:
        super().__init__(**params)
        self.title = "Dataset Analysis Dashboard"

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
        self.load_models_from_widgets()
        model_names = "-".join(list(self.loaded_models.keys())).replace(" ", "_")

        # =========================
        # Load dataset(s)
        self.load_datasets_from_widgets()

        # =========================
        # Run analysis
        report_link = self._run_all_tests()

        # =========================
        # Update reports table
        # prepare new row of data for the results table
        # NOTE: column names defined here must match the `tabulator_widths` columns in init

        new_data = {
            "Model(s)": model_names,
            "Metric": self.metric,
            "Threshold": self.threshold,
            "Gradient Report": report_link,
        }

        new_data["Dataset"] = (
            "-".join(list(self.loaded_models.keys())).replace(" ", "_")
            if self.dataset_2_visible
            else self.dataset_1_selector.value.replace(" ", "_")
        )

        # add the new run information to the dataframe
        self.results_df = pd.concat([self.results_df, pd.DataFrame(new_data, index=[0])], ignore_index=True)

    def panel(self) -> pn.Column:
        """View of entire dashboard"""
        return pn.Column(
            self.view_title,
            self.view_config_input,
            self.view_dataset_1_selectors,
            self.view_dataset_2_selectors,
            self.input_model_pane,
            self.view_threshold_metric,
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
    app = DatasetAnalysisDashboard(
        task="object_detection",
        output_dir=Path(".").resolve(),
    )
    app.panel().servable()
