"""heart_app

Run with `--ci` flag to save the app as html instead of serving it.
"""

# Python generic imports
from __future__ import annotations

import os
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

# Panel app imports
import panel as pn
import param
from bokeh.resources import INLINE

# local imports
from jatic_ri._common._panel.configurations.base_app import BaseApp

mpl.use("agg")
pn.extension("tabulator")
pn.extension("jsoneditor")

DOCUMENTATION_LINK = "https://heart-library.readthedocs.io/en/latest/"


class HeartApp(BaseApp):
    """
    This class is the main class for the HeartApp configuration page. It inherits from the BaseApp class and
    provides the necessary methods and visualizations for the HeartApp configuration page.
    """

    title: str = param.String(default="HEART: Object Detection")
    title_font_size: int = param.Integer(default=24)
    status_text: str = param.String("Waiting for input...")
    add_button: pn.widgets.Button
    clear_button: pn.widgets.Button
    config_upload: pn.widgets.FileInput
    attack_select: pn.widgets.Select
    parameter_select: pn.widgets.Select
    defense_pgd_select: pn.widgets.Select
    defense_patch_select: pn.widgets.Select
    test_attack_button: pn.widgets.Button
    original_plot: pn.pane.Matplotlib
    augmented_plot: pn.pane.Matplotlib
    test_stages: list[dict[str, Any]]
    all_widgets: list[pn.Card]
    widget_values: list[dict[str, Any]]
    finished_factory_display: list[pn.pane.Markdown]

    def __init__(self, **params: dict[str, object]) -> None:
        self.add_button = pn.widgets.Button(
            name="Add Test Stage",
            button_type="primary",
        )  # declare here since its used in pn.depends
        self.clear_button = pn.widgets.Button(
            name="Clear Test Stages",
            button_type="primary",
        )  # declare here since its used in pn.depends
        self.config_upload = pn.widgets.FileInput(accept=".json")  # declare here since its used in pn.depends
        self.attack_select = pn.widgets.Select(
            name="Attack Type",
            options=["Patch Attack", "Projected Gradient Descent (PGD)"],
            value="Patch Attack",
        )
        self.parameter_select = pn.widgets.Select(
            name="Parameters",
            options=["Weaker Attack", "Stronger Attack"],
            value="Weaker Attack",
        )
        self.defense_pgd_select = pn.widgets.Select(
            name="Defenses",
            options=["Spatial Smoothing", "None"],
            value="None",
        )
        self.defense_patch_select = pn.widgets.Select(name="Defenses", options=["None"], value="None")

        super().__init__(**params)
        self.add_button.stylesheets = [self.button_stylesheet]
        self.clear_button.stylesheets = [self.button_stylesheet]
        self.config_upload.stylesheets = [self.widget_stylesheet]
        self.attack_select.stylesheets = [self.widget_stylesheet]
        self.parameter_select.stylesheets = [self.widget_stylesheet]
        self.defense_pgd_select.stylesheets = [self.widget_stylesheet]
        self.defense_patch_select.stylesheets = [self.widget_stylesheet]

        self.add_button.on_click(self.add_test_stage_callback)
        self.clear_button.on_click(self.clear_test_stage_callback)

        self.original_plot = self.create_original_plot()
        self.augmented_plot = pn.pane.Matplotlib(plt.figure(), tight=True, visible=False)

        self.test_stages = []

        self.left_column_width = 410

        def single_attack_callback(_target: object, _event: object) -> None:  # pragma: no cover
            self.all_widgets = [
                pn.Card(
                    self.add_attack_config_widget(),
                    title=f"{self.attack_select.value} ",
                    header_color=self.color_gray_400,
                ),
            ]

        self.all_widgets = [
            pn.Card(
                self.add_attack_config_widget(),
                title=f"{self.attack_select.value} ",
                header_color=self.color_gray_400,
                width=self.left_column_width,
            ),
        ]

        self.widget_values = []
        self.finished_factory_display = []

        self.attack_select.link(self.attack_select, callbacks={"value": single_attack_callback})

    def _run_export(self) -> None:
        """This function runs when `export_button` is clicked"""
        for index, stage in enumerate(self.test_stages):
            self.output_test_stages[f"heart-{index}"] = stage

    def add_test_stage_to_json(self, _event: object = None) -> None:
        """Adds the test stage to the JSON file"""
        test_stage = {
            "TYPE": "HeartTestStage",
            "CONFIG": {
                "attack_type": self.attack_select.value,
                "parameters": self.parameter_select.value,
                "defenses": self.get_defense_string(),
            },
        }
        self.test_stages.append(test_stage)

    def add_attack_config_widget(self) -> pn.Column | None:  # pragma: no cover
        """Returns the widget for the attack configuration"""
        if self.attack_select.value == "Projected Gradient Descent (PGD)":
            return pn.Column(
                self.parameter_select,
                self.defense_pgd_select,
            )
        if self.attack_select.value == "Patch Attack":
            return pn.Column(
                self.parameter_select,
                self.defense_patch_select,
            )
        return None

    def get_defense_string(self) -> str | None:  # pragma: no cover
        """Returns the defense string"""
        if self.attack_select.value == "Projected Gradient Descent (PGD)":
            return self.defense_pgd_select.value
        if self.attack_select.value == "Patch Attack":
            return self.defense_patch_select.value
        return None

    def add_test_stage_callback(self, _event: object) -> None:
        """Adds the test stage to the list of test stages"""
        self.add_test_stage_to_json()

        self.finished_factory_display.append(
            pn.pane.Markdown(
                f"""
            <style>
            * {{
              color: {self.color_gray_900};
            }}
            </style>
            * Attack Type: {self.attack_select.value}
            * Parameters: {self.parameter_select.value}
            * Defenses: {self.get_defense_string()}
            """,
            ),
        )

    def clear_test_stage_callback(self, _event: object) -> None:
        """Clears all the stored widgets when the clear_button is clicked"""
        self.finished_factory_display = []
        self.test_stages = []

    def create_original_plot(self, _target: object = None, _event: object = None) -> pn.pane.Markdown:
        """Create a pane with a documentation link"""
        # Create a Markdown pane to display the link with inline HTML styling
        return pn.pane.Markdown(
            f'<a href="{DOCUMENTATION_LINK}" '
            f'style="color: white; font-size: 22px;">'
            "HEART-library documentation</a>",
        )

    @pn.depends("add_button.clicks", "clear_button.clicks", "attack_select.value")
    def view_sweep_params(self, _events: object = None) -> pn.Row:
        """When the add test stage button is clicked or the "clear test stages"
        button is clicked, this will trigger and update the view of the widgets"""
        # using pn.Card here to match the look of the collapsible config section
        return pn.Row(
            pn.Spacer(width=5),  # added spacer for visual separation
            pn.Column(
                pn.Column(*self.all_widgets),
                pn.layout.Divider(),
                pn.Row(
                    self.add_button,
                    self.clear_button,
                ),
                pn.layout.Divider(),
                pn.Column(*self.finished_factory_display),
            ),
        )

    @pn.depends("attack_select.value")
    def view_sweep_param(self, _events: object = None) -> pn.Row:  # pragma: no cover
        """When the add test stage button is clicked or the 'Clear Test Stages'
        button is clicked, this will trigger and update the view of the widgets"""
        # using pn.Card here to match the look of the collapsible config section
        return pn.Row(
            pn.Spacer(width=5),  # added spacer for visual separation
            pn.Column(*self.all_widgets),
        )

    def view_plots(self) -> pn.Column:
        """View of the plots"""
        return pn.Column(
            self.original_plot,
            self.augmented_plot,
        )

    def panel(self) -> pn.Column:
        """High level view of the full app"""
        left_column = pn.Column(
            self.attack_select,
            pn.Spacer(height=10),  # added spacer for visual separation
            self.view_sweep_params,
            width=self.left_column_width,
        )
        right_column = pn.Column(self.view_plots)
        return pn.Column(
            pn.Row(self.view_title, pn.layout.HSpacer()),
            pn.Row(
                left_column,
                right_column,
            ),
            pn.Row(pn.layout.HSpacer(), self.export_button),
            self.view_status_bar,
            width=self.page_width,
            styles={"background": self.color_main_bg},
        )


if __name__ == "__main__":  # pragma: no cover
    sd: HeartApp = HeartApp()
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        os.makedirs("artifacts", exist_ok=True)
        sd.panel().save(os.path.join("artifacts", "heart_app.html"), resources=INLINE)
    elif len(sys.argv) > 1:
        msg = f"Got unexpected flag: {sys.argv[1]}"
        sys.stderr(msg)
        sys.exit(1)
    else:
        pn.serve(sd.panel(), host="127.0.0.1", port=5008)
