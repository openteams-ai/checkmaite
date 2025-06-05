"""Model Evaluation Configuration Panel Application

XAITK, NRTK, and Baseline Evaluation (maite) are included for both
OD and IC tasks.
"""

import argparse
import textwrap
from typing import Any

import panel as pn
import param

from jatic_ri._common._panel.configurations.base_app import DEFAULT_STYLING, AppStyling, BaseApp


class MEConfigurationLandingPage(BaseApp):
    """Initial landing page for the ME configuration app"""

    title = "Model Evaluation Configuration"  # this is updated upon instantiation based on the task
    description = """
    Select analyses to be included in the configuration. \n Any unselected options will not be included.
    """
    task = param.Selector(default="object_detection", objects=["object_detection", "image_classification"])

    # toggles for displaying optional pages
    show_xaitk_config = param.Boolean(default=False)
    show_nrtk_config = param.Boolean(default=False)

    # dictionary for holding all the output configurations
    # this dictionary is passed between all the stages and
    # is used in the final stage to generate the json file
    output_test_stages = param.Dict({})

    # special parameter for dynamically setting the next stage on THIS PAGE
    next_parameter = param.Selector(
        default="ModelEvaluationTestbed",
        objects=[
            "Configure NRTKOD",
            "Configure NRTKIC",
            "Configure XAITKOD",
            "Configure XAITKIC",
            "ModelEvaluationTestbed",
        ],
    )

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, Any]) -> None:
        super().__init__(styles, **params)
        self.title = f"Model evaluation configuration: {self.task.replace('_', ' ').title()}"
        # make sure that the next stage matches the defaults
        self._update_next_parameter()

        # checkbox for baseline evaluation (aka run maite workflow)
        self.baseline_eval = pn.widgets.Checkbox(stylesheets=[self.styles.css_checkbox])

    @param.depends("show_nrtk_config", "show_xaitk_config", watch=True)
    def _update_next_parameter(self) -> None:
        """If show_nrtk_config or show_xaitk_config is touched, this runs to
        set the next stage."""
        # nrtk
        if self.show_nrtk_config:
            self.next_parameter = f"Configure NRTK{self.suffix}"
        # if not nrtk
        else:
            # if xaitk
            if self.show_xaitk_config:
                self.next_parameter = f"Configure XAITK{self.suffix}"
            # not nrtk and not xaitk
            else:
                self.next_parameter = "ModelEvaluationTestbed"

    @param.output(next_parameter=param.Selector, task=param.String, output_test_stages=param.Dict, local=param.Boolean)
    def output(self) -> tuple:
        """Update the output configuration based on the settings in
        the UI and then output to the next stage.

        The output that is in the returned tuple become variables in the next stage defined
        by the variable names and types listed in the decorator.

        If the xaitk toggle is triggered, set the next stage as `xaitk`.
        If not, set the next stage as `ModelEvaluationTestbed`
        """
        # tell the next stage what it's next stage is going to be
        followup_next_parameter = (
            f"Configure XAITK{self.suffix}" if self.show_xaitk_config else "ModelEvaluationTestbed"
        )

        self.output_test_stages = {
            "task": self.task,
        }
        if self.baseline_eval.value:
            self.output_test_stages.update(
                {
                    "baseline_evaluate": {
                        "TYPE": "BaselineEvaluationTestStage",
                        "CONFIG": {},
                    },
                }
            )

        return followup_next_parameter, self.task, self.output_test_stages, self.local

    def _generate_checkbox_subsection(
        self,
        checkbox: pn.widgets.Checkbox,
        label: str,
        description: str,
        section_height: int = 62,
        requires_config: bool = False,
    ) -> pn.Row:
        """
        Construct a viewable object for the checkbox subsections.

        NOTE: Section height default is set at 62 for single line descriptions. This is a
        departure from the figma spec. Resolving this will require additional css work.
        """

        # build label + optional requires config icon
        label_row = pn.Row(
            pn.pane.Markdown(
                label,
                styles=self.styles.style_text_subtitle,
                stylesheets=[self.styles.css_paragraph],
            ),
            pn.Spacer(sizing_mode="stretch_width"),
            (
                pn.widgets.ButtonIcon(icon="settings", size="1.5em", styles={"opacity": "0.5"})
                if requires_config
                else pn.Spacer(width=0)
            ),
        )

        return pn.Row(
            pn.Spacer(width=24),
            pn.Row(
                pn.Column(
                    pn.Spacer(height=5),
                    checkbox,
                ),
                pn.Column(
                    pn.Spacer(height=4),
                    label_row,
                    pn.pane.Markdown(
                        description,
                        styles=self.styles.style_text_body2,
                        stylesheets=[self.styles.css_paragraph],
                    ),
                ),
                width=649,
                height=section_height,  # style guide departure to account for line height
                styles=self.styles.style_border,
            ),
        )

    def view_analysis_tools(self) -> pn.Column:
        """View of the tools row of the app."""
        nrtk_checkbox = pn.widgets.Checkbox.from_param(
            self.param.show_nrtk_config,
            name="",
            stylesheets=[self.styles.css_checkbox],
        )

        xaitk_checkbox = pn.widgets.Checkbox.from_param(
            self.param.show_xaitk_config,
            name="",
            stylesheets=[self.styles.css_checkbox],
        )

        # build list of rows, starting & ending with spacers
        tools_viewable = [
            pn.Spacer(height=24),
            self._generate_checkbox_subsection(
                self.baseline_eval,
                "Baseline Evaluation",
                "Evaluate model performance against a given metric (specified at runtime).",
            ),
            pn.Spacer(height=12),
            self._generate_checkbox_subsection(
                nrtk_checkbox,
                "NRTK",
                "Generate perturbations for evaluating model robustness.",
                requires_config=True,
            ),
            pn.Spacer(height=12),
            self._generate_checkbox_subsection(
                xaitk_checkbox,
                "XAITK",
                "Generate explanations of model predictions.",
                requires_config=True,
            ),
        ]

        # close out with bottom spacer
        tools_viewable.append(pn.Spacer(height=24))

        # wrap everything in the bordered container
        return pn.Column(
            *tools_viewable,
            styles={
                "background-color": self.styles.color_white,
                "border-color": self.styles.color_border,
                "border-width": "thin",
                "border-style": "solid",
                "border-radius": "8px",
            },
            width=697,
        )

    def panel(self) -> pn.Column:
        """Visualize the landing page"""

        title_row = pn.Column(
            pn.pane.Markdown(
                self.task.replace("_", " ").title(),
                styles=self.styles.style_text_body2,
                stylesheets=[self.styles.css_paragraph],
            ),
            pn.pane.Markdown(
                "Model Evaluation Configuration",
                styles=self.styles.style_text_h2,
                stylesheets=[self.styles.css_paragraph],
            ),
            pn.pane.Markdown(
                "Setup your model evaluation configuration to include tools from various JATIC "
                "products. Once complete, you will be able to download the configuration file to "
                "test your models and datasets",
                styles=self.styles.style_text_body2,
                stylesheets=[self.styles.css_paragraph],
            ),
        )

        analysis_row = pn.Row(
            pn.Column(
                pn.pane.Markdown(
                    "Evaluation Tool Selection",
                    styles=self.styles.style_text_h3,
                    stylesheets=[self.styles.css_paragraph],
                ),
                pn.pane.Markdown(
                    textwrap.dedent(
                        """
                        Setup your model evaluation configuration to include tools from
                        various JATIC products. These tools are designed to assess the
                        quality, consistency and structure of your models.

                        Some tools, marked with ⚙️, require additional configuration on
                        the following pages.
                        """
                    ),
                    styles=self.styles.style_text_body2,
                    stylesheets=[self.styles.css_paragraph],
                    width=395,
                ),
            ),
            pn.Spacer(width=124),
            self.view_analysis_tools,
        )

        return pn.Column(
            self.view_header,
            pn.Row(
                pn.Spacer(width=12),
                pn.Column(
                    title_row,
                    pn.Spacer(height=24),
                    self.horizontal_line(),
                    pn.Spacer(height=24),
                    analysis_row,
                    pn.Spacer(height=24),
                    self.horizontal_line(),
                    pn.Spacer(height=24),
                ),
                pn.Spacer(width=24),
            ),
            pn.Spacer(height=24),
            pn.Row(pn.HSpacer(), self.next_button),
            styles={"background": self.styles.color_main_bg},
            width=self.styles.app_width,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, default="object_detection", choices=["object_detection"])
    args = parser.parse_args()
    app = MEConfigurationLandingPage(args.task)
    pn.panel(app.panel()).servable()
