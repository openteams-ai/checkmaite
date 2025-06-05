"""Model Evaluation Configuration for Object Detection Panel Application

XAITK, NRTK, and Baseline Evaluation (maite) are included for both
OD and IC tasks.
"""

import argparse
import importlib
import json
import textwrap
from io import StringIO
from pathlib import Path
from typing import Any

import panel as pn
import param

from jatic_ri._common._panel.configurations.base_app import DEFAULT_STYLING, AppStyling, BaseApp
from jatic_ri.util.dashboard_utils import _center_horizontally, _center_vertically


class ConfigurationLandingPage(BaseApp):
    """Initial landing page for the ME configuration app"""

    task = param.Selector(default="object_detection", objects=["object_detection", "image_classification"])
    title = param.String("Model Evaluation Configuration")  # this is updated upon instantiation based on the task
    description = """
    Select analyses to be included in the configuration. \n Any unselected options will not be included.
    """

    # toggles for displaying optional pages
    show_xaitk_config = param.Boolean(default=False)
    show_nrtk_config = param.Boolean(default=False)

    # dictionary for holding all the output configurations
    # this dictionary is passed between all the stages and
    # is used in the final stage to generate the json file
    output_test_stages = param.Dict({})

    # special parameter for dynamically setting the next stage on THIS PAGE
    next_parameter = param.Selector(default="Configure NRTK", objects=["Configure NRTK", "Configure XAITK", "Finalize"])

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
            self.next_parameter = "Configure NRTK"
        # if not nrtk
        else:
            # if xaitk
            if self.show_xaitk_config:
                self.next_parameter = "Configure XAITK"
            # not nrtk and not xaitk
            else:
                self.next_parameter = "Finalize"

    @param.output(next_parameter=param.Selector, task=param.String, output_test_stages=param.Dict, local=param.Boolean)
    def output(self) -> tuple:
        """Update the output configuration based on the settings in
        the UI and then output to the next stage.

        The output that is in the returned tuple become variables in the next stage defined
        by the variable names and types listed in the decorator.

        If the xaitk toggle is triggered, set the next stage as `xaitk`.
        If not, set the next stage as `Finalize`
        """
        # tell the next stage what it's next stage is going to be
        followup_next_parameter = "Configure XAITK" if self.show_xaitk_config else "Finalize"

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
            styles={"background": self.styles.color_main_bg},
            width=self.styles.app_width,
        )


class FinalPage(BaseApp):
    """Finalization page for ME OD configuration app. Contains the
    button to download results
    """

    task = param.Selector(default="object_detection", objects=["object_detection", "image_classification"])
    title = param.String("Dashboard Config Finalization")

    # dictionary for holding all the output configurations
    # this dictionary is passed between all the stages and
    # is used in the final stage to generate the json file
    output_test_stages = param.Dict({})

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]) -> None:
        super().__init__(styles, **params)
        self.filename = Path("config.json")
        self.writeout_button = pn.widgets.FileDownload(
            filename=str(self.filename),
            callback=self._get_filestream,
            stylesheets=[self.styles.css_button],
        )

    def _get_filestream(self) -> StringIO:
        config = {"task": self.task}
        config.update(self.output_test_stages)
        sio = StringIO()
        json.dump(config, sio, indent=4)
        sio.seek(0)
        return sio

    def panel(self) -> pn.Column:
        """Visualize the Final page"""

        if self.local:
            config = {"task": self.task}
            config.update(self.output_test_stages)
            # Write JSON to a file
            with open(self.filename, "w") as file:
                json.dump(config, file, indent=4)

            summary_view = pn.Column(
                _center_horizontally(
                    pn.widgets.ButtonIcon(
                        icon="checkbox", size="4em", description="favorite", styles={"color": "blue"}
                    ),
                ),
                _center_horizontally(
                    pn.pane.Markdown(
                        "You're all set! Your .json configuration file is located at",
                        stylesheets=[self.styles.css_paragraph],
                    )
                ),
                _center_horizontally(
                    pn.pane.Markdown(f"{self.filename.resolve()}", stylesheets=[self.styles.css_paragraph])
                ),
                styles={**self.styles.style_border, "padding": "15px"},
            )
        else:
            summary_view = pn.Column(
                _center_horizontally(
                    pn.widgets.ButtonIcon(icon="checkbox", size="4em", description="favorite", styles={"color": "blue"})
                ),
                _center_horizontally(
                    pn.pane.Markdown(
                        "You're all set! Download your .json file below to",
                        stylesheets=[self.styles.css_paragraph],
                    )
                ),
                _center_horizontally(
                    pn.pane.Markdown("continue your evaluation pipeline.", stylesheets=[self.styles.css_paragraph])
                ),
                _center_horizontally(self.writeout_button),
                styles={**self.styles.style_border, "padding": "15px"},
            )

        return pn.Column(
            self.view_header,
            _center_vertically(
                pn.Row(
                    _center_horizontally(
                        pn.Column(
                            _center_horizontally(
                                pn.pane.Markdown(
                                    "Success!",
                                    styles=self.styles.style_text_h1,
                                )
                            ),
                            summary_view,
                        ),
                    ),
                ),
            ),
            styles={"background-color": self.styles.color_main_bg},
            width=self.styles.app_width,
            height=400,
        )


class ModelEvaluationConfigApp(BaseApp):
    """High level constructor for the Model Evaluation Configuration App

    Creates a Panel Pipeline object from the individual panel apps and
    connects them together for sequential viewing.

    There are two task modes for this app - 'object_detection' and 'image_classification'.

    To view the app (via notebook):

    >>> from jatic_ri._common._panel.configurations.model_evaluation_configuration import (
    ...     ModelEvaluationConfigApp,
    ... )
    >>> import panel as pn
    >>> pn.extension()

    >>> app = ModelEvaluationConfigApp(task="object_detection")
    >>> app.panel()
    """

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]) -> None:
        super().__init__(styles, **params)

        # setup panel pipeline by adding individual apps and connecting them together
        self.pipeline = pn.pipeline.Pipeline(inherit_params=False, debug=True)

        nrtk_app_module = importlib.import_module(f"jatic_ri.{self.task}._panel.configurations.nrtk_app")
        NRTKApp = nrtk_app_module.NRTKApp  # noqa: N806 (this really is a class, not a variable)

        xaitk_app_module = importlib.import_module(f"jatic_ri.{self.task}._panel.configurations.xaitk_app")
        XAITKApp = xaitk_app_module.XAITKApp  # noqa: N806 (this really is a class, not a variable)

        self.pipeline.add_stage(
            "Introduction",
            ConfigurationLandingPage(styles, **params),
            next_parameter="next_parameter",
        )
        self.pipeline.add_stage("Configure NRTK", NRTKApp, next_parameter="next_parameter")
        self.pipeline.add_stage("Configure XAITK", XAITKApp)
        self.pipeline.add_stage("Finalize", FinalPage)

        # setup nonlinear dag, actual path is dynamic based on choices made on the intro page
        self.pipeline.define_graph(
            {
                "Introduction": ("Configure NRTK", "Configure XAITK", "Finalize"),
                "Configure NRTK": ("Configure XAITK", "Finalize"),
                "Configure XAITK": "Finalize",
            }
        )

    def panel(self) -> pn.Column:
        """Visualize the ME configuration app"""
        return pn.Column(
            self.pipeline.stage,
            pn.Row(
                pn.HSpacer(),
                pn.Row(
                    self.pipeline.prev_button,
                    self.pipeline.next_button,
                ),
            ),
            styles={"background": self.styles.color_main_bg},
            width=self.styles.app_width,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, default="object_detection", choices=["object_detection"])
    args = parser.parse_args()
    app = ModelEvaluationConfigApp(args.task)
    pn.panel(app.panel()).servable()
