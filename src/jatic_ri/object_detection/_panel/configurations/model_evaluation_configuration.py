"""Model Evaluation Configuration for Object Detection Panel Application"""

import argparse
import json
from io import StringIO
from typing import Any

import panel as pn
import param

from jatic_ri.object_detection._panel.configurations.base_app import BaseApp
from jatic_ri.object_detection._panel.configurations.nrtk_app import NRTKApp
from jatic_ri.object_detection._panel.configurations.xaitk_app import XAITKApp


class ConfigurationLandingPage(BaseApp):
    """Initial landing page for the ME OD configuration app"""

    task = param.Selector(default="object_detection", objects=["object_detection"])
    title = param.String("Model Evaluation Configuration")  # this is updated upon instantiation based on the task
    description = """
    Select analyses to be included in the configuration. \n Any unselected options will not be included.
    """

    # toggles for displaying optional pages
    show_xaitk_config = param.Boolean(default=False)
    show_nrtk_config = param.Boolean(default=False)

    # run MAITE evaluation workflow
    baseline_evaluate = param.Boolean(default=False)

    # dictionary for holding all the output configurations
    # this dictionary is passed between all the stages and
    # is used in the final stage to generate the json file
    output_test_stages = param.Dict({})

    # special parameter for dynamically setting the next stage on THIS PAGE
    next_parameter = param.Selector(default="Configure NRTK", objects=["Configure NRTK", "Configure XAITK", "Finalize"])

    def __init__(self, **params: dict[str, Any]) -> None:
        super().__init__(**params)
        self.title = f"Model evaluation configuration: {self.task.replace('_', ' ').title()}"
        # make sure that the next stage matches the defaults
        self._update_next_parameter()

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

    @param.output(next_parameter=param.Selector, task=param.String, output_test_stages=param.Dict)
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
        if self.baseline_evaluate:
            self.output_test_stages.update(
                {
                    "baseline_evaluate": {
                        "TYPE": "BaselineEvaluation",
                        "CONFIG": {},
                    },
                }
            )

        return followup_next_parameter, self.task, self.output_test_stages

    def view_optional_configs(self) -> pn.Row:
        """View the toggles configs"""
        return pn.Row(
            pn.Spacer(width=20),
            pn.WidgetBox(
                "### Select Model Evaluation configurations",
                pn.widgets.Toggle.from_param(
                    self.param.baseline_evaluate,
                    name="Include Baseline Evaluation",
                    button_type="primary",
                    button_style="outline",
                ),
                styles={"background": self.color_light_gray},
            ),
            pn.WidgetBox(
                "### Add advanced configuration \n (will be configured on subsequent pages)",
                pn.widgets.Toggle.from_param(
                    self.param.show_xaitk_config,
                    name="Configure XAITK",
                    button_type="primary",
                    button_style="outline",
                ),
                pn.widgets.Toggle.from_param(
                    self.param.show_nrtk_config,
                    name="Configure NRTK",
                    button_type="primary",
                    button_style="outline",
                ),
                styles={"background": self.color_light_gray},
            ),
        )

    def panel(self) -> pn.Column:
        """Visualize the landing page"""
        return pn.Column(
            self.view_title(),
            pn.pane.Markdown(
                self.description,
                styles={"font-size": f"{self.summary_text_size}px", "color": f"{self.color_light_gray}"},
                stylesheets=[self.widget_stylesheet],
            ),
            self.view_optional_configs,
            pn.Spacer(height=20),
            width=self.page_width,
            height=self.page_height,
            styles={"background": self.color_dark_blue},
        )


class FinalPage(BaseApp):
    """Finalization page for ME OD configuration app. Contains the
    button to download results
    """

    task = param.Selector(default="object_detection", objects=["object_detection"])
    title = param.String("Dashboard Config Finalization")

    # dictionary for holding all the output configurations
    # this dictionary is passed between all the stages and
    # is used in the final stage to generate the json file
    output_test_stages = param.Dict({})

    def __init__(self, **params: dict[str, object]) -> None:
        super().__init__(**params)
        self.writeout_button = pn.widgets.FileDownload(
            filename="config.json",
            callback=self._get_filestream,
            name="Download the dashboard config",
            styles={"font-size": f"{self.summary_text_size}px", "color": f"{self.color_light_gray}"},
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
        return pn.Column(
            self.view_title(),
            pn.Row(self.writeout_button, width=self.page_width, stylesheets=[self.widget_stylesheet]),
            width=self.page_width,
            height=self.page_height,
            styles={"background": self.color_dark_blue},
        )


class ModelEvaluationConfigApp(BaseApp):
    """High level constructor for the Model Evaluation Configuration App

    Creates a Panel Pipeline object from the individual panel apps and
    connects them together for sequential viewing.

    There are two task modes for this app - 'object_detection' and 'image_classification'.
    The 'image_classification' mode is TBD.

    To view the app (via notebook):

    >>> from jatic_ri.object_detection._panel.configurations.model_evaluation_configuration import (
    ...     ModelEvaluationConfigApp,
    ... )
    >>> import panel as pn
    >>> pn.extension()

    >>> app = ModelEvaluationConfigApp(task="object_detection")
    >>> app.panel()
    """

    def __init__(self, task: str = "object_detection", **params: dict[str, object]) -> None:
        super().__init__(**params)

        # setup panel pipeline by adding individual apps and connecting them together
        self.pipeline = pn.pipeline.Pipeline(inherit_params=False, debug=True)

        if task == "object_detection":
            self.pipeline.add_stage(
                "Introduction",
                ConfigurationLandingPage(task=task),
                next_parameter="next_parameter",
            )
            self.pipeline.add_stage("Configure NRTK", NRTKApp, next_parameter="next_parameter")
            self.pipeline.add_stage("Configure XAITK", XAITKApp)
            self.pipeline.add_stage("Finalize", FinalPage)
        else:
            raise RuntimeError(f"Sorry the task type, {task}, has not been coded yet. WOMP WOMP.")

        # setup nonlinear dag, actual path is dynamic based on choices made on the intro page
        self.pipeline.define_graph(
            {
                "Introduction": ("Configure NRTK", "Configure XAITK", "Finalize"),
                "Configure NRTK": ("Configure XAITK", "Finalize"),
                "Configure XAITK": "Finalize",
            }
        )

    def panel(self) -> pn.Column:
        """Visualize the ME OD configuration app"""
        return pn.Column(
            self.pipeline.stage,
            pn.Row(
                pn.HSpacer(),
                pn.Row(
                    self.pipeline.prev_button,
                    self.pipeline.next_button,
                ),
            ),
            styles={"background": self.color_dark_blue},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, default="object_detection", choices=["object_detection"])
    args = parser.parse_args()
    app = ModelEvaluationConfigApp(args.task)
    pn.panel(app.panel()).servable()
