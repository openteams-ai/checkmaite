"""Dataset Analysis Configuration for Object Detection and Image Classification Panel Application

Object detection includes Reallabel, Survivor, and the Dataeval tools
Image Classification includes Survivor and the Dataeval tools
"""

import argparse
import importlib
import json
from io import StringIO
from typing import Any

import panel as pn
import param

from jatic_ri._common._panel.configurations.base_app import BaseApp


class ConfigurationLandingPage(BaseApp):
    """Initial landing page for the DA OD configuration app"""

    task = param.Selector(default="object_detection", objects=["object_detection", "image_classification"])
    title = param.String("Dataset Analysis Configuration")  # this is updated upon instantiation based on the task
    description = """
    Select analyses to be included in the configuration. \n Any unselected options will not be included.
    """

    # toggles for displaying optional pages
    show_survivor_config = param.Boolean(default=False)
    show_reallabel_config = param.Boolean(default=False)

    # dataeval configuration
    # bias - parity, coverage, diversity, balance
    bias = param.Boolean(default=False)

    # shift - Out-of-distributation (AE, VAE) and drift (CMV, MMD, KS)
    shift = param.Boolean(default=False)

    # linting - duplicates detection and outlier detection (dimensional stats, pixel stats, visual stats)
    linting = param.Boolean(default=False)

    # feasibility
    feasibility = param.Boolean(default=False)

    # dictionary for holding all the output configurations
    # this dictionary is passed between all the stages and
    # is used in the final stage to generate the json file
    output_test_stages = param.Dict({})

    # special parameter for dynamically setting the next stage on THIS PAGE
    next_parameter = param.Selector(
        default="Configure Reallabel", objects=["Configure Reallabel", "Configure Survivor", "Finalize"]
    )

    def __init__(self, **params: dict[str, Any]) -> None:
        super().__init__(**params)
        self.title = f"Dataset analysis configuration: {self.task.replace('_', ' ').title()}"
        # make sure that the next stage matches the defaults
        self._update_next_parameter()

    @param.depends("show_reallabel_config", "show_survivor_config", watch=True)
    def _update_next_parameter(self) -> None:
        """If show_reallabel_config or show_survivor_config is touched, this runs to
        set the next stage."""
        # reallabel
        if self.show_reallabel_config:
            self.next_parameter = "Configure Reallabel"
        # if not reallabel
        else:
            # if survivor
            if self.show_survivor_config:
                self.next_parameter = "Configure Survivor"
            # not reallabel and not survivor
            else:
                self.next_parameter = "Finalize"

    @param.output(next_parameter=param.Selector, task=param.String, output_test_stages=param.Dict)
    def output(self) -> tuple:
        """Update the output configuration based on the settings in
        the UI and then output to the next stage.

        The output that is in the returned tuple become variables in the next stage defined
        by the variable names and types listed in the decorator.

        If the survivor toggle is triggered, set the next stage as `survivor`.
        If not, set the next stage as `Finalize`
        """
        # tell the next stage what it's next stage is going to be
        followup_next_parameter = "Configure Survivor" if self.show_survivor_config else "Finalize"

        self.output_test_stages = {
            "task": self.task,
        }
        if self.bias:
            self.output_test_stages.update(
                {
                    "bias": {
                        "TYPE": "DatasetBiasTestStage",
                        "CONFIG": {},
                    },
                }
            )
        if self.feasibility:
            self.output_test_stages.update(
                {
                    "feasibility": {
                        "TYPE": "DatasetFeasibilityTestStage",
                        "CONFIG": {},
                    },
                }
            )
        if self.shift:
            self.output_test_stages.update(
                {
                    "shift": {
                        "TYPE": "DatasetShiftTestStage",
                        "CONFIG": {},
                    },
                }
            )
        if self.linting:
            self.output_test_stages.update(
                {
                    "linting": {
                        "TYPE": "DatasetLintingTestStage",
                        "CONFIG": {},
                    },
                }
            )

        return followup_next_parameter, self.task, self.output_test_stages

    def view_optional_configs(self) -> pn.Row:
        """View the toggles configs"""
        # widget box with config toggles for survivor and maybe reallabel
        optional_configs_widgetbox = pn.WidgetBox(
            "### Add advanced configuration \n (will be configured on subsequent pages)",
            pn.widgets.Toggle.from_param(
                self.param.show_survivor_config,
                name="Configure Survivor",
                button_type="primary",
                button_style="outline",
            ),
            styles={"background": self.color_light_gray},
        )
        # add reallabel toggle only for OD
        if self.task == "object_detection":
            optional_configs_widgetbox.append(
                pn.widgets.Toggle.from_param(
                    self.param.show_reallabel_config,
                    name="Configure Reallabel",
                    button_type="primary",
                    button_style="outline",
                )
            )
        return pn.Row(
            pn.Spacer(width=20),
            pn.WidgetBox(
                "### Select Dataset Evaluation configurations",
                pn.widgets.Toggle.from_param(
                    self.param.linting,
                    name="Include Linting",
                    button_type="primary",
                    button_style="outline",
                ),
                pn.widgets.Toggle.from_param(
                    self.param.shift,
                    name="Include Shift",
                    button_type="primary",
                    button_style="outline",
                ),
                pn.widgets.Toggle.from_param(
                    self.param.bias,
                    name="Include Bias",
                    button_type="primary",
                    button_style="outline",
                ),
                pn.widgets.Toggle.from_param(
                    self.param.feasibility,
                    name="Include Feasibility",
                    button_type="primary",
                    button_style="outline",
                ),
                styles={"background": self.color_light_gray},
            ),
            optional_configs_widgetbox,
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
    """Finalization page for DA OD configuration app. Contains the
    button to download results
    """

    task = param.Selector(default="object_detection", objects=["object_detection", "image_classification"])
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
        """Helper to get configuration as a stringio object for downloading"""
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


class DatasetAnalysisConfigApp(BaseApp):
    """High level constructor for the Dataset Analysis Configuration App

    Creates a Panel Pipeline object from the individual panel apps and
    connects them together for sequential viewing.

    There are two task modes for this app - 'object_detection' and 'image_classification'.

    To view the app (via notebook):

    >>> from jatic_ri._common._panel.configurations.dataset_analysis_configuration import (
    ...     DatasetAnalysisConfigApp,
    ... )
    >>> import panel as pn
    >>> pn.extension()

    >>> app = DatasetAnalysisConfigApp(task="object_detection")
    >>> app.panel()
    """

    def __init__(self, **params: dict[str, object]) -> None:
        super().__init__(**params)

        # setup panel pipeline by adding individual apps and connecting them together
        self.pipeline = pn.pipeline.Pipeline(inherit_params=False, debug=True)

        self.pipeline.add_stage(
            "Introduction",
            ConfigurationLandingPage(task=self.task),
            next_parameter="next_parameter",
        )

        survivor_app_module = importlib.import_module(f"jatic_ri.{self.task}._panel.configurations.survivor_app")
        SurvivorApp = survivor_app_module.SurvivorApp  # noqa: N806 (this really is a class, not a variable)

        if self.task == "object_detection":
            # add reallabel test stage only for object_detection
            reallabel_app_module = importlib.import_module(f"jatic_ri.{self.task}._panel.configurations.reallabel_app")
            RealLabelApp = reallabel_app_module.RealLabelApp  # noqa: N806 (this really is a class, not a variable)
            self.pipeline.add_stage("Configure Reallabel", RealLabelApp, next_parameter="next_parameter")

            # add remaining stages
            self.pipeline.add_stage("Configure Survivor", SurvivorApp)
            self.pipeline.add_stage("Finalize", FinalPage)
            # setup nonlinear dag, actual path is dynamic based on choices made on the intro page
            self.pipeline.define_graph(
                {
                    "Introduction": ("Configure Reallabel", "Configure Survivor", "Finalize"),
                    "Configure Reallabel": ("Configure Survivor", "Finalize"),
                    "Configure Survivor": "Finalize",
                }
            )
        else:
            # add remaining stages
            self.pipeline.add_stage("Configure Survivor", SurvivorApp)
            self.pipeline.add_stage("Finalize", FinalPage)
            # NOTE: define_graph is unneccesasry here since the dag is linear and
            # stages will be added in the order of `.add_stage` above.

    def panel(self) -> pn.Column:
        """Visualize the DA OD configuration app"""
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
    parser.add_argument(
        "--task", "-t", type=str, default="object_detection", choices=["object_detection", "image_classification"]
    )
    args = parser.parse_args()
    app = DatasetAnalysisConfigApp(args.task)
    pn.panel(app.panel()).servable()
