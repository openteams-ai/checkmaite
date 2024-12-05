"""Dataset Analysis Configuration for Object Detection Panel Application"""

import argparse
import json
from io import StringIO
from typing import Any

import panel as pn
import param

from jatic_ri.object_detection._panel.configurations.reallabel_app import RealLabelApp
from jatic_ri.object_detection._panel.configurations.survivor_app import SurvivorApp


class StyleTemplate(param.Parameterized):
    """Styling class to be inherited by all configuration apps.

    This class holds common UI elements and styling parameters, allowing
    the individual apps to be more streamlined and organize all styling
    aspects into a single location.
    """

    page_width = param.Integer(800)
    title_font_size = param.Integer(default=24)
    blurb_font_size = param.Integer(default=18)
    title = param.String("")

    def __init__(self, **params: dict[str, object]) -> None:
        super().__init__(**params)
        self.color_light_gray = "#fafefe"  # style guide
        self.color_dark_blue = "#1d385b"  # style guide
        # self.color_medium_blue = '#7AB8EF'  # variable currently unused
        self.color_light_blue = "#abd1f5"  # style guide
        self.color_black = "#000000"

        # this adjusts the margins on the title
        self.title_stylesheet = """
                :host {
                  --line-height: 10px;
                }

                p {
                  padding: 0px;
                  margin: 10px;
                }
                """
        self.widget_width = (
            140  # the widget construct is overriding this via css so we'll specify on the widget object instead
        )
        self.widget_height = "20px"
        self.widget_stylesheet = f"""
                :host {{
                  color: {self.color_light_gray}; /* label text color */
                }}

                select:not([multiple]).bk-input, select:not([size]).bk-input {{
                  height: {self.widget_height}; /* dropdown widget height */
                  color: {self.color_black} /* text color on value of Dropdown widgets */
                }}

                .bk-input {{
                  height: {self.widget_height};  /* FloatInput widget height */
                  color: {self.color_black} /* text color on value of FloatInput widgets */
                }}
                """
        self.button_bgcolor = self.color_light_blue
        self.button_textcolor = self.color_black
        self.button_stylesheet = f"""
                :host(.solid) .bk-btn.bk-btn-primary {{
                  background-color: {self.button_bgcolor}
                }}

                .bk-btn-primary {{
                  color: {self.button_textcolor}
                }}
                """

    def view_title(self) -> pn.pane.Markdown:
        """View of the app title
        DO NOT OVERWRITE THIS METHOD
        """
        return pn.pane.Markdown(
            self.title,
            styles={"font-size": f"{self.title_font_size}px", "color": f"{self.color_light_gray}"},
            stylesheets=[self.title_stylesheet],
        )


class ConfigurationLandingPage(StyleTemplate):
    """Initial landing page for the DA OD configuration app"""

    task = param.Selector(default="object_detection", objects=["object_detection", "image_classification"])
    title = param.String("")  # this is updated upon instantiation based on the task
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
            pn.WidgetBox(
                "### Add advanced configuration \n (will be configured on subsequent pages)",
                pn.widgets.Toggle.from_param(
                    self.param.show_survivor_config,
                    name="Configure Survivor",
                    button_type="primary",
                    button_style="outline",
                ),
                pn.widgets.Toggle.from_param(
                    self.param.show_reallabel_config,
                    name="Configure Reallabel",
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
                styles={"font-size": f"{self.blurb_font_size}px", "color": f"{self.color_light_gray}"},
                stylesheets=[self.widget_stylesheet],
            ),
            self.view_optional_configs,
            pn.Spacer(height=20),
            width=self.page_width,
            styles={"background": self.color_dark_blue},
        )


class RealLabelAppPipeline(RealLabelApp):
    """Superclass for adding Reallabel Panel app into the configuration pipeline"""

    task = param.Selector(default="object_detection", objects=["object_detection", "image_classification"])

    # dictionary for holding all the output configurations
    # this dictionary is passed between all the stages and
    # is used in the final stage to generate the json file
    output_test_stages = param.Dict({})

    # special parameter for dynamically setting the next stage
    # in this case, its set by output from previous stage
    next_parameter = param.Selector(default="Configure Survivor", objects=["Configure Survivor", "Finalize"])

    @param.output(task=param.Selector, output_test_stages=param.Dict)
    def output(self) -> tuple:
        """Output handler for passing variables from one pipeline page to another"""
        self._run_export()
        return self.task, self.output_test_stages


class SurvivorAppPipeline(SurvivorApp):
    """ "Superclass for adding Survivor Panel app into the configuration pipeline"""

    task = param.Selector(default="object_detection", objects=["object_detection", "image_classification"])

    # dictionary for holding all the output configurations
    # this dictionary is passed between all the stages and
    # is used in the final stage to generate the json file
    output_test_stages = param.Dict({})

    @param.output(task=param.Selector, output_test_stages=param.Dict)
    def output(self) -> tuple:
        """Output handler for passing variables from one pipeline page to another"""
        self._run_export()
        return self.task, self.output_test_stages


class FinalPage(StyleTemplate):
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
            styles={"font-size": f"{self.blurb_font_size}px", "color": f"{self.color_light_gray}"},
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
            styles={"background": self.color_dark_blue},
        )


class DatasetAnalysisConfigApp:
    """High level constructor for the Dataset Analysis Configuration App

    Creates a Panel Pipeline object from the individual panel apps and
    connects them together for sequential viewing.

    There are two task modes for this app - 'object_detection' and 'image_classification'.
    The 'image_classification' mode is TBD.

    To view the app (via notebook):

    >>> from jatic_ri.object_detection._panel.configurations.dataset_analysis_configuration import (
    ...     DatasetAnalysisConfigApp,
    ... )
    >>> import panel as pn
    >>> pn.extension()

    >>> app = DatasetAnalysisConfigApp(task="object_detection")
    >>> app.panel()
    """

    def __init__(self, task: str = "object_detection", **params: dict[str, object]) -> None:
        super().__init__(**params)

        # setup panel pipeline by adding individual apps and connecting them together
        self.pipeline = pn.pipeline.Pipeline(inherit_params=False, debug=True)

        if task == "object_detection":
            self.pipeline.add_stage(
                "Introduction", ConfigurationLandingPage(task=task), next_parameter="next_parameter"
            )
            self.pipeline.add_stage("Configure Reallabel", RealLabelAppPipeline, next_parameter="next_parameter")
            self.pipeline.add_stage("Configure Survivor", SurvivorAppPipeline)
            self.pipeline.add_stage("Finalize", FinalPage)
        else:
            raise RuntimeError(f"Sorry the task type, {task}, has not been coded yet. WOMP WOMP.")

        # setup nonlinear dag, actual path is dynamic based on choices made on the intro page
        self.pipeline.define_graph(
            {
                "Introduction": ("Configure Reallabel", "Configure Survivor", "Finalize"),
                "Configure Reallabel": ("Configure Survivor", "Finalize"),
                "Configure Survivor": "Finalize",
            }
        )

    def panel(self) -> pn.pipeline.Pipeline:
        """Visualize the DA OD configuration app"""
        return self.pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", "-t", type=str, default="object_detection", choices=["object_detection", "image_classification"]
    )
    args = parser.parse_args()
    app = DatasetAnalysisConfigApp(args.task)
    pn.panel(app.panel()).servable()
