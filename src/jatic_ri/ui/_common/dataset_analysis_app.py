"""Dataset Analysis Configuration for Object Detection and Image Classification Panel Application

Object detection includes Reallabel, Survivor, and the Dataeval capabilities
Image Classification includes Survivor and the Dataeval capabilities
"""

import argparse
import textwrap
from typing import Any

import panel as pn
import param

from jatic_ri.ui._common.base_app import DEFAULT_STYLING, AppStyling, BaseApp

try:
    import reallabel  # noqa: F401

    HAS_REALLABEL = True
except ImportError:
    HAS_REALLABEL = False

try:
    import survivor  # noqa: F401

    HAS_SURVIVOR = True
except ImportError:
    HAS_SURVIVOR = False


class DAConfigurationLandingPage(BaseApp):
    """Initial landing page for the DA OD configuration app"""

    task = param.Selector(default="object_detection", objects=["object_detection", "image_classification"])

    # toggles for displaying optional pages
    show_survivor_config = param.Boolean(default=False)
    show_reallabel_config = param.Boolean(default=False)

    # dictionary for holding all the output configurations
    # this dictionary is passed between all the stages and
    # is used in the final stage to generate the json file
    output_test_stages = param.Dict({})

    # special parameter for dynamically setting the next stage on THIS PAGE
    next_parameter = param.Selector(
        default="DatasetAnalysisDashboard",
        objects=["Configure Reallabel", "Configure SurvivorOD", "Configure SurvivorIC", "DatasetAnalysisDashboard"],
    )

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, Any]) -> None:
        super().__init__(styles, **params)
        self.title = f"Dataset analysis configuration: {self.task.replace('_', ' ').title()}"
        # make sure that the next stage matches the defaults
        self._update_next_parameter()

        # dataeval configuration
        # bias - parity, coverage, diversity, balance
        self.bias = pn.widgets.Checkbox(stylesheets=[self.styles.css_checkbox])
        # shift - Out-of-distributation (AE, VAE) and drift (CMV, MMD, KS)
        self.shift = pn.widgets.Checkbox(stylesheets=[self.styles.css_checkbox])
        # cleaning - duplicates detection and outlier detection (dimensional stats, pixel stats, visual stats)
        self.cleaning = pn.widgets.Checkbox(stylesheets=[self.styles.css_checkbox])
        # feasibility
        self.feasibility = pn.widgets.Checkbox(stylesheets=[self.styles.css_checkbox])

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
                if self.task == "object_detection":
                    self.next_parameter = "Configure SurvivorOD"
                else:
                    self.next_parameter = "Configure SurvivorIC"

            # not reallabel and not survivor
            else:
                self.next_parameter = "DatasetAnalysisDashboard"

    @param.output(next_parameter=param.Selector, task=param.String, output_test_stages=param.Dict, local=param.Boolean)
    def output(self) -> tuple:
        """Update the output configuration based on the settings in
        the UI and then output to the next stage.

        The output that is in the returned tuple become variables in the next stage defined
        by the variable names and types listed in the decorator. `followup_next_parameter`
        is the variable that will be assigned as `next_parameter` in the following stage.

        If the survivor toggle is triggered, set the next stage as `survivor`.
        If not, set the next stage as `DatasetAnalysisDashboard`
        """
        # tell the next stage what it's next stage is going to be
        if self.show_survivor_config:
            if self.task == "object_detection":
                followup_next_parameter = "Configure SurvivorOD"
            else:
                followup_next_parameter = "Configure SurvivorIC"
        else:
            followup_next_parameter = "DatasetAnalysisDashboard"

        self.output_test_stages = {
            "task": self.task,
        }
        if self.bias.value:
            self.output_test_stages.update(
                {
                    "bias": {
                        "TYPE": "DatasetBiasTestStage",
                        "CONFIG": {},
                    },
                }
            )
        if self.feasibility.value:
            self.output_test_stages.update(
                {
                    "feasibility": {
                        "TYPE": "DatasetFeasibilityTestStage",
                        "CONFIG": {},
                    },
                }
            )
        if self.shift.value:
            self.output_test_stages.update(
                {
                    "shift": {
                        "TYPE": "DatasetShiftTestStage",
                        "CONFIG": {},
                    },
                }
            )
        if self.cleaning.value:
            self.output_test_stages.update(
                {
                    "cleaning": {
                        "TYPE": "DatasetCleaningTestStage",
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
        """View of the analysis tools row of the app."""

        # build list of rows, starting & ending with spacers
        tools_viewable = [
            pn.Spacer(height=24),
            self._generate_checkbox_subsection(
                self.cleaning,
                "Cleaning",
                "Identify potential image quality issues.",
            ),
            pn.Spacer(height=12),
            self._generate_checkbox_subsection(
                self.shift,
                "Shift",
                "Estimate the dataset drift and out-of-distribution fraction.",
            ),
            pn.Spacer(height=12),
            self._generate_checkbox_subsection(
                self.bias,
                "Bias",
                "Measure sampling bias and correlation.",
            ),
        ]

        # task-specific tools
        if self.task == "image_classification":
            tools_viewable += [
                pn.Spacer(height=12),
                self._generate_checkbox_subsection(
                    self.feasibility,
                    "Feasibility",
                    "Estimate asymptotic model performance.",
                ),
            ]

        # add survivor section
        if HAS_SURVIVOR:
            survivor_cb = pn.widgets.Checkbox.from_param(
                self.param.show_survivor_config,
                name="",
                stylesheets=[self.styles.css_checkbox],
            )

            # add survivor checkbox
            tools_viewable += [
                pn.Spacer(height=12),
                self._generate_checkbox_subsection(
                    survivor_cb,
                    "Survivor",
                    "Identify pieces of data (e.g. images with ground truth labels) "
                    "in an evaluation dataset that are most valuable for ranking models.",
                    requires_config=True,
                    section_height=76,
                ),
            ]

        # add reallabel section
        if self.task == "object_detection" and HAS_REALLABEL:
            reallabel_checkbox = pn.widgets.Checkbox.from_param(
                self.param.show_reallabel_config,
                name="",
                stylesheets=[self.styles.css_checkbox],
            )

            tools_viewable += [
                pn.Spacer(height=12),
                self._generate_checkbox_subsection(
                    reallabel_checkbox,
                    "Reallabel",
                    "Identify potentially missing and erroneous ground truth labels in object-detection datasets.",
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
                "Dataset Analysis Configuration",
                styles=self.styles.style_text_h2,
                stylesheets=[self.styles.css_paragraph],
            ),
        )

        analysis_row = pn.Row(
            pn.Column(
                pn.pane.Markdown(
                    "Analysis Tool Selection",
                    styles=self.styles.style_text_h3,
                    stylesheets=[self.styles.css_paragraph],
                ),
                pn.pane.Markdown(
                    textwrap.dedent(
                        """
                        Setup your dataset analysis configuration to include tools from
                        various JATIC products. These tools are designed to assess the
                        quality, consistency and structure of your dataset.

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
                    # configurable_row,
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
    parser.add_argument(
        "--task", "-t", type=str, default="object_detection", choices=["object_detection", "image_classification"]
    )
    args = parser.parse_args()
    app = DAConfigurationLandingPage(task=args.task)
    app.panel().servable()
