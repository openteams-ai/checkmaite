"""Combined app for JATIC Console, integrating various configurations and final testbed."""

import json
import logging
from typing import Any

import panel as pn
import param

from checkmaite.ui._common.base_app import DEFAULT_STYLING, AppStyling, BaseApp
from checkmaite.ui._common.dataset_analysis_app import DAConfigurationLandingPage
from checkmaite.ui._common.model_evaluation_configuration import MEConfigurationLandingPage
from checkmaite.ui.configuration_pages.image_classification.nrtk_app import NRTKAppIC

# from checkmaite.ui.configuration_pages.image_classification.survivor_app import SurvivorAppIC
from checkmaite.ui.configuration_pages.image_classification.xaitk_app import XAITKAppIC
from checkmaite.ui.configuration_pages.object_detection.nrtk_app import NRTKAppOD

# from checkmaite.ui.configuration_pages.object_detection.reallabel_app import RealLabelApp
# from checkmaite.ui.configuration_pages.object_detection.survivor_app import SurvivorAppOD
from checkmaite.ui.configuration_pages.object_detection.xaitk_app import XAITKAppOD
from checkmaite.ui.dashboards.dataset_analysis_dashboard import DatasetAnalysisDashboard
from checkmaite.ui.dashboards.model_evaluation_dashboard import ModelEvaluationTestbed

# Parameters passed between pipeline pages:
# * next_parameter
# * task
# * output_test_stages
# * local
# * workflow


class LandingPage(BaseApp):
    """Landing page for the JATIC Console app."""

    # special parameter for dynamically setting the next stage on THIS PAGE
    next_parameter = param.Selector(
        default="DAConfigurationLandingPage",
        objects=[
            "DAConfigurationLandingPage",
            "MEConfigurationLandingPage",
            "DatasetAnalysisDashboard",
            "ModelEvaluationTestbed",
        ],
    )

    def __init__(self, workflow: str, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, Any]) -> None:
        self.file_dropper = pn.widgets.FileDropper(multiple=False, stylesheets=[styles.css_filedropper])
        # ME/DA toggle
        self.me_da_toggle = pn.widgets.RadioButtonGroup(
            value=workflow,
            options={"Model Evaluation": "model_evaluation", "Dataset Analysis": "dataset_analysis"},
            button_type="primary",
            width=350,
            stylesheets=[styles.css_button],
        )

        super().__init__(styles, **params)
        self.workflow = workflow
        self._update_next_parameter_for_config()  # pyright: ignore[reportCallIssue]  # set the next parameter based on the workflow

        button_width = 195
        self.od_button = pn.widgets.Button(
            name="Object Detection",
            button_type="primary",
            width=button_width,
            stylesheets=[self.styles.css_button],
            icon="pencil",
        )
        self.od_button.on_click(self.od_button_callback)
        self.ic_button = pn.widgets.Button(
            name="Image Classification",
            button_type="primary",
            width=button_width,
            stylesheets=[self.styles.css_button],
            icon="pencil",
        )
        self.ic_button.on_click(self.ic_button_callback)

    @param.output(
        next_parameter=param.String,
        task=param.String,
        output_test_stages=param.Dict,
        local=param.Boolean,
        testbed_config=param.Dict,
    )
    def output(self) -> tuple:
        """Output parameters for the pipeline.
        next_parameter is only provided for stages that need to set the next stage dynamically, but it
        must be a valid option on the next stage."""
        # raise NotImplementedError("THIS SHOULD BE IMPLEMENTED IN THE HIGHER LEVEL CLASS")

        followup_stage = "DatasetAnalysisDashboard" if self.workflow == "dataset_analysis" else "ModelEvaluationTestbed"
        return (
            followup_stage,
            self.task,
            self.output_test_stages,
            self.local,
            self.testbed_config,
        )

    @param.depends("workflow", watch=True)
    def _update_next_parameter_for_config(self) -> None:
        """Update the next parameter based on the current workflow"""
        if self.workflow == "dataset_analysis":
            self.next_parameter = "DAConfigurationLandingPage"
        elif self.workflow == "model_evaluation":
            self.next_parameter = "MEConfigurationLandingPage"
        else:
            raise ValueError(f"Invalid workflow: {self.workflow}")

    def ic_button_callback(self, event: param.parameterized.Event) -> None:  # noqa: ARG002
        """Callback for the Image Classification button"""
        # set task
        self.task = "image_classification"
        # move to next stage
        self.ready = True

    def od_button_callback(self, event: param.parameterized.Event) -> None:  # noqa: ARG002
        """Callback for the Object Detection button"""
        # set task
        self.task = "object_detection"
        # move to next stage
        self.ready = True

    @pn.depends("me_da_toggle.value", watch=True)
    def me_da_toggle_callback(self) -> None:
        """Callback for the ME/DA toggle to set the workflow"""
        self.workflow = self.me_da_toggle.value

    @pn.depends("file_dropper.value", watch=True)
    def file_dropper_callback(self, event: param.Event | None = None) -> None:  # noqa: ARG002
        """Callback for the file dropper"""
        # our file dropper widget only accepts one file so we just need to grab the
        # the first value from the dictionary
        if not isinstance(self.file_dropper.value, dict) or not self.file_dropper.value:
            logging.warning("File dropper value is either not a valid dictionary or is empty.")
            return
        value = next(iter(self.file_dropper.value.values()))
        self.output_test_stages = json.loads(value)
        self.task = self.output_test_stages.get("task", "object_detection")  # default to OD if not set

        self.next_parameter = (
            "ModelEvaluationTestbed" if self.workflow == "model_evaluation" else "DatasetAnalysisDashboard"
        )
        self.ready = True

    def view_title_row(self) -> pn.Column:
        """View of the title row which contains the task (OD/IC),
        objective (Model Evaluation/Dataset Analysis), and a
        brief description of the app"""
        return pn.Column(
            pn.Spacer(height=10),
            pn.pane.Markdown(
                "JATIC Console",
                styles=self.styles.style_text_h2,
                stylesheets=[self.styles.css_paragraph],
            ),
            pn.pane.Markdown(
                "A unified user interface for configuring and running the JATIC tool suite.",
                styles=self.styles.style_text_body2,
                stylesheets=[self.styles.css_paragraph],
            ),
            pn.Spacer(height=10),
        )

    def view_workflow_row(self) -> pn.Row:
        """View of the workflow row which contains the ME/DA toggle"""
        return pn.Row(
            pn.Column(
                pn.Spacer(height=20),
                pn.pane.Markdown(
                    "1. Select Pipeline",
                    styles=self.styles.style_text_h2,
                    stylesheets=[self.styles.css_paragraph],
                ),
                pn.Row(
                    pn.Spacer(width=38),  # Line up with "S" in "1. Select Pipeline"
                    pn.Column(
                        pn.pane.Markdown(
                            ("**Model Evaluation**: Analyze model performance, robustness, and explainability."),
                            width=380,
                            styles=self.styles.style_text_body2,
                            stylesheets=[self.styles.css_paragraph],
                        ),
                        pn.pane.Markdown(
                            (
                                "**Dataset Analysis**: Understand and improve dataset quality by analyzing biases, "
                                "label errors, and data distributions."
                            ),
                            width=380,
                            styles=self.styles.style_text_body2,
                            stylesheets=[self.styles.css_paragraph],
                        ),
                    ),
                ),
                pn.Spacer(height=20),
            ),
            pn.Spacer(width=10),
            pn.Column(
                pn.Spacer(height=35),
                self.me_da_toggle,
            ),
        )

    def view_config_options_row(self) -> pn.Column:
        """View of the configuration options row"""
        column_width = 400

        left_column = pn.Column(
            pn.Spacer(height=50),
            pn.pane.Markdown(
                "Load configuration",
                styles=self.styles.style_text_h2,
                sizing_mode="stretch_width",
                stylesheets=[self.styles.css_paragraph, self.styles.css_center_text],
            ),
            pn.pane.Markdown(
                "Load an existing JSON test parameterization and execute test sequence",
                styles=self.styles.style_text_body2,
                sizing_mode="stretch_width",
                stylesheets=[self.styles.css_paragraph, self.styles.css_center_text],
            ),
            pn.Row(
                pn.layout.HSpacer(),
                self.file_dropper,
                pn.layout.HSpacer(),
            ),
            width=column_width,
        )

        right_column = pn.Column(
            pn.Spacer(height=50),
            pn.pane.Markdown(
                "Create a new configuration",
                styles=self.styles.style_text_h2,
                sizing_mode="stretch_width",
                stylesheets=[self.styles.css_paragraph, self.styles.css_center_text],
            ),
            pn.pane.Markdown(
                "Generate the required parameterization and execute test sequence",
                styles=self.styles.style_text_body2,
                sizing_mode="stretch_width",
                stylesheets=[self.styles.css_paragraph, self.styles.css_center_text],
            ),
            pn.Spacer(height=37),
            pn.Row(
                pn.layout.HSpacer(),
                self.od_button,
                pn.layout.Spacer(width=10),
                self.ic_button,
                pn.layout.HSpacer(),
            ),
            width=column_width,
        )

        config_opts_view = pn.Row(
            left_column,
            pn.Spacer(styles={"background": self.styles.color_gray_500}, sizing_mode="stretch_height", width=2),
            pn.Spacer(width=10),
            right_column,
            height=270,
        )

        return pn.Column(
            pn.Spacer(height=20),
            pn.pane.Markdown(
                "2. Select a configuration option",
                styles=self.styles.style_text_h2,
                stylesheets=[self.styles.css_paragraph],
            ),
            pn.Spacer(height=20),
            config_opts_view,
            pn.Spacer(height=20),
        )

    def panel(self) -> pn.Column:
        """Visualize the Landing page app"""
        return pn.Column(
            self.view_header(display_task=False),
            self.view_title_row,
            self.horizontal_line(),
            self.view_workflow_row,
            self.horizontal_line(),
            self.view_config_options_row,
            self.horizontal_line(),
        )


class FullApp(BaseApp):
    """High level constructor for the fully integrated `checkmaite` App

    Creates a Panel Pipeline object from the individual panel apps and
    connects them together for sequential viewing.

    There are two task modes for this app - 'object_detection' and 'image_classification'.

    To view the app, deploy the script `/ui/app.py via the terminal:

    ```bash
    poetry run panel serve src/checkmaite/ui/app.py --show
    ```
    """

    def __init__(
        self, workflow: str = "dataset_analysis", styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]
    ) -> None:
        super().__init__(styles, **params)
        self.workflow = workflow

        # setup panel pipeline by adding individual apps and connecting them together
        self.pipeline = pn.pipeline.Pipeline(inherit_params=False, debug=True)

        self.pipeline.add_stage(
            "Introduction",
            LandingPage(workflow, styles, **params),
            next_parameter="next_parameter",
            ready_parameter="ready",
            auto_advance=True,
        )

        self.pipeline.add_stage(
            "DAConfigurationLandingPage",
            DAConfigurationLandingPage,
            next_parameter="next_parameter",
            ready_parameter="ready",
            auto_advance=True,
        )
        self.pipeline.add_stage(
            "DatasetAnalysisDashboard",
            DatasetAnalysisDashboard,
            ready_parameter="ready",
            auto_advance=True,
        )

        self.pipeline.add_stage(
            "MEConfigurationLandingPage",
            MEConfigurationLandingPage,
            next_parameter="next_parameter",
            ready_parameter="ready",
            auto_advance=True,
        )
        self.pipeline.add_stage(
            "ModelEvaluationTestbed", ModelEvaluationTestbed, ready_parameter="ready", auto_advance=True
        )

        # self.pipeline.add_stage(
        #     "Configure Reallabel",
        #     RealLabelApp,
        #     next_parameter="next_parameter",
        #     ready_parameter="ready",
        #     auto_advance=True,
        # )
        # self.pipeline.add_stage("Configure SurvivorOD", SurvivorAppOD, ready_parameter="ready", auto_advance=True)
        self.pipeline.add_stage(
            "Configure NRTKOD", NRTKAppOD, next_parameter="next_parameter", ready_parameter="ready", auto_advance=True
        )
        self.pipeline.add_stage("Configure XAITKOD", XAITKAppOD, ready_parameter="ready", auto_advance=True)

        # self.pipeline.add_stage("Configure SurvivorIC", SurvivorAppIC, ready_parameter="ready", auto_advance=True)
        self.pipeline.add_stage(
            "Configure NRTKIC", NRTKAppIC, next_parameter="next_parameter", ready_parameter="ready", auto_advance=True
        )
        self.pipeline.add_stage("Configure XAITKIC", XAITKAppIC, ready_parameter="ready", auto_advance=True)

        self.pipeline.define_graph(
            {
                "Introduction": (
                    "DAConfigurationLandingPage",
                    "MEConfigurationLandingPage",
                    "DatasetAnalysisDashboard",
                    "ModelEvaluationTestbed",
                ),
                # DA
                "DAConfigurationLandingPage": (
                    # "Configure Reallabel",
                    # "Configure SurvivorOD",
                    # "Configure SurvivorIC",
                    "DatasetAnalysisDashboard",
                ),
                # ME
                "MEConfigurationLandingPage": (
                    "Configure NRTKOD",
                    "Configure XAITKOD",
                    "Configure NRTKIC",
                    "Configure XAITKIC",
                    "ModelEvaluationTestbed",
                ),
                # DA IC - Survivor
                # "Configure SurvivorIC": "DatasetAnalysisDashboard",
                # DA OD - Reallabel
                # "Configure Reallabel": (
                #    "Configure SurvivorOD",
                #    "DatasetAnalysisDashboard",
                # ),
                # DA OD - Survivor
                # "Configure SurvivorOD": "DatasetAnalysisDashboard",
                # ME OD - NRTK
                "Configure NRTKOD": (
                    "Configure XAITKOD",
                    "ModelEvaluationTestbed",
                ),
                # ME OD - XAITK
                "Configure XAITKOD": "ModelEvaluationTestbed",
                # ME IC - NRTK
                "Configure NRTKIC": (
                    "Configure XAITKIC",
                    "ModelEvaluationTestbed",
                ),
                # ME IC - XAITK
                "Configure XAITKIC": "ModelEvaluationTestbed",
            }
        )

    def panel(self) -> pn.Column:
        """Visualize the Full app app"""
        return pn.Column(
            self.pipeline.stage,
            styles={"background": self.styles.color_main_bg},
            width=self.styles.app_width,
            sizing_mode="stretch_height",
        )
