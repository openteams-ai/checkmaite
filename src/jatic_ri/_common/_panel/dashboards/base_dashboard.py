"""Base Dashboard for Testbeds
This dashboard object is subclassed by the ME and DA Dashboards."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn
import param
from bokeh.models import HTMLTemplateFormatter
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri import PACKAGE_DIR
from jatic_ri._common.test_stages.interfaces.plugins import (
    EvalToolPlugin,
    MetricPlugin,
    MultiModelPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    ThresholdPlugin,
    TwoDatasetPlugin,
)
from jatic_ri._common.test_stages.interfaces.test_stage import TestStage
from jatic_ri.image_classification.models import SUPPORTED_MODELS as SUPPORTED_MODELS_IC
from jatic_ri.image_classification.models import SUPPORTED_TORCHVISION_MODELS as SUPPORTED_TORCHVISION_MODELS_IC
from jatic_ri.object_detection.models import SUPPORTED_MODELS as SUPPORTED_MODELS_OD
from jatic_ri.object_detection.models import SUPPORTED_TORCHVISION_MODELS as SUPPORTED_TORCHVISION_MODELS_OD
from jatic_ri.object_detection.models import SUPPORTED_VISDRONE_MODELS as SUPPORTED_VISDRONE_MODELS_OD
from jatic_ri.util.dashboard_utils import create_download_link, rehydrate_test_stage_ic, rehydrate_test_stage_od
from jatic_ri.util.evaluation import EvaluationTool

JATIC_LOGO_PATH = PACKAGE_DIR.joinpath(
    "_sample_imgs",
    "JATIC_Logo_Acronym_Spelled_Out_RGB_white_type.svg",
)

pn.extension("tabulator", "floatpanel")

logger = logging.getLogger()


# mapping between the visible dataset labels in the UI and
# the underlying wrapper class
DATASET_LABEL_MAP_OD = {
    "Coco dataset": "CocoDetectionDataset",
    "Yolo dataset": "YoloDetectionDataset",
}

DATASET_LABEL_MAP_IC = {
    "Yolo dataset": "YoloClassificationDataset",
}

METRICS_LABEL_MAP_OD = {
    "mAP": "map50_torch_metric_factory",
}

METRICS_LABEL_MAP_IC = {
    "Accuracy": "accuracy_multiclass_torch_metric_factory",
    "F1 Score": "f1score_multiclass_torch_metric_factory",
}

EVAL_TOOL_CACHE_DIR = ".eval_tool"


class BaseDashboard(param.Parameterized):
    """Base Dashboard/Testbed. This class is inherited by the
    dataset analysis and model evaluation dashboard classes. It contains
    css styling, common widgets, and common functions"""

    title = param.String(default="Object Detection Testing Pipeline")
    title_font_size = param.Integer(default=24)
    task = param.String(default="object_detection")

    # Dictionary for holding test stages
    test_stages = param.Dict(default={})

    # trigger for redrawing the set of model widgets
    redraw_models_trigger = param.Integer(default=0)
    # Dictionary for holding loaded models
    loaded_models = param.Dict(default={})
    # multi model visible
    multi_model_visible = param.Boolean(default=True)

    # Dictionary for holding loaded datasets
    loaded_datasets = param.Dict({})
    # boolean for viewing dataset 2 widgets
    dataset_2_visible = param.Boolean(default=False)

    # Input for the target threshold
    threshold = param.String(default="0.5")

    # This parameter configures the dashboard to use caching.  Caching can occur both at the
    # test stage output and in the underlying EvaluationTool.  The full flow looks like this:
    #   1) TestStage.run() will first check for a previously cached output for the given arguments.
    #   Each tool's test stage may implement this differently. If the stage cache hits, flow ends here.
    #   2) If no cache hit, the TestStage's _run implementation executes.
    #   3) If the test stage implements the EvalToolPlugin, it will have loaded an EvaluationTool object
    #      When 'use_caches' is True, that object will have a backend cache.
    #   4) The EvaluationTool checks for a cache on two levels:
    #           Predictions - i.e. model + dataset + batch size.  Both `predict()` and `evaluate()` will check
    #           and use if available.  Since `evaluate()` runs `predict()`, the same test stage can run multiple
    #           metrics against a given model and dataset (and batch size) without having to redo the inference.
    #           Evaluations - i.e. model + dataset + metric + batch size.  Both `evaluate()` and `compute_metric()`
    #           will check for a cache.
    #   5) Only after a cache miss for `predict()` or `evaluate()` will the actual execution (model inference and/or
    #      metric calculation) occur, with the results both cached at the EvaluationTool level and then passed back
    #      to the TestStage._run(), which will in turn cache the output results for subsequent calls.
    use_caches = param.Boolean(default=True)

    # Input for loading the pipeline config
    config_file = pn.widgets.FileInput(accept=".json")

    # Table for storing the results
    results_df = param.DataFrame(pd.DataFrame({}))

    # Location for storing all output
    output_dir = param.Path(default=Path.cwd(), check_exists=False)

    # Text of status bar
    status_text = param.String("Waiting for input...")

    def __init__(self, **params: dict[str, Any]) -> None:
        super().__init__(**params)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.app_width = 1280
        # style guide
        self.color_blue_900 = "#001B4D"  # blue-900
        self.color_blue_800 = "#0F388A"  # blue-800
        self.color_blue_700 = "#1550C1"  # blue-700
        self.color_blue_600 = "#195FE6"  # blue-600
        self.color_blue_500 = "#5284E5"  # blue-500
        self.color_blue_400 = "#770FEE"  # blue-400
        self.color_blue_300 = "#A3BEF5"  # blue-300
        self.color_blue_200 = "#D5E0F6"  # blue-200
        self.color_blue_100 = "#EDF2FD"  # blue-100
        self.color_white = "#FFFFFF"  # pure-white
        self.color_gray_900 = "#00050A"  # gray-900
        self.color_gray_800 = "#1E2C3E"  # gray-800
        self.color_gray_700 = "#415062"  # gray-700
        self.color_gray_600 = "#788BA5"  # gray-600
        self.color_gray_500 = "#BBC9DD"  # gray-500
        self.color_gray_400 = "#DDE4EE"  # gray-400
        self.color_gray_300 = "#F1F4F9"  # gray-300
        self.color_gray_200 = "#F8FAFC"  # gray-200
        self.color_main_bg = self.color_gray_200

        self.font_family = "'Helvetica Neue', 'Arial'"
        self.style_text_h1 = {
            "font-size": "24px",
            "font-family": self.font_family,
            "font-weight": "bold",
            "color": self.color_gray_900,
        }
        self.style_text_h2 = {
            "font-size": "18px",
            "font-family": self.font_family,
            "font-weight": "bold",
            "color": self.color_gray_900,
        }
        self.style_text_h3 = {
            "font-size": "13px",
            "font-family": self.font_family,
            "font-weight": "bold",
            "color": self.color_gray_900,
        }
        self.style_text_subtitle = {
            "font-size": "12px",
            "font-family": self.font_family,
            "font-weight": "semibold",
            "color": self.color_gray_900,
        }
        self.style_text_body1 = {
            "font-size": "13px",
            "font-family": self.font_family,
            "color": self.color_gray_900,
        }
        self.style_text_body2 = {
            "font-size": "12px",
            "font-family": self.font_family,
            "color": self.color_gray_700,
        }

        self.style_border = {
            "background-color": self.color_white,
            "border-color": self.color_gray_500,
            "border-width": "thin",
            "border-style": "solid",
            "border-radius": "8px",
        }

        # removes paragraph margins and overrides font-family
        self.css_paragraph = """
            :host p {
              margin: 0px;
              font-family: "Helvetica Neue", "Arial";
            }
            """
        # adjust the dimensions of a checkbox widget
        self.css_checkbox = """
            input {
                height: 16px;
                width: 16px;
            }
            """
        # adjust button styling
        self.css_button = f"""
            :host(.solid) .bk-btn.bk-btn-default {{
              background-color: {self.color_blue_500};
              color: #FFFFFF;
            }}
            """
        # adjust switch toggle styling
        self.css_switch = f"""
            :host(.active) .knob {{
                background-color:{self.color_blue_500};
            }}
            :host(.active) .bar {{
                background-color: {self.color_blue_200};
            }}
            """

        self.width_input_default = 580
        self.width_subwidget_offset = 20  # path widget offset from type dropdown
        self.dropdown_height = "20px"
        self.css_dropdown = f"""
                label {{
                  color: {self.color_gray_700};  /* widget title color */
                }}

                select:not([multiple]).bk-input, select:not([size]).bk-input {{
                  height: {self.dropdown_height};  /* height of selection box */
                  color: {self.color_gray_900};  /* color of text in selection box */
                }}
                """

        self.css_config_input = f"""
                .bk-input {{
                  color: {self.color_gray_900} /* text color */
                }}

                input[type='file'] {{
                    height: 40px;  /* widget height */
                    border: 1px dashed;
                    padding: 0;  /* this is not being obeyed */
                }}
                """

        self.css_tabulator_table = f"""
                .tabulator-row.tabulator-selectable:hover {{
                  background-color: {self.color_blue_200} !important;  /* bg color for row hover */
                }}
                host: .tabulator-row.tabulator-selected {{
                  background-color: {self.color_blue_400} !important; /* bg color for row selection */
                }}
                .tabulator-row {{
                  background-color: {self.color_white} !important; /* bg color for all other rows */
                  border: none !important;  /* horizontal border between rows */
                }}
                .tabulator .tabulator-header .tabulator-col {{
                  background-color: {self.color_gray_300} !important; /* header bg color */
                  color: {self.color_gray_900} !important; /* header font color */
                  font-size: 13px; !important;  /* header font size */
                  font-family: "Helvetica Neue", "Arial";  /* header font types */
                  font-weight: 500;
                  border-bottom: 1px solid {self.color_gray_500};
                }}
                /* table outer border styles */
                :host .tabulator {{
                  border-color: {self.color_gray_500} !important;
                  border-width: 1px !important;
                  border-style: solid;
                  border-radius: 5px;
                }}
                .tabulator-row .tabulator-cell {{
                  border: none !important;  /* vertical border between cells */
                }}
                """

        # metric dropdown widget, options populated in _update_task_related_objects()
        self.metric_selector = pn.widgets.Select(
            name="Evaluation Metric",
            width=self.width_input_default,
            stylesheets=[self.css_dropdown],
        )
        self._update_task_related_objects()
        # holder for the model widgets (for dynamic handling of the number of models)
        self.model_widgets = {}
        # ensure we don't visualize multiple models (no add model button).
        # Must be set before the add_model_button_callback.
        self.multi_model_visible = False
        # button for adding another model (adds two widgets per model)
        self.add_model = pn.widgets.Button(name="Add model")
        self.add_model.on_click(self.add_model_button_callback)
        self.add_model_button_callback(None)  # trigger on init to build the first set

        # this is set here to avoid a race condition since both config_file and config_input is in `self`
        self.config_file.stylesheets = [self.css_config_input]

        # button to run the analysis
        self.run_analysis_button = pn.widgets.Button(name="Run Analysis", button_type="primary", disabled=False)

        self.dataset_1_selector = pn.widgets.Select(
            options=["Select Dataset Type", *list(self.dataset_label_map.keys())],
            width=self.width_input_default,
            name="Dataset",
            stylesheets=[self.css_dropdown],
            value="Select Dataset Type",
        )
        self.dataset_2_selector = pn.widgets.Select(
            options=["Select Dataset Type", *list(self.dataset_label_map.keys())],
            width=self.width_input_default,
            name="Comparison Dataset",
            stylesheets=[self.css_dropdown],
            value="Select Dataset Type",
        )
        self.dataset_1_split_path = pn.widgets.TextInput(
            name="Path of split folder",
            placeholder="Path of split folder",
            width=self.width_input_default - self.width_subwidget_offset,
            stylesheets=[self.css_dropdown],
        )
        self.dataset_2_split_path = pn.widgets.TextInput(
            name="Path of split folder",
            placeholder="Path of split folder",
            width=self.width_input_default - self.width_subwidget_offset,
            stylesheets=[self.css_dropdown],
        )
        self.dataset_1_metadata_path = pn.widgets.TextInput(
            name="Path to metadata file",
            placeholder="Path to metadata file",
            width=self.width_input_default - self.width_subwidget_offset,
            stylesheets=[self.css_dropdown],
        )
        self.dataset_2_metadata_path = pn.widgets.TextInput(
            name="Path to metadata file",
            placeholder="Path to metadata file",
            width=self.width_input_default - self.width_subwidget_offset,
            stylesheets=[self.css_dropdown],
        )

        # controls for viewing configuration
        self.view_config_btn = pn.widgets.ButtonIcon(
            icon="eye",
            size="2em",
            description="View loaded config",
            disabled=False,
            styles={"color": "black"},
        )
        self.view_config_btn.on_click(self._on_view_config_callback)
        # create invisible container to put floatpanel "in", cannot be completely empty
        self.config_floatpanel_container = pn.Column(pn.widgets.Checkbox(visible=False))

    @param.depends("task", watch=True)
    def _update_task_related_objects(self, event=None) -> None:  # noqa: ANN001, ARG002
        if self.task == "object_detection":
            self.dataset_label_map = DATASET_LABEL_MAP_OD
            self.metric_selector.options = list(METRICS_LABEL_MAP_OD.keys())
            self.metric_selector.value = list(METRICS_LABEL_MAP_OD.keys())[0]
            self.metric_label_map = METRICS_LABEL_MAP_OD
            self.model_label_map = {value.replace("_", " "): key for key, value in SUPPORTED_MODELS_OD.items()}
            self.torchvision_models = SUPPORTED_TORCHVISION_MODELS_OD
            self.visdrone_models = SUPPORTED_VISDRONE_MODELS_OD

        elif self.task == "image_classification":
            self.dataset_label_map = DATASET_LABEL_MAP_IC
            self.metric_selector.options = list(METRICS_LABEL_MAP_IC.keys())
            self.metric_selector.value = list(METRICS_LABEL_MAP_IC.keys())[0]
            self.metric_label_map = METRICS_LABEL_MAP_IC
            self.model_label_map = {value.replace("_", " "): key for key, value in SUPPORTED_MODELS_IC.items()}
            self.torchvision_models = SUPPORTED_TORCHVISION_MODELS_IC

    @pn.depends("config_file.value", watch=True)
    def _update_default_config(self, event=None) -> None:  # noqa: ANN001, ARG002
        logger.debug("update default config")
        config = json.loads(self.config_file.value)
        success = self.load_pipeline(config)
        if success:
            self.run_analysis_button.disabled = False

    def _on_view_config_callback(self, event=None) -> None:  # noqa: ANN001, ARG002
        """Callback that fires when the "view config" button is clicked.
        This adds a float panel widget to the visualized (otherwise empty) container"""
        floatpanel = pn.layout.FloatPanel(
            pn.pane.JSON(self.config_file.value, depth=-1, height=200),
            name="Loaded configurations",
            margin=20,
            contained=False,
            position="center",
            config={
                "headerControls": {
                    "minimize": "remove",
                    "maximize": "remove",
                    "smallify": "remove",
                },
                "resizeit": False,
            },
            theme=f"{self.color_blue_300} filledLight",
        )
        floatpanel.param.watch(self._on_config_panel_close_callback, "status")
        self.config_floatpanel_container.append(floatpanel)

    def _on_config_panel_close_callback(self, event=None) -> None:  # noqa: ANN001
        """Callback that fires when floating config viewer panel is closed
        This removes the float panel from the visualized container."""
        if event.obj.status == "closed":
            self.config_floatpanel_container.remove(event.obj)

    def _on_model_type_change(self, event) -> None:  # noqa: ANN001 # pragma: no cover
        """Callback listening for changes to all model selector widgets.
        When called, this uses the name of the model selector widget
        to look up the model_weights_path widget and change the value of the placeholder
        text. This allows us to dynamically change the placeholder text
        to guide users to upload the proper type of file.
        """

        if self.model_label_map[event.obj.value] in self.torchvision_models:
            self.model_widgets[event.obj.name][
                "model_weights_path"
            ].placeholder = "Filepath to model weights file containing the state dict"
            # self.model_widgets[event.obj.name][
            #     "tooltip"
            # ].value = "Supported extensions are .pt and .pth. A config.json should exist in the same directory )"
        else:
            self.model_widgets[event.obj.name]["model_weights_path"].placeholder = "Select file"
            # self.model_widgets[event.obj.name]["tooltip"].value = "Select file"

    def _remove_model_widget(self, event) -> None:  # noqa: ANN001
        del self.model_widgets[event.obj.description]
        # redraw the model widgets
        self.redraw_models_trigger += 1

    def add_model_button_callback(self, event) -> None:  # noqa: ANN001, ARG002
        """Callback that runs when the add model button is clicked
        When called, this adds a two new widgets for setting a new model
        """
        # construct the dropdown for a new model widgets
        model_weights_path = pn.widgets.TextInput(
            name="Path to model weights",
            placeholder="Path to weights file",
            width=self.width_input_default,
            stylesheets=[self.css_dropdown],
        )
        selector_label = f"Model {len(self.model_widgets) + 1} type"
        model_selector = pn.widgets.Select(
            name=selector_label,
            options=["Select Model type", *list(self.model_label_map.keys())],
            width=self.width_input_default,
            stylesheets=[self.css_dropdown],
            value="Select Model type",
        )
        # tooltip = pn.widgets.TooltipIcon(
        #     value="Path to model weights file (config.json should exist in the same directory)"
        # )
        # link a callback method to the model dropdown so that we
        # can change the placeholder text when the model type is changed
        model_selector.param.watch(self._on_model_type_change, ["value"], onlychanged=False)

        if self.multi_model_visible:
            # button for removing model from widget list
            remove_model_button = pn.widgets.ButtonIcon(
                icon="circle-x", size="2em", name="", description=selector_label
            )

            remove_model_button.param.watch(self._remove_model_widget, ["clicks"])
        else:
            remove_model_button = None

        # store the set of widgets for this model in a dict for reference later
        self.model_widgets[model_selector.name] = {
            "model_selector": model_selector,
            "model_weights_path": model_weights_path,
            # "tooltip": tooltip,
            "remove_button": remove_model_button,
        }

        # redraw the model widgets - this has its own trigger to
        # avoid race conditions
        self.redraw_models_trigger += 1

    def load_models_from_widgets(self) -> bool:  # pragma: no cover
        """Collect all the model metadata from widgets and instantiate the
        model wrapper classes"""

        module = importlib.import_module(f"jatic_ri.{self.task}.models")
        model_meta_class = module.ModelSpecification
        load_models = module.load_models

        # Load model(s)
        model_dict = {}
        for widget_label, widget_dict in self.model_widgets.items():
            model_number = re.search(r"\d+", widget_label).group()
            # require a path input
            if widget_dict["model_weights_path"].value:
                if widget_dict["model_selector"].value not in self.model_label_map:
                    self.status_text = f"Skipping model {model_number}, invalid type"
                model_name = (
                    f"{widget_dict['model_selector'].value}-{Path(widget_dict['model_weights_path'].value).stem}"
                )
                model_meta: model_meta_class = {
                    "model_type": self.model_label_map[
                        widget_dict["model_selector"].value
                    ],  # map from the visual name of the model to a model key we can use to reference the class
                    "model_weights_path": widget_dict["model_weights_path"].value,
                }
                model_dict[model_name] = model_meta
            else:
                self.status_text = f"Skipping model {model_number}, invalid path"

        self.loaded_models = load_models(model_dict)
        return True

    @param.depends("redraw_models_trigger")
    def view_model_widget_pairs(self) -> pn.Column:
        """Draw the model widgets except for the first one which is a special
        case.
        Whenever the "add model" button is called, first the widgets are
        built in `add_model_button_callback` and then this method is triggered
        to redraw the full dictionary of models
        """
        view = pn.Column()
        for value in self.model_widgets.values():
            # view everything except the first model widget which is a special case
            if value["model_selector"].name != "Model 1 type":
                view.append(
                    pn.Row(
                        value["model_selector"],
                        value["remove_button"],
                    )
                )
                view.append(
                    pn.Row(
                        pn.Spacer(width=self.width_subwidget_offset),
                        value["model_weights_path"],
                        # value["tooltip"],
                    )
                )

        return view

    def _view_dataset_1_selectors(self) -> pn.Column:
        """View of the dataset 1 widgets"""
        return pn.Column(
            self.dataset_1_selector,
            pn.Row(
                pn.Spacer(width=self.width_subwidget_offset),
                pn.Column(
                    self.dataset_1_split_path,
                    self.dataset_1_metadata_path,
                ),
            ),
            sizing_mode="stretch_width",
        )

    @param.depends("dataset_2_visible")
    def _view_dataset_2_selectors(self) -> pn.Column:
        """View of the dataset 2 selectors"""
        if self.dataset_2_visible:
            return pn.Column(
                self.dataset_2_selector,
                pn.Row(
                    pn.Spacer(width=self.width_subwidget_offset),
                    pn.Column(
                        self.dataset_2_split_path,
                        self.dataset_2_metadata_path,
                    ),
                ),
                sizing_mode="stretch_width",
            )
        return pn.Column()

    def load_datasets_from_widgets(self) -> bool:
        """Collect dataset metadata from widgets and instantiate
        dataset wrapper objects"""
        # Load dataset(s)
        if self.dataset_1_selector.value not in self.dataset_label_map:  # pragma: no cover
            self.status_text = "Please select dataset type"
            return False

        module = importlib.import_module(f"jatic_ri.{self.task}.datasets")
        load_datasets = module.load_datasets

        # gather dataset information from the widgets
        dataset_1_meta: DatasetSpecification = {  # noqa: F821
            "dataset_type": self.dataset_label_map[self.dataset_1_selector.value],
            "data_dir": self.dataset_1_split_path.value,
            "metadata_path": self.dataset_1_metadata_path.value,
        }

        dataset_meta = {"dataset_1": dataset_1_meta}

        # load dataset 2 only if the widget is visualized
        if self.dataset_2_visible:  # pragma: no cover
            if self.dataset_2_selector.value not in self.dataset_label_map:
                self.status_text = "Please select dataset 2 type"
                return False

            dataset_2_meta: DatasetSpecification = {  # noqa: F821
                "dataset_type": self.dataset_label_map[self.dataset_2_selector.value],
                "data_dir": self.dataset_2_split_path.value,
                "metadata_path": self.dataset_2_metadata_path.value,
            }
            dataset_meta["dataset_2"] = dataset_2_meta

        # load the datasets
        self.loaded_datasets = load_datasets(dataset_meta)

        return True

    def load_metric_from_widget(self) -> bool:
        """Collect input from widget and instantiate metric object, set the result
        into `self.loaded_metric`.

        WARNING: This needs information about the dataset for IC usecases (number
        of classes). Therefore, it must be run after the datasets are loaded.
        """
        metrics_module = importlib.import_module(f"jatic_ri.{self.task}.metrics")
        metric_class_name = self.metric_label_map[self.metric_selector.value]
        if self.task == "object_detection":
            kwargs = {}
        elif self.task == "image_classification":
            # NOTE: we have to make the assumption here that the same metric can be applied to both datasets.
            # This might come back to bite us.
            kwargs = {"num_classes": len(self.loaded_datasets["dataset_1"].metadata["index2label"])}
        else:
            raise RuntimeError(f"Task type {self.task} not supported.")

        metric_function = getattr(metrics_module, metric_class_name)

        self.loaded_metric = metric_function(**kwargs)

        return True

    def load_pipeline(self, configs: dict) -> bool:
        """Instantiate test stage objects based on configurations

        Example structure:
        {
            'task': 'object_detection',
            'survivor': {
                'TYPE': 'SurvivorTestStage',
                'CONFIG': {'metric_column': 'metric',
                    'conversion_type': 'original',
                    'otb_threshold': 0.5,
                    'easy_hard_threshold': 0.5
                    }
                }
            }

        Returns
        -------
        True if successful, False if there were issues.
        """
        logger.debug("load pipeline")
        if "task" not in configs:
            self.status_text = "Task must be specified in the provided config."
            logger.debug("Task must be specified in the provided config.")
            return False
        if configs["task"] != self.task:
            self.status_text = f"Mismatch between dashboard type, {self.task}, and provided config"
            return False
        for stage_label, config in configs.items():
            if stage_label != "task":
                self.status_text = f'Loading {config["TYPE"]}'
                logger.debug(f'Loading {config["TYPE"]}')
                if self.task == "object_detection":
                    stage = rehydrate_test_stage_od(config)
                elif self.task == "image_classification":
                    stage = rehydrate_test_stage_ic(config)
                self.test_stages[stage_label] = stage
                # allow multiple models for multi-model test stages
                if config["TYPE"] == "RealLabelTestStage" or config["TYPE"] == "SurvivorTestStage":
                    # trigger addition of the "add model" button
                    self.multi_model_visible = True

        self.status_text = "Configuration file loaded"

        return True

    def load_stage_inputs(self, test_stage: TestStage) -> None:  # pragma: no cover # noqa: C901
        """Loads the inputs to a given test stage based on
        values set in the UI and in the class itself
        """
        self.status_text = f"Loading inputs for {test_stage.__class__.__name__}"
        if isinstance(test_stage, TwoDatasetPlugin):
            test_stage.load_datasets(
                self.loaded_datasets["dataset_1"],
                self.loaded_datasets["dataset_1"].metadata["id"],
                self.loaded_datasets["dataset_2"],
                self.loaded_datasets["dataset_2"].metadata["id"],
            )
        elif isinstance(test_stage, SingleDatasetPlugin):
            if self.dataset_2_visible:
                self.status_text = (
                    f"Dataset {self.dataset_2_selector.value} is unused for {test_stage.__class__.__name__}"
                )
            test_stage.load_dataset(self.loaded_datasets["dataset_1"], self.loaded_datasets["dataset_1"].metadata["id"])

        if isinstance(test_stage, MetricPlugin):
            test_stage.load_metric(self.loaded_metric, self.loaded_metric.return_key)

        if isinstance(test_stage, ThresholdPlugin):
            test_stage.load_threshold(float(self.threshold))

        if isinstance(test_stage, SingleModelPlugin):
            if len(self.loaded_models) == 0:
                raise RuntimeError("No model loaded. Please select model.")
            if len(self.loaded_models) != 1:
                self.status_text = (
                    f"Model(s) {list(self.loaded_models.keys())[1:]} unused for {test_stage.__class__.__name__}"
                )
            model_1 = self.loaded_models[list(self.loaded_models.keys())[0]]
            test_stage.load_model(model_1, model_id=model_1.metadata["id"])

        elif isinstance(test_stage, MultiModelPlugin):
            test_stage.load_models(self.loaded_models)

        if isinstance(test_stage, EvalToolPlugin):
            test_stage.load_eval_tool(self.eval_tool)

    def _construct_report_filename(self) -> str:
        """Construct a report filename of the output report based on the UI selections"""
        model_name = "-".join(list(self.loaded_models.keys())).replace(" ", "_")
        dataset_name = (
            "-".join(list(self.loaded_datasets.keys())).replace(" ", "_")
            if self.dataset_2_visible
            else self.dataset_1_selector.value.replace(" ", "_")
        )
        return f"{model_name}_{dataset_name}_{self.metric_selector.value.replace(' ','_')}_{self.threshold}_report"

    def _run_all_tests(self) -> str:  # pragma: no cover
        """Run all the tests on all the stages and return a link to the resulting report.
        Common across IC/OD usecases.
        Should be triggered in `_run_button_callback` implementation
        """
        self.status_text = "Processing. Please wait..."

        self.eval_tool: EvaluationTool
        if not self.use_caches:
            # By default EvaluationTool is just a container for the functions without cache
            self.eval_tool = EvaluationTool()
        elif self.task == "image_classification" or self.task == "object_detection":
            self.eval_tool = EvaluationTool()
        else:
            raise RuntimeError(f"Task type {self.task} not supported.")

        slides = []
        for stage in self.test_stages.values():
            self.load_stage_inputs(stage)
            # run the stage, saving output to the class
            stage.run(use_stage_cache=self.use_caches)
            # collect the slides
            stage_slides = stage.collect_report_consumables()
            slides += stage_slides

        report_path = Path(self.output_dir)
        report_title = self._construct_report_filename()
        report = create_deck(slides, report_path, deck_name=report_title)

        self.status_text = f"Report saved to {report}"

        return create_download_link(
            str(report),
            label="Download Report",
            download_filename=f"{report_title}.pptx",
        )

    def horizontal_line(self) -> pn.pane.HTML:
        """Creates a horizontal line in an HTML element.
        This is a function because the same panel object
        cannot be visualized twice so in order to reuse
        this code snippet, we need to create a new object
        every time we need a line.
        """
        return pn.pane.HTML(
            """""",
            styles={
                "display": "block",
                "height": "1px",
                "border": "0",
                "border-top": f"1px solid {self.color_gray_500}",
                "margin": "0em 0",
                "padding": "0",
            },
            width=self.app_width - 48,
        )

    def view_status_bar(self) -> pn.Column:
        """View of status bar. Change the text on the status bar by modifying
        the `self.status_text` variable.
        DO NOT OVERWRITE THIS METHOD
        """
        return pn.Row(
            pn.Spacer(width=12),
            pn.Column(
                pn.Spacer(height=4),
                pn.pane.Markdown(
                    "Status",
                    styles={**self.style_text_body1, "color": self.color_blue_900},
                    stylesheets=[self.css_paragraph],
                ),
                pn.pane.Markdown(
                    self.status_text,
                    sizing_mode="stretch_width",
                    styles={**self.style_text_body1, "color": self.color_blue_800},
                    stylesheets=[self.css_paragraph],
                ),
                pn.Spacer(height=4),
                styles={
                    "background": self.color_blue_100,
                    "border-color": self.color_blue_300,
                    "border-width": "thin",
                    "border-style": "solid",
                    "border-radius": "3px",
                },
                sizing_mode="stretch_width",
            ),
            pn.Spacer(width=25),
        )

    def view_header(self) -> pn.Row:
        """View header row with JATIC logo"""
        return pn.Row(
            pn.pane.SVG(JATIC_LOGO_PATH, width=150),
            styles={"background": self.color_blue_900},
            width=self.app_width,
        )

    def view_config_input(self) -> pn.Row:
        """View of the configuration file loader"""
        logger.debug("view config input container")
        return pn.Column(
            pn.Spacer(height=20),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown(
                        "1. Upload configuration file",
                        styles=self.style_text_h3,
                        stylesheets=[self.css_paragraph],
                    ),
                    pn.Row(
                        pn.Spacer(width=12),  # padding to align this with title text above
                        pn.pane.Markdown(
                            "Upload for JSON configuration file (*.json)",
                            styles=self.style_text_body2,
                            stylesheets=[self.css_paragraph],
                            width=395,
                        ),
                    ),
                    # pn.Spacer(height=40),
                ),
                pn.Spacer(width=124),
                # white box on the right:
                pn.Column(
                    pn.Spacer(height=18),
                    pn.Row(
                        pn.Spacer(width=12),
                        self.config_file,
                        self.view_config_btn,
                        self.config_floatpanel_container,
                    ),
                    pn.Spacer(height=18),
                    styles={
                        "background": self.color_white,
                        "border-color": self.color_blue_300,
                        "border-width": "thin",
                        "border-style": "solid",
                        "border-radius": "5px",
                    },
                    sizing_mode="stretch_width",
                ),
                pn.Spacer(width=23),
            ),
            pn.Spacer(height=20),  # this controls the padding at the bottom of this section
        )

    def view_advanced(self) -> pn.Row:
        """View of the advanced row that contains the use cache toggle and threshold entry"""
        logger.debug("view threshold metric")
        return pn.Row(
            pn.Spacer(width=18),  # padding to the left of the box
            pn.Column(
                pn.Spacer(height=20),  # padding at the top of the box
                # blue box:
                pn.Column(
                    pn.pane.Markdown(
                        "Advanced (optional)",
                        styles=self.style_text_h3,
                        stylesheets=[self.css_paragraph],
                    ),
                    pn.Row(
                        pn.Spacer(width=15),  # padding on the left inside of the blue box
                        pn.Column(
                            "Use Cache",
                            pn.widgets.Switch.from_param(
                                self.param.use_caches,
                                name="",
                                stylesheets=[self.css_switch],
                            ),
                            pn.Spacer(height=10),  # padding for the bottom of the blue box
                        ),
                        pn.Column(
                            pn.Spacer(height=10),  # padding to make threshold align with usecache
                            pn.widgets.TextInput.from_param(
                                self.param.threshold,
                                name="Target Threshold",
                                width=175,
                                stylesheets=[self.css_dropdown],
                            ),
                        ),
                        sizing_mode="stretch_width",
                        styles={
                            "background": self.color_blue_100,
                            "border-color": self.color_blue_300,
                            "border-width": "thin",
                            "border-style": "solid",
                            "border-radius": "3px",
                        },
                    ),
                ),
                pn.Spacer(height=40),  # padding at the bottom of the box
            ),
            pn.Spacer(width=25),  # padding to the right of the box
        )

    def _view_df_tabulator(self) -> pn.widgets.Tabulator:
        """View of the data analysis tabulator widget
        This is broken out into a separate method to avoid challenges in
        automatic updating of dataframe-based widgets
        """
        formatters = {"Gradient Report": HTMLTemplateFormatter(template="<code><%= value %></code>")}
        # column widths can contain extra columns so here we assign all possible columns that
        # any of the inherited apps may have. Total should be around 1210.
        column_widths = {
            "Gradient Report": 330,
            "Model(s)": 280,
            "Dataset": 280,
            "Metric": 160,
            "Threshold": 160,
        }
        return pn.widgets.Tabulator(
            self.results_df,
            show_index=False,
            stylesheets=[self.css_tabulator_table],
            disabled=True,
            formatters=formatters,
            widths=column_widths,
        )

    def view_results_row(self) -> pn.Row:
        """View of the results row. This contains the table of results."""
        return pn.Row(
            pn.Spacer(width=10),  # padding to the left of the section
            pn.Column(
                pn.Row(
                    pn.Spacer(width=10),  # padding to the left of Test Results title
                    pn.pane.Markdown(
                        "Test Results",
                        styles=self.style_text_h3,
                        stylesheets=[self.css_paragraph],
                    ),
                ),
                self._view_df_tabulator,
            ),
        )

    def view_run_row(self) -> pn.Row:
        """View of the run row that contains the `Run Analysis` button."""
        return pn.Row(
            pn.layout.HSpacer(),
            self.run_analysis_button,
            pn.Spacer(width=15),
            sizing_mode="stretch_width",
        )

    def view_title_row(self) -> pn.Column:
        """View of the title row which contains the task (OD/IC),
        objective (Model Evaluation/Dataset Analysis), and a
        brief description of the app"""
        return pn.Column(
            pn.Spacer(height=10),
            pn.pane.Markdown(
                self.task.replace("_", " ").title(),
                styles=self.style_text_body2,
                stylesheets=[self.css_paragraph],
            ),
            pn.pane.Markdown(
                self.title,
                styles=self.style_text_h2,
                stylesheets=[self.css_paragraph],
            ),
            pn.pane.Markdown(
                "Configure your test setup to begin the analysis. You can view your results below",
                styles=self.style_text_body2,
                stylesheets=[self.css_paragraph],
            ),
        )

    def panel(self) -> pn.Column:
        """View of the entire dashboard"""
        return pn.Column(
            self.view_header,  # full width
            # rest of the app has some padding on left and right
            pn.Row(
                pn.Spacer(width=20),
                pn.Column(
                    self.view_title_row,
                    pn.Spacer(height=10),
                    self.view_status_bar,
                    pn.Spacer(height=12),
                    pn.pane.Markdown(
                        "Test Setup",
                        styles=self.style_text_h2,
                        stylesheets=[self.css_paragraph],
                    ),
                    pn.Spacer(height=10),
                    self.view_config_input,
                    self.horizontal_line(),
                    self.view_test_subject_row,
                    self.horizontal_line(),
                    self.view_input_artifacts_row,
                    self.horizontal_line(),
                    self.view_advanced,
                    self.view_run_row,
                    self.view_results_row,
                    pn.layout.Spacer(height=50),
                ),
            ),
            styles={"background": self.color_main_bg},
            width=self.app_width,
        )
