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

from jatic_ri._common.test_stages.interfaces.plugins import (
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
from jatic_ri.util.dashboard_utils import create_download_link, rehydrate_test_stage_ic, rehydrate_test_stage_od

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
    threshold = param.Number(default=0.5)

    # Load results from cache/save to cache
    use_cache = param.Boolean(default=True)

    # Input for loading the pipeline config
    config_file = pn.widgets.FileInput(accept=".json")

    # Table for storing the results
    results_df = param.DataFrame(pd.DataFrame({}))

    # Location for storing all output
    output_dir = param.Path(default=Path.cwd(), check_exists=False)

    def __init__(self, **params: dict[str, Any]) -> None:
        super().__init__(**params)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        self.color_dark_blue = "#233758"
        self.color_medium_blue = "#7AB8EF"  # varible currently unused
        self.color_light_blue = "#D6E6F9"
        self.color_light_gray = "#e8e4e3"  # light gray
        self.color_black = "#000000"
        self.table_style_headers = {
            "selector": "th:not(.index_name)",
            "props": "background-color: #7AB8EF; color: white; border-left: 1px solid #e3e3e3;",
        }
        self.table_style_borders = {"selector": "td", "props": "border: 1px solid #adadac"}  # light grey

        self.text_input_width = 200
        self.model_dropdown_width = 300
        self.dropdown_width = 140
        self.dropdown_height = "20px"
        self.dropdown_stylesheet = f"""
                :host {{
                  color: {self.color_light_gray};
                }}

                select:not([multiple]).bk-input, select:not([size]).bk-input {{
                  height: {self.dropdown_height};
                  color: {self.color_black}
                }}

                .bk-input {{
                  height: {self.dropdown_height};  /* widget height */
                  color: {self.color_black} /* text color */
                }}
                """
        # this reduces the vertical margins on the title
        self.title_stylesheet = """
                :host {
                  --line-height: 0px;
                }

                p {
                  padding: 0px;
                  margin: 10px;
                }
                """

        self.widget_stylesheet = f"""
                :host {{
                  color: {self.color_light_gray}; /* label text color */
                }}

                select:not([multiple]).bk-input, select:not([size]).bk-input {{
                  height: {self.dropdown_height}; /* dropdown widget height */
                  color: {self.color_black} /* text color on value of Dropdown widgets */
                }}

                .bk-input {{
                  height: {self.dropdown_height};  /* FloatInput widget height */
                  color: {self.color_black} /* text color on value of FloatInput widgets */
                }}
                """

        self.config_input = f"""
                .bk-input {{
                  color: {self.color_black} /* text color */
                }}

                input[type='file'] {{
                    height: 40px;  /* widget height */
                    border: 1px dashed;
                    padding: 0;  /* this is not being obeyed */
                }}
                """

        self.tabulator_style = """
                .tabulator {
                    background-color: #7AB8EF;
                    color: white;
                    border-left: 1px solid #e3e3e3;
                    }
                """

        self.download_button_style = f"""
                :host(.solid) .bk-btn.bk-btn-primary {{
                    color: {self.color_black};
                    background-color: {self.color_light_gray};
                    height: {self.dropdown_height}; /* widget height */
                }}

                .bk-btn a {{
                    text-decoration: none;
                    color: {self.color_black};
                    background-color: {self.color_light_gray};
                    height: {self.dropdown_height}; /* widget height */
                }}

                :host(.solid) .bk-btn.bk-btn-default a, :host(.outline) .bk-btn.bk-btn-default a {{
                    color: {self.color_black};
                    background-color: {self.color_light_gray};
                    height: {self.dropdown_height}; /* widget height */
                }}
                """
        # metric dropdown widget, options populated in _update_task_related_objects()
        self.metric_selector = pn.widgets.Select(
            name="Metric",
            width=self.dropdown_width,
            stylesheets=[self.dropdown_stylesheet],
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
        self.config_file.stylesheets = [self.config_input]

        # button to run the analysis
        self.run_analysis_button = pn.widgets.Button(name="Run Analysis", button_type="primary", disabled=False)

        self.dataset_1_selector = pn.widgets.Select(
            options=["Select Type", *list(self.dataset_label_map.keys())],
            width=self.dropdown_width,
            name="Dataset 1",
            stylesheets=[self.dropdown_stylesheet],
            value="Select Type",
        )
        self.dataset_2_selector = pn.widgets.Select(
            options=["Select Type", *list(self.dataset_label_map.keys())],
            width=self.dropdown_width,
            name="Dataset 2",
            stylesheets=[self.dropdown_stylesheet],
            value="Select Type",
        )
        self.dataset_1_split_path = pn.widgets.TextInput(
            name="Path of split folder",
            placeholder="Path of split folder",
            width=self.text_input_width,
            stylesheets=[self.dropdown_stylesheet],
        )
        self.dataset_2_split_path = pn.widgets.TextInput(
            name="Path of split folder",
            placeholder="Path of split folder",
            width=self.text_input_width,
            stylesheets=[self.dropdown_stylesheet],
        )
        self.dataset_1_metadata_path = pn.widgets.TextInput(
            name="Path to metadata file",
            placeholder="Path to metadata file",
            width=self.text_input_width,
            stylesheets=[self.dropdown_stylesheet],
        )
        self.dataset_2_metadata_path = pn.widgets.TextInput(
            name="Path to metadata file",
            placeholder="Path to metadata file",
            width=self.text_input_width,
            stylesheets=[self.dropdown_stylesheet],
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

        self.status_text = pn.widgets.StaticText(
            name="Status ",
            value="Upload config to begin",
            sizing_mode="stretch_width",
            styles={"color": self.color_light_gray},
        )

    @param.depends("task", watch=True)
    def _update_task_related_objects(self, event=None) -> None:  # noqa: ANN001, ARG002
        if self.task == "object_detection":
            self.dataset_label_map = DATASET_LABEL_MAP_OD
            self.metric_selector.options = list(METRICS_LABEL_MAP_OD.keys())
            self.metric_selector.value = list(METRICS_LABEL_MAP_OD.keys())[0]
            self.metric_label_map = METRICS_LABEL_MAP_OD
            self.model_label_map = {value.replace("_", " "): key for key, value in SUPPORTED_MODELS_OD.items()}
            self.torchvision_models = SUPPORTED_TORCHVISION_MODELS_OD

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
            theme=f"{self.color_light_blue} filledLight",
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
            self.model_widgets[event.obj.name][
                "tooltip"
            ].value = "Supported extensions are .pt and .pth. A config.json should exist in the same directory )"
        else:
            self.model_widgets[event.obj.name]["model_weights_path"].placeholder = "Select file"
            self.model_widgets[event.obj.name]["tooltip"].value = "Select file"

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
            name="Path to model file",
            placeholder="Select weights file",
            width=self.model_dropdown_width,
            stylesheets=[self.dropdown_stylesheet],
        )
        selector_label = f"Select Model {len(self.model_widgets) + 1} type"
        model_selector = pn.widgets.Select(
            name=selector_label,
            options=["Select Model type", *list(self.model_label_map.keys())],
            width=self.model_dropdown_width,
            stylesheets=[self.dropdown_stylesheet],
            value="Select Model type",
        )
        tooltip = pn.widgets.TooltipIcon(
            value="Path to model weights file (config.json should exist in the same directory)"
        )
        # link a callback method to the model dropdown so that we
        # can change the placeholder text when the model type is changed
        model_selector.param.watch(self._on_model_type_change, ["value"], onlychanged=False)

        if self.multi_model_visible:
            # button for removing model from widget list
            remove_model_button = pn.widgets.ButtonIcon(
                icon="circle-x", size="2em", name="Remove model", description=selector_label
            )

            remove_model_button.param.watch(self._remove_model_widget, ["clicks"])
        else:
            remove_model_button = None

        # store the set of widgets for this model in a dict for reference later
        self.model_widgets[model_selector.name] = {
            "model_selector": model_selector,
            "model_weights_path": model_weights_path,
            "tooltip": tooltip,
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
                    self.status_text.value = f"Please select model {model_number} type"
                    return False
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
                self.status_text.value = f"Please select model {model_number} path"
                return False

        self.loaded_models = load_models(model_dict)
        return True

    @param.depends("redraw_models_trigger")
    def view_model_selections(self) -> pn.Column:
        """Draw the model widgets.
        Whenever the "add model" button is called, first the widgets are
        built in `add_model_button_callback` and then this method is triggered
        to redraw the full dictionary of models
        """
        view = pn.Column()
        for value in self.model_widgets.values():
            view.append(
                pn.Row(
                    value["model_selector"],
                    value["remove_button"],
                )
            )
            view.append(
                pn.Row(
                    value["model_weights_path"],
                    value["tooltip"],
                )
            )

        return view

    def input_model_pane(self) -> pn.Row:
        """View of the model input widgets"""
        view = pn.Row(
            self.view_model_selections,
            sizing_mode="stretch_width",
        )
        # only add the add_model option for multi-model cases
        if self.multi_model_visible:
            view.append(self.add_model)

        return view

    def view_dataset_1_selectors(self) -> pn.Column:
        """View of the dataset 1 widgets"""
        return pn.Column(
            self.dataset_1_selector,
            self.dataset_1_split_path,
            self.dataset_1_metadata_path,
            sizing_mode="stretch_width",
        )

    @param.depends("dataset_2_visible")
    def view_dataset_2_selectors(self) -> pn.Column:
        """View of the dataset 2 selectors"""
        if self.dataset_2_visible:
            return pn.Column(
                self.dataset_2_selector,
                self.dataset_2_split_path,
                self.dataset_2_metadata_path,
                sizing_mode="stretch_width",
            )
        return pn.Column()

    def load_datasets_from_widgets(self) -> bool:
        """Collect dataset metadata from widgets and instantiate
        dataset wrapper objects"""
        # Load dataset(s)
        if self.dataset_1_selector.value not in self.dataset_label_map:  # pragma: no cover
            self.status_text.value = "Please select dataset type"
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
                self.status_text.value = "Please select dataset 2 type"
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

    def load_pipeline(self, configs: dict) -> None:
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
            self.status_text.value = "Task must be specified in the provided config."
            logger.debug("Task must be specified in the provided config.")
            return False
        if configs["task"] != self.task:
            self.status_text.value = f"Mismatch between dashboard type, {self.task}, and provided config"
            return False
        for stage_label, config in configs.items():
            if stage_label != "task":
                self.status_text.value = f'Loading {config["TYPE"]}'
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

        self.status_text.value = "Configuration file loaded"

        return True

    def load_stage_inputs(self, test_stage: TestStage) -> None:  # pragma: no cover
        """Loads the inputs to a given test stage based on
        values set in the UI and in the class itself
        """
        self.status_text.value = f"Loading inputs for {test_stage.__class__.__name__}"
        if isinstance(test_stage, TwoDatasetPlugin):
            test_stage.load_datasets(
                self.loaded_datasets["dataset_1"], "dataset1", self.loaded_datasets["dataset_2"], "dataset2"
            )
        elif isinstance(test_stage, SingleDatasetPlugin):
            if self.dataset_2_visible:
                self.status_text.value = (
                    f"Dataset {self.dataset_2_selector.value} is unused for {test_stage.__class__.__name__}"
                )
            test_stage.load_dataset(self.loaded_datasets["dataset_1"], "dataset1")

        if isinstance(test_stage, MetricPlugin):
            test_stage.load_metric(self.loaded_metric, self.loaded_metric.return_key)

        if isinstance(test_stage, ThresholdPlugin):
            test_stage.load_threshold(self.threshold)

        if isinstance(test_stage, SingleModelPlugin):
            if len(self.loaded_models) == 0:
                raise RuntimeError("No model loaded. Please select model.")
            if len(self.loaded_models) == 1:
                test_stage.load_model(self.loaded_models[list(self.loaded_models.keys())[0]], model_id="model_1")
            else:
                list(self.loaded_models.keys())[1:]
                self.status_text.value = (
                    f"Model(s) {list(self.loaded_models.keys())[1:]} unused for {test_stage.__class__.__name__}"
                )
        elif isinstance(test_stage, MultiModelPlugin):
            test_stage.load_models(self.loaded_models)

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
        self.status_text.value = "Processing. Please wait..."

        slides = []
        for stage in self.test_stages.values():
            self.load_stage_inputs(stage)
            # run the stage, saving output to the class
            stage.run(use_cache=self.use_cache)
            # collect the slides
            stage_slides = stage.collect_report_consumables()
            slides += stage_slides

        report_path = Path(self.output_dir)
        report_title = self._construct_report_filename()
        report = create_deck(slides, report_path, deck_name=report_title)

        self.status_text.value = f"Report saved to {report}"

        return create_download_link(
            str(report),
            label="Download Report",
            download_filename=f"{report_title}.pptx",
        )

    def view_title(self) -> pn.pane.Markdown:
        """View of the app title
        DO NOT OVERWRITE THIS METHOD
        """
        logger.debug("view title")
        return pn.pane.Markdown(
            f"{self.title}: {self.task.replace('_',' ').title()}",
            styles={"font-size": f"{self.title_font_size}px", "color": f"{self.color_light_gray}"},
            stylesheets=[self.title_stylesheet],
        )

    def view_status_bar(self) -> pn.Column:
        """View of status bar"""
        logger.debug("view status bar")
        return pn.Column(
            pn.layout.Divider(),
            self.status_text,
        )

    def view_config_input(self) -> pn.Row:
        """View of the configuration file loader"""
        logger.debug("view config input container")
        return pn.Row(
            pn.layout.Spacer(width=10),
            pn.Card(
                pn.Row(
                    self.config_file,
                    self.view_config_btn,
                    self.config_floatpanel_container,
                ),
                title="Configuration Setup",
                styles={"background": f"{self.color_light_gray}"},
            ),
        )

    def view_threshold_metric(self) -> pn.Row:
        """View of the threshold and metric input widgets"""
        logger.debug("view threshold metric")
        return pn.Row(
            pn.widgets.FloatInput.from_param(
                self.param.threshold,
                step=0.01,
                start=0,
                end=1,
                format=".00",
                name="Target Threshold",
                width=self.dropdown_width,
                stylesheets=[self.dropdown_stylesheet],
            ),
            self.metric_selector,
            pn.Column(
                "Use Cache",
                pn.widgets.Switch.from_param(self.param.use_cache, name=""),
            ),
            sizing_mode="stretch_width",
        )

    def view_df_tabulator(self) -> pn.widgets.Tabulator:
        """View of the data analysis tabulator widget
        This is broken out into a separate method to avoid challenges in
        automatic updating of dataframe-based widgets
        """
        formatters = {"Gradient Report": HTMLTemplateFormatter(template="<code><%= value %></code>")}
        return pn.widgets.Tabulator(
            self.results_df, show_index=False, stylesheets=[self.tabulator_style], disabled=True, formatters=formatters
        )
