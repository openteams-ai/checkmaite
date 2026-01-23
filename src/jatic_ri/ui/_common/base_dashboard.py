"""Base dashboard for JATIC RI testbeds.

This module provides the `BaseTestbed` class, which serves as a foundational
Panel application for model evaluation and dataset analysis workflows. It
includes common UI elements, styling, and logic for configuring and running
test stages.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import panel as pn
import param
from bokeh.models import HTMLTemplateFormatter
from streamz import Stream

from jatic_ri.core.capability_core import Capability, CapabilityConfigBase, Number
from jatic_ri.core.image_classification.models import SUPPORTED_MODELS as SUPPORTED_MODELS_IC
from jatic_ri.core.image_classification.models import SUPPORTED_TORCHVISION_MODELS as SUPPORTED_TORCHVISION_MODELS_IC
from jatic_ri.core.object_detection.models import SUPPORTED_MODELS as SUPPORTED_MODELS_OD
from jatic_ri.core.object_detection.models import SUPPORTED_TORCHVISION_MODELS as SUPPORTED_TORCHVISION_MODELS_OD
from jatic_ri.core.object_detection.models import SUPPORTED_VISDRONE_MODELS as SUPPORTED_VISDRONE_MODELS_OD
from jatic_ri.core.report import _gradient as gd
from jatic_ri.ui._common.base_app import AppStyling, BaseApp
from jatic_ri.ui.dashboard_utils import (
    create_download_link,
    get_capability_from_app_config_ic,
    get_capability_from_app_config_od,
    with_loading,
)

if TYPE_CHECKING:
    from panel.widgets import Widget

JATIC_LOGO_PATH = Path(__file__).parents[2] / "assets" / "JATIC_Logo_Acronym_Spelled_Out_RGB_white_type.svg"

pn.extension(
    "tabulator",
    "floatpanel",
    # Spiner settings for loading animation
    loading_spinner="dots",
    loading_color="#001B4D",
)

logger = logging.getLogger()


# mapping between the visible dataset labels in the UI and
# the underlying wrapper class
DATASET_LABEL_MAP_OD = {
    "COCO dataset": "CocoDetectionDataset",
    "YOLO dataset": "YoloDetectionDataset",
    "Visdrone dataset": "VisdroneDetectionDataset",
}

DATASET_LABEL_MAP_IC = {
    "YOLO dataset": "YoloClassificationDataset",
}

METRICS_LABEL_MAP_OD = {
    "mAP": "map50_torch_metric_factory",
    "mAP (per class)": "multiclass_map50_torch_metric_factory",
}

METRICS_LABEL_MAP_IC = {
    "Accuracy": "accuracy_multiclass_torch_metric_factory",
    "F1 Score": "f1score_multiclass_torch_metric_factory",
}

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


class BaseTestbed(BaseApp):
    """Base class for testbed applications.

    This class is inherited by dataset analysis and model evaluation
    testbed classes. It provides common CSS styling, widgets, and
    functions for building interactive testing pipelines.

    Parameters
    ----------
    styles : AppStyling
        Styling configuration object.
    **params : dict[str, Any]
        Additional parameters for the `param.Parameterized` base class.

    Attributes
    ----------
    title : param.String
        The title of the testbed application.
    test_stages : param.Dict
        Dictionary holding configured test stage instances.
    redraw_models_trigger : param.Integer
        Trigger to redraw model configuration widgets.
    loaded_models : param.Dict
        Dictionary of loaded model instances.
    multi_model_visible : param.Boolean
        Controls visibility of multi-model selection UI.
    loaded_datasets : param.Dict
        Dictionary of loaded dataset instances.
    dataset_2_visible : param.Boolean
        Controls visibility of UI elements for a second dataset.
    threshold : param.String
        Input for target threshold if a test stage requires it.
    threshold_visible : param.Boolean
        Controls visibility of the threshold input widget.
    use_caches : param.Boolean
        Flag to enable/disable caching for test stage runs.
    results_df : param.DataFrame
        Pandas DataFrame to store and display analysis results.
    output_dir : param.Path
        Directory for storing all output files, including reports.
    metric_selector : pn.widgets.Select
        Widget for selecting the evaluation metric.
    model_widgets : dict
        Stores Panel widgets related to model configuration.
    add_model : pn.widgets.Button
        Button to add UI elements for configuring an additional model.
    run_analysis_button : pn.widgets.Button
        Button to trigger the execution of the configured analysis.
    dataset_1_selector : pn.widgets.Select
        Widget to select the type for the primary dataset.
    dataset_2_selector : pn.widgets.Select
        Widget to select the type for the comparison dataset.
    dataset_1_directory : pn.widgets.TextInput
        Input widget for the primary dataset's image/data directory.
    dataset_2_directory : pn.widgets.TextInput
        Input widget for the comparison dataset's image/data directory.
    dataset_1_metadata_path : pn.widgets.TextInput
        Input widget for the primary dataset's metadata file path.
    dataset_2_metadata_path : pn.widgets.TextInput
        Input widget for the comparison dataset's metadata file path.
    status_source : streamz.Stream
        Stream for emitting status updates to the UI.
    status_pane : pn.pane.Streamz
        Panel pane to display status messages.
    dataset_label_map : dict
        Mapping from UI dataset labels to internal dataset class names.
    metric_label_map : dict
        Mapping from UI metric labels to internal metric factory names.
    model_label_map : dict
        Mapping from UI model labels to internal model keys.
    torchvision_models : dict
        Supported TorchVision models for the current task.
    visdrone_models : dict, optional
        Supported VisDrone models (specific to Object Detection).
    loaded_metric : Any
        The instantiated metric object.

    """

    title = param.String(default="Object Detection Testing Pipeline")

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
    # boolean for viewing threshold widget
    threshold_visible = param.Boolean(default=False)

    # This parameter configures the dashboard to use caching.  Caching can occur both on the
    # test stage output and on the underlying model inference.  The full flow looks like this:
    #   1) TestStage.run() will first check for a previously cached output for the given arguments.
    #   Each capability's test stage may implement this differently. If the stage cache hits, flow ends here.
    #   2) If no cache hit, the TestStage's _run implementation executes.
    #   3) When 'use_caches' is True, there will be a global cache for model predictions and evaluations:
    #       Predictions - i.e. model + dataset + batch size.  Both `predict()` and `evaluate_from_predictions()`
    #       will check and use if available.  Since `evaluate_from_predictions()` runs `predict()`, the same test
    #       stage can run multiple metrics against a given model and dataset (and batch size) without having to
    #       redo the inference.
    #       Evaluations - i.e. model + dataset + metric + batch size.  Both `evaluate_from_predictions()` and
    #       `compute_metric()` will check for a cache.
    #   5) Only after a cache miss for `predict()` or `evaluate_from_predictions()` will the actual execution
    #      (model inference and/or metric calculation) occur, with the results both cached globally and then
    #      passed back to the TestStage._run(), which will in turn cache the output results for subsequent calls.
    use_caches = param.Boolean(default=True)

    # Table for storing the results
    results_df = param.DataFrame(pd.DataFrame({}))

    # Location for storing all output
    output_dir = param.Path(default=Path.cwd().joinpath("report"), check_exists=False)

    def __init__(self, styles: AppStyling, **params: dict[str, Any]) -> None:
        super().__init__(styles, **params)
        # ensure we don't visualize multiple models (no add model button).
        # Must be set before the add_model_button_callback and process_testbed_config
        self.multi_model_visible = False
        self._process_testbed_config()
        self.styles = styles
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # metric dropdown widget, options populated in _update_task_related_objects()
        self.metric_selector = pn.widgets.Select(
            name="Evaluation Metric",
            width=self.styles.width_input_default,
            stylesheets=[self.styles.css_dropdown],
        )
        self._update_task_related_objects()
        # holder for the model widgets (for dynamic handling of the number of models)
        self.model_widgets = {}
        # button for adding another model (adds two widgets per model)
        self.add_model = pn.widgets.Button(name="Add model")
        self.add_model.on_click(self.add_model_button_callback)
        self.add_model_button_callback(None)  # trigger on init to build the first set

        # button to run the analysis
        self.run_analysis_button = pn.widgets.Button(name="Run Analysis", button_type="primary", disabled=False)
        self.run_analysis_button.on_click(self._run_analysis_loading)

        self.dataset_1_selector = pn.widgets.Select(
            options=["Define Dataset Type", *list(self.dataset_label_map.keys())],
            width=self.styles.width_input_default,
            name="Dataset type",
            stylesheets=[self.styles.css_dropdown],
            value="Define Dataset Type",
        )
        self.dataset_2_selector = pn.widgets.Select(
            options=["Define Dataset Type", *list(self.dataset_label_map.keys())],
            width=self.styles.width_input_default,
            name="Comparison Dataset type",
            stylesheets=[self.styles.css_dropdown],
            value="Define Dataset Type",
        )
        # link a callback method to the dataset dropdown so that we
        # can change the placeholder text when the dataset type is changed
        self.dataset_1_selector.param.watch(self._on_dataset_type_change, ["value"])
        self.dataset_2_selector.param.watch(self._on_dataset_type_change, ["value"])
        self.dataset_1_directory = pn.widgets.TextInput(
            name="Images directory",
            placeholder="Full path to images directory.",
            description="Full filepath to the directory containing dataset images.",
            width=self.styles.width_input_default - self.styles.width_subwidget_offset,
            stylesheets=[self.styles.css_dropdown],
            disabled=True,
        )
        self.dataset_2_directory = pn.widgets.TextInput(
            name="Images directory",
            placeholder="Full path to images directory.",
            description="Full filepath to the directory containing dataset images.",
            width=self.styles.width_input_default - self.styles.width_subwidget_offset,
            stylesheets=[self.styles.css_dropdown],
            disabled=True,
        )
        self.dataset_1_metadata_path = pn.widgets.TextInput(
            name="Metadata filepath",
            placeholder="Full filepath to dataset metadata file.",
            description="Full filepath to dataset metadata file.",
            width=self.styles.width_input_default - self.styles.width_subwidget_offset,
            stylesheets=[self.styles.css_dropdown],
            disabled=True,
        )
        self.dataset_2_metadata_path = pn.widgets.TextInput(
            name="Metadata filepath",
            placeholder="Full filepath to dataset metadata file.",
            description="Full filepath to dataset metadata file.",
            width=self.styles.width_input_default - self.styles.width_subwidget_offset,
            stylesheets=[self.styles.css_dropdown],
            disabled=True,
        )

        # controls for viewing configuration
        self.view_config_btn = pn.widgets.ButtonIcon(
            icon="eye",
            size="2em",
            description="View loaded config",
            disabled=False,
            styles={"color": "black"},
        )
        # self.view_config_btn.on_click(self._on_view_config_callback)
        # create invisible container to put floatpanel "in", cannot be completely empty
        self.config_floatpanel_container = pn.Column(pn.widgets.Checkbox(visible=False))

        # Create a Stream that will carry text updates
        self.status_source = Stream()
        # Create a Streamz pane that will display the status messages
        self.status_pane = pn.pane.Streamz(
            self.status_source,
            always_watch=True,
            sizing_mode="stretch_width",
            styles={**self.styles.style_text_body1, "color": self.styles.color_blue_800},
            stylesheets=[self.styles.css_paragraph],
        )
        # Emit an initial message
        self.status_source.emit("Waiting for input...")

    @abstractmethod
    def _run_button_callback(self, event: Any = None) -> None:
        """Callback executed when the main run button is clicked.

        This method is a placeholder and must be overridden by subclasses to
        implement the specific analysis logic.

        Parameters
        ----------
        event : Any, optional
            The event object, by default None.
        """
        pass

    @with_loading("run_analysis_button")
    def _run_analysis_loading(self, event: Any = None) -> None:
        """Wrapper for `_run_button_callback` to show a loading spinner.

        This function is connected to the `run_analysis_button` click event.
        It displays a loading spinner while `_run_button_callback` executes.

        Parameters
        ----------
        event : Any, optional
            The event object, by default None.
        """
        self._run_button_callback(event)

    @param.depends("task", watch=True)
    def _update_task_related_objects(self, event: Any = None) -> None:  # noqa: ARG002
        if self.task == "object_detection":
            self.dataset_label_map = DATASET_LABEL_MAP_OD
            self.metric_selector.options = list(METRICS_LABEL_MAP_OD.keys())
            self.metric_selector.value = list(METRICS_LABEL_MAP_OD.keys())[0]
            self.metric_label_map = METRICS_LABEL_MAP_OD
            self.model_label_map = {
                value.replace("_", " ").removesuffix(" Weights"): key for key, value in SUPPORTED_MODELS_OD.items()
            }
            self.torchvision_models = SUPPORTED_TORCHVISION_MODELS_OD
            self.visdrone_models = SUPPORTED_VISDRONE_MODELS_OD

        elif self.task == "image_classification":
            self.dataset_label_map = DATASET_LABEL_MAP_IC
            self.metric_selector.options = list(METRICS_LABEL_MAP_IC.keys())
            self.metric_selector.value = list(METRICS_LABEL_MAP_IC.keys())[0]
            self.metric_label_map = METRICS_LABEL_MAP_IC
            self.model_label_map = {
                value.replace("_", " ").removesuffix(" Weights"): key for key, value in SUPPORTED_MODELS_IC.items()
            }
            self.torchvision_models = SUPPORTED_TORCHVISION_MODELS_IC

    def _process_testbed_config(self) -> None:
        logger.debug("populated test stages via load_pipeline")
        success = self.load_pipeline(self.output_test_stages)

        if not success:
            self.status_source.emit("Configuration file failed to load properly.")

    def _on_model_type_change(self, event: Any) -> None:  # pragma: no cover
        """Handle changes in model type selection widgets.

        This callback updates the placeholder text and enabled state of
        associated model weights and config path widgets based on the
        selected model type.

        Parameters
        ----------
        event : Any
            The event object triggered by a model selector widget change.

        Notes
        -----
        Additional behavior changes related to `model_weights_path` are
        handled by `_on_model_weights_path_change`.
        """
        # get a list of model numbers as strings
        str_numbers = re.findall(r"\d+", event.obj.name)
        # convert string list to int list and get the max model number
        model_no = max(list(map(int, str_numbers)))

        # if a model type is selected:
        if self.model_widgets[event.obj.name]["model_selector"].value != "Select Model type":
            self.model_widgets[event.obj.name]["model_weights_path"].disabled = False
            # if a torchvision type
            if self.model_label_map[event.obj.value] in self.torchvision_models:
                self.model_widgets[event.obj.name]["model_weights_path"].name = "Path to weights file"
                self.model_widgets[event.obj.name][
                    "model_weights_path"
                ].placeholder = "Filepath to model weights file. Leave empty for torchvision default weights."
                self.model_widgets[event.obj.name][
                    "model_config_path"
                ].placeholder = "Path to JSON config with 'index2label' key defining the categories."
                self.model_widgets[event.obj.name][
                    "model_weights_path"
                ].description = f"Model {model_no} weights pickle file containing the model state dictionary."
                if self.model_widgets[event.obj.name]["model_weights_path"].value:
                    self.model_widgets[event.obj.name]["model_config_path"].disabled = False
            # if non-torchvision (aka visdrone for now)
            else:
                self.model_widgets[event.obj.name]["model_weights_path"].name = "Output directory"
                self.model_widgets[event.obj.name][
                    "model_weights_path"
                ].placeholder = "Output directory for Visdrone model weights."
                self.model_widgets[event.obj.name]["model_weights_path"].description = (
                    f"Set model {model_no} output directory in which the Visdrone model will be "
                    f"downloaded from Kitware data server."
                )
                self.model_widgets[event.obj.name][
                    "model_weights_path"
                ].value = ""  # triggers _on_model_weights_path_change
        else:
            self.model_widgets[event.obj.name]["model_weights_path"].disabled = True
            self.model_widgets[event.obj.name]["model_config_path"].disabled = True

    def _on_model_weights_path_change(self, event: Any) -> None:  # pragma: no cover
        # get a list of model numbers as strings
        str_numbers = re.findall(r"\d+", event.obj.description)
        # convert string list to int list and get the max model number
        model_no = max(list(map(int, str_numbers)))

        model_key = f"Model {model_no} type"
        # if weights path has something in it and this is a torchvision model
        if (
            self.model_widgets[model_key]["model_weights_path"].value
            and self.model_label_map[self.model_widgets[model_key]["model_selector"].value] in self.torchvision_models
        ):
            self.model_widgets[model_key]["model_config_path"].disabled = False
        else:
            # visdrone models never require config
            self.model_widgets[model_key]["model_config_path"].disabled = True

    def _remove_model_widget(self, event: Any) -> None:
        del self.model_widgets[event.obj.description]
        # redraw the model widgets
        self.redraw_models_trigger += 1

    def add_model_button_callback(self, event: Any) -> None:  # noqa: ARG002
        """Add UI widgets for configuring an additional model.

        This callback is triggered when the 'Add model' button is clicked.
        It dynamically creates and adds a new set of widgets (model selector,
        weights path, config path, remove button) to the UI.

        Parameters
        ----------
        event : Any
            The event object from the button click (can be None if called directly).
        """

        # get a list of model numbers as strings
        str_numbers = re.findall(r"\d+", " ".join(self.model_widgets.keys()))
        # convert string list to int list and get the max model number,
        # increment from the current highest model number to avoid name collisions
        new_model_no = max(list(map(int, str_numbers))) + 1 if str_numbers else 1
        selector_label = f"Model {new_model_no} type"
        model_selector = pn.widgets.Select(
            name=selector_label,
            options=["Select Model type", *list(self.model_label_map.keys())],
            width=self.styles.width_input_default,
            stylesheets=[self.styles.css_dropdown],
            value="Select Model type",
        )
        model_weights_path = pn.widgets.TextInput(
            name="Path to model weights",
            placeholder="Path to weights file",
            width=self.styles.width_input_default,
            stylesheets=[self.styles.css_dropdown],
            disabled=True,
            description=f"Model {new_model_no} weights pickle file containing the model state dictionary.",
        )
        model_config_path = pn.widgets.TextInput(
            name="Path to model config",
            placeholder="Path to config file",
            width=self.styles.width_input_default,
            stylesheets=[self.styles.css_dropdown],
            disabled=True,
            description=(
                "JSON-formatted configuration file with translation from index to category "
                "label stored in a dictionary under 'index2label' key. Required if providing custom weights."
            ),
        )
        # link a callback method to the model dropdown so that we
        # can change the placeholder text when the model type is changed
        model_selector.param.watch(self._on_model_type_change, ["value"])
        # link a callback method to change config path anytime weights path changes
        model_weights_path.param.watch(self._on_model_weights_path_change, ["value"])

        if self.multi_model_visible:
            # button for removing model from widget list
            remove_model_button = pn.widgets.ButtonIcon(
                icon="circle-x", size="2em", name="", description=selector_label
            )

            remove_model_button.param.watch(self._remove_model_widget, ["clicks"])
        else:
            remove_model_button = None

        # store the set of widgets for this model in a dict for reference later
        self.model_widgets[selector_label] = {
            "model_selector": model_selector,
            "model_weights_path": model_weights_path,
            "remove_button": remove_model_button,
            "model_config_path": model_config_path,
        }

        # redraw the model widgets - this has its own trigger to
        # avoid race conditions
        self.redraw_models_trigger += 1

    def load_models_from_widgets(self) -> bool:  # pragma: no cover
        """Load models based on current widget configurations.

        Collects model metadata from the UI widgets, instantiates the
        corresponding model wrapper classes, and stores them in
        `self.loaded_models`.

        Returns
        -------
        bool
            True if models were loaded successfully, False otherwise.
        """

        module = importlib.import_module(f"jatic_ri.core.{self.task}.models")
        model_meta_class = module.ModelSpecification
        load_models = module.load_models

        # Load model(s)
        model_dict = {}
        for widget_label, widget_dict in self.model_widgets.items():
            model_number = re.search(r"\d+", widget_label).group()
            # if no model type is selected, warn and skip
            if widget_dict["model_selector"].value not in self.model_label_map:
                # For as long as model dropdowns with 'Select Model type' appear in the UI in the case where no
                # test stages require a model, this is a valid path.
                self.status_source.emit(f"Skipping model {model_number}, invalid type")
                continue
            model_meta: model_meta_class = {"model_type": self.model_label_map[widget_dict["model_selector"].value]}
            # if torchvision
            if self.model_label_map[widget_dict["model_selector"].value] in self.torchvision_models:
                # if no weights path, use OOTB weights
                if not widget_dict["model_weights_path"].value:
                    # map from the visual name of the model to a model key we can use to reference the class
                    # constructed without weights path by default, but appending model number for uniqueness
                    model_name = f"{widget_dict['model_selector'].value}-{model_number}"
                    self.status_source.emit(f"Using {model_name} with default weights and configuration.")
                # weights path
                else:
                    model_meta["model_weights_path"] = widget_dict["model_weights_path"].value
                    # if no config path, warn and skip
                    if not widget_dict["model_config_path"].value:
                        self.status_source.emit(f"Skipping model {model_number}, config path must be set")
                        continue
                    model_meta["model_config_path"] = widget_dict["model_config_path"].value

                    model_name = (
                        f"{widget_dict['model_selector'].value}-{Path(widget_dict['model_weights_path'].value).stem}"
                        f"-{Path(widget_dict['model_config_path'].value).stem}"
                    )
            # if visdrone
            else:
                # if weights path (aka output dir)
                if widget_dict["model_weights_path"].value:
                    model_meta["model_weights_path"] = widget_dict["model_weights_path"].value
                # if no weights path (aka output dir), assume cwd
                else:
                    model_meta["model_weights_path"] = os.getcwd()
                model_name = f"{widget_dict['model_selector'].value}-{model_number}"

            model_dict[model_name] = model_meta
        try:
            self.loaded_models = load_models(model_dict)
            return True
        except Exception as e:  # noqa: BLE001
            self.status_source.emit(f"An error occurred during model loading: {e}")
            return False

    @param.depends("redraw_models_trigger")
    def view_model_widget_pairs(self) -> pn.Column:
        """Generate the view for dynamically added model configuration widgets.

        This method is triggered by `redraw_models_trigger` and rebuilds the
        Panel Column containing widget sets for all models beyond the first one.
        The first model's widgets are typically handled separately in the main layout.

        Returns
        -------
        pn.Column
            A Panel Column containing rows of widgets for each additional model.
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
                        pn.Spacer(width=self.styles.width_subwidget_offset),
                        value["model_weights_path"],
                    )
                )
                view.append(
                    pn.Row(
                        pn.Spacer(width=self.styles.width_subwidget_offset),
                        value["model_config_path"],
                    )
                )

        return view

    def _view_dataset_1_selectors(self) -> pn.Column:
        """Generate the view for primary dataset selection widgets.

        Returns
        -------
        pn.Column
            A Panel Column containing widgets for configuring the first dataset.
        """
        return pn.Column(
            self.dataset_1_selector,
            pn.Row(
                pn.Spacer(width=self.styles.width_subwidget_offset),
                pn.Column(
                    self.dataset_1_directory,
                    self.dataset_1_metadata_path,
                ),
            ),
            sizing_mode="stretch_width",
        )

    @param.depends("dataset_2_visible")
    def _view_dataset_2_selectors(self) -> pn.Column:
        """Generate the view for comparison (second) dataset selection widgets.

        This view is conditional on `self.dataset_2_visible`.

        Returns
        -------
        pn.Column
            A Panel Column containing widgets for configuring the second dataset,
            or an empty Column if not visible.
        """
        if self.dataset_2_visible:
            return pn.Column(
                self.dataset_2_selector,
                pn.Row(
                    pn.Spacer(width=self.styles.width_subwidget_offset),
                    pn.Column(
                        self.dataset_2_directory,
                        self.dataset_2_metadata_path,
                    ),
                ),
                sizing_mode="stretch_width",
            )
        return pn.Column()

    def load_datasets_from_widgets(self) -> bool:
        """Load datasets based on current widget configurations.

        Collects dataset metadata from UI widgets, instantiates dataset
        wrapper objects, and stores them in `self.loaded_datasets`.

        Returns
        -------
        bool
            True if datasets were loaded successfully, False otherwise.
        """
        # Load dataset 1 (always required)
        if self.dataset_1_selector.value not in self.dataset_label_map:  # pragma: no cover
            self.status_source.emit("Please define dataset type")
            return False

        module = importlib.import_module(f"jatic_ri.core.{self.task}.dataset_loaders")
        load_datasets = module.load_datasets
        dataset_spec = module.DatasetSpecification

        # gather dataset information from the widgets
        if self.task == "object_detection":
            dataset_1_meta: dataset_spec = {
                "dataset_type": self.dataset_label_map[self.dataset_1_selector.value],
                "data_dir": self.dataset_1_directory.value,
                "metadata_path": self.dataset_1_metadata_path.value,
            }
        else:
            dataset_1_meta: dataset_spec = {
                "dataset_type": self.dataset_label_map[self.dataset_1_selector.value],
                "data_dir": self.dataset_1_directory.value,
                "split_folder": self.dataset_1_metadata_path.value,
            }

        dataset_meta = {"dataset_1": dataset_1_meta}

        # load dataset 2 only if the widget is visualized
        if self.dataset_2_visible:  # pragma: no cover
            if self.dataset_2_selector.value not in self.dataset_label_map:
                self.status_source.emit("Please define dataset 2 type")
                return False

            if self.task == "object_detection":
                dataset_2_meta: dataset_spec = {
                    "dataset_type": self.dataset_label_map[self.dataset_2_selector.value],
                    "data_dir": self.dataset_2_directory.value,
                    "metadata_path": self.dataset_2_metadata_path.value,
                }
            else:
                dataset_2_meta: dataset_spec = {
                    "dataset_type": self.dataset_label_map[self.dataset_2_selector.value],
                    "data_dir": self.dataset_2_directory.value,
                    "split_folder": self.dataset_2_metadata_path.value,
                }
            dataset_meta["dataset_2"] = dataset_2_meta

        # load the datasets
        self.loaded_datasets = load_datasets(dataset_meta)

        return True

    def _on_dataset_type_change(self, event) -> None:  # noqa: ANN001 # pragma: no cover
        """Handle changes in dataset type selection widgets.

        This callback updates the visibility, name, placeholder text, and
        description of associated dataset path and metadata widgets based on the
        selected dataset type for either dataset 1 or dataset 2.

        Parameters
        ----------
        event : Any
            The event object triggered by a dataset selector widget change.
        """

        def _set_widget_parameters(
            widget: Widget, name: str, placeholder: str, description: str, disabled: bool
        ) -> None:
            widget.name = name
            widget.placeholder = placeholder
            widget.description = description
            widget.disabled = disabled

        # dataset 1 selector changed
        if event.obj.name == "Dataset type":  # self.dataset_1_selector.name
            path_widget = self.dataset_1_directory
            metadata_widget = self.dataset_1_metadata_path
        # dataset 2 selector changed
        else:
            path_widget = self.dataset_2_directory
            metadata_widget = self.dataset_2_metadata_path

        dataset_type = event.obj.value
        if dataset_type == "COCO dataset":  # not sure about full filepath here
            path_name = "Images directory"
            path_placeholder = "Full path to images directory."
            path_description = "Full filepath to the directory containing dataset images."
            metadata_name = "Metadata filepath"
            metadata_placeholder = "Full filepath to dataset metadata file."
            metadata_description = "Full filepath to COCO-formatted annotation JSON file."
            _set_widget_parameters(path_widget, path_name, path_placeholder, path_description, False)
            _set_widget_parameters(metadata_widget, metadata_name, metadata_placeholder, metadata_description, False)

        elif dataset_type == "YOLO dataset":
            if self.task == "object_detection":
                path_name = "Annotations directory"
                path_placeholder = "Full filepath to annotations directory."
                path_description = "Full filepath to the directory containing the YOLO annotation files."
                metadata_name = "Metadata filepath"
                metadata_placeholder = "Full filepath to dataset metadata file."
                metadata_description = "Full filepath to YOLO-formatted metadata YAML file."
            else:
                path_name = "Root directory"
                path_placeholder = "Full filepath to root dataset directory."
                path_description = "Full filepath to root dataset directory containing split folders, each with their own category folders."  # noqa: E501
                metadata_name = "Split folder"
                metadata_placeholder = "Name of split foldername to load"
                metadata_description = "Within the root dataset directory, this folder contains a directory for each category which holds the images."  # noqa: E501
            _set_widget_parameters(path_widget, path_name, path_placeholder, path_description, False)
            _set_widget_parameters(metadata_widget, metadata_name, metadata_placeholder, metadata_description, False)

        elif dataset_type == "Visdrone dataset":
            path_name = "Root directory"
            path_placeholder = "Full filepath to root dataset directory."
            path_description = "Full filepath to the directory containing the images/ and annotations/ directories."
            _set_widget_parameters(path_widget, path_name, path_placeholder, path_description, False)
            metadata_widget.disabled = True

        elif dataset_type == "Define Dataset Type":
            path_widget.disabled = True
            metadata_widget.disabled = True
        else:
            raise RuntimeError("Dataset type selector not recognized.")

    def load_metric_from_widget(self) -> bool:
        """Load the metric based on the current widget selection.

        Collects input from the metric selector widget, instantiates the
        metric object, and stores it in `self.loaded_metric`.

        Returns
        -------
        bool
            True if the metric was loaded successfully.

        Warnings
        --------
        For Image Classification tasks, this method requires dataset information
        (number of classes) and thus must be run after datasets are loaded.
        """
        metrics_module = importlib.import_module(f"jatic_ri.core.{self.task}.metrics")
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
        """Instantiate test stage objects based on configurations.

        Parameters
        ----------
        configs : dict
            Contains the task definition and configurations for one or more
            test stages.

        Returns
        -------
        bool
            True if test stages were successfully instantiated, False otherwise.

        Examples
        --------
        Example `configs` structure:

        .. code-block:: python

            {
                "task": "object_detection",
                "survivor": {
                    "TYPE": "SurvivorTestStage",
                    "CONFIG": {
                        "metric_column": "metric",
                        "conversion_type": "original",
                        "otb_threshold": 0.5,
                        "easy_hard_threshold": 0.5,
                    },
                },
            }
        """
        logger.debug("load pipeline")
        self.threshold_visible = False
        self.dataset_2_visible = False
        self.test_stages: dict[Any, tuple[Capability, CapabilityConfigBase]] = {}
        if "task" not in configs:
            self.status_source.emit("Task must be specified in the provided config.")
            logger.debug("Task must be specified in the provided config.")
            return False
        if configs["task"] != self.task:
            self.status_source.emit(f"Mismatch between dashboard type, {self.task}, and provided config")
            return False
        for stage_label, config in configs.items():
            if stage_label != "task":
                self.status_source.emit(f'Loading {config["TYPE"]}')
                logger.debug(f'Loading {config["TYPE"]}')
                if self.task == "object_detection":
                    stage = get_capability_from_app_config_od(config)
                else:
                    stage = get_capability_from_app_config_ic(config)
                self.test_stages[stage_label] = stage
                # allow multiple models for multi-model test stages
                if config["TYPE"] == "RealLabelTestStage" or config["TYPE"] == "SurvivorTestStage":
                    # trigger addition of the "add model" button
                    self.multi_model_visible = True
                # enable dataset 2 for shift as needed
                elif config["TYPE"] == "DatasetShiftTestStage":
                    self.dataset_2_visible = True

                # TODO: determine a better way to display threshold for relevant stages only
                self.threshold_visible = True

        self.status_source.emit("Configuration file loaded")

        return True

    def _construct_report_filename(self) -> str:
        """Construct a filename for the output report.

        The filename is based on the configured test stages and the current timestamp.

        Returns
        -------
        str
            The generated report filename (without extension).
        """
        return f"{'-'.join(self.test_stages.keys())}_{datetime.now().strftime(TIMESTAMP_FORMAT)}"

    def _run_all_tests(self) -> str:  # pragma: no cover  # noqa: C901
        """Execute all configured test stages and generate a report.

        This method iterates through all test stages in `self.test_stages`,
        loads their inputs, runs them, collects their report consumables,
        and generates a PowerPoint report.

        Returns
        -------
        str
            If `self.local` is True, returns the path to the saved report file.
            Otherwise, returns an HTML string for a download link to the report.

        Notes
        -----
        This method is common across Image Classification and Object Detection
        use cases. It should typically be triggered within the subclass's
        `_run_button_callback` implementation.
        """
        self.status_source.emit("Processing. Please wait...")

        slides = []
        for stage in self.test_stages.values():
            capabilty = stage[0]
            self.status_source.emit(f"Loading inputs for {stage.__class__.__name__}")

            all_datasets = list(self.loaded_datasets.values())
            all_models = list(self.loaded_models.values())
            all_metrics = []

            if hasattr(self, "loaded_metric") and self.loaded_metric:
                self.loaded_metric.metadata["id"] = self.loaded_metric.return_key
                all_metrics = [self.loaded_metric]

            # For each test stages, we need to extract only the subset of input models, datasets, and metrics
            # that the stage supports, based on its supports_* attributes.

            # We assume that if multiple datasets, models, or metrics are loaded, we take the first N that the stage
            # supports in order.  (Also the UI currently only supports one metric but all of the code is included below
            # for symmetry.)

            # Assemble datasets based on stage's supports_datasets
            if capabilty.supports_datasets == Number.ZERO:
                datasets = []
            elif capabilty.supports_datasets == Number.ONE:
                datasets = all_datasets[:1] if all_datasets else []
            elif capabilty.supports_datasets == Number.TWO:
                datasets = all_datasets[:2] if len(all_datasets) >= 2 else all_datasets
            else:  # Number.MANY
                datasets = all_datasets

            # Assemble models based on stage's supports_models
            if capabilty.supports_models == Number.ZERO:
                models = []
            elif capabilty.supports_models == Number.ONE:
                models = all_models[:1] if all_models else []
            elif capabilty.supports_models == Number.TWO:
                models = all_models[:2] if len(all_models) >= 2 else all_models
            else:  # Number.MANY
                models = all_models

            # Assemble metrics based on stage's supports_metrics
            if capabilty.supports_metrics == Number.ZERO:
                metrics = []
            elif capabilty.supports_metrics == Number.ONE:
                metrics = all_metrics[:1] if all_metrics else []
            elif capabilty.supports_metrics == Number.TWO:
                metrics = all_metrics[:2] if len(all_metrics) >= 2 else all_metrics
            else:  # Number.MANY
                metrics = all_metrics

            # Build kwargs for stage.run, only including parameters the stage supports
            run_kwargs = {"use_cache": self.use_caches}

            if capabilty.supports_datasets != Number.ZERO:
                run_kwargs["datasets"] = datasets
            if capabilty.supports_models != Number.ZERO:
                run_kwargs["models"] = models
            if capabilty.supports_metrics != Number.ZERO:
                run_kwargs["metrics"] = metrics

            run_kwargs["config"] = stage[1]

            # run the stage, saving output to the class
            run = capabilty.run(**run_kwargs)
            # collect the slides
            stage_slides = run.collect_report_consumables(threshold=float(self.threshold))
            slides += stage_slides

        report_path = Path(self.output_dir)
        report_title = self._construct_report_filename()
        report = gd.create_deck(slides, report_path, deck_name=report_title)

        self.status_source.emit(f"Report saved to {report}")

        return (
            str(report_title)
            if self.local
            else create_download_link(
                str(report),
                label="Download Report",
                download_filename=f"{report_title}.pptx",
            )
        )

    def horizontal_line(self) -> pn.pane.HTML:
        """Create a Panel HTML pane representing a horizontal line.

        Returns
        -------
        pn.pane.HTML
            A Panel HTML pane for a styled horizontal line.

        Notes
        -----
        This is a method because the same Panel object cannot be visualized
        multiple times. A new object must be created each time a line is needed.
        """
        return pn.pane.HTML(
            """""",
            styles={
                "display": "block",
                "height": "1px",
                "border": "0",
                "border-top": f"1px solid {self.styles.color_gray_500}",
                "margin": "0em 0",
                "padding": "0",
            },
            width=self.styles.app_width - 48,
        )

    def view_status_bar(self) -> pn.Row:
        """Generate the view for the status bar.

        The status bar displays messages emitted to `self.status_source`.

        Returns
        -------
        pn.Row
            A Panel Row containing the styled status bar.

        Notes
        -----
        DO NOT OVERWRITE THIS METHOD in subclasses. To update status text,
        use `self.status_source.emit("Your message")`.
        """
        return pn.Row(
            pn.Spacer(width=12),
            pn.Column(
                pn.Spacer(height=4),
                pn.pane.Markdown(
                    "Status",
                    styles={**self.styles.style_text_body1, "color": self.styles.color_blue_900},
                    stylesheets=[self.styles.css_paragraph],
                ),
                self.status_pane,
                pn.Spacer(height=4),
                styles={
                    "background": self.styles.color_blue_100,
                    "border-color": self.styles.color_blue_300,
                    "border-width": "thin",
                    "border-style": "solid",
                    "border-radius": "3px",
                },
                sizing_mode="stretch_width",
            ),
            pn.Spacer(width=25),
        )

    def view_header(self) -> pn.Row:
        """Generate the view for the application header.

        The header typically includes the JATIC logo.

        Returns
        -------
        pn.Row
            A Panel Row containing the application header.
        """
        return pn.Row(
            pn.pane.SVG(JATIC_LOGO_PATH, width=150),
            styles={"background": self.styles.color_blue_900},
            width=self.styles.app_width,
        )

    def view_config(self) -> pn.Column:
        """Generate the view for displaying the current JSON configuration.

        Returns
        -------
        pn.Column
            A Panel Column containing a scrollable, preformatted display
            of `self.output_test_stages`.
        """
        logger.debug("view config container")
        return pn.Column(
            pn.pane.HTML(
                f"""<pre><code>{json.dumps(self.output_test_stages, indent=2)}</code></pre>""",
                sizing_mode="stretch_height",
            ),
            scroll=True,
            height=200,
        )

    def view_config_input(self) -> pn.Column:
        """Generate the UI section for viewing the loaded configuration.

        Returns
        -------
        pn.Column
            A Panel Column containing elements to display the current
            testbed configuration.
        """
        logger.debug("view config input container")
        return pn.Column(
            pn.Spacer(height=20),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown(
                        "1. View configuration file",
                        styles=self.styles.style_text_h3,
                        stylesheets=[self.styles.css_paragraph],
                    ),
                    pn.Row(
                        pn.Spacer(width=12),  # padding to align this with title text above
                        pn.pane.Markdown(
                            "View the current JSON configuration.",
                            styles=self.styles.style_text_body2,
                            stylesheets=[self.styles.css_paragraph],
                            width=395,
                        ),
                    ),
                ),
                pn.Spacer(width=124),
                # white box on the right:
                pn.Column(
                    pn.Spacer(height=18),
                    pn.Row(
                        pn.Spacer(width=12),
                        self.view_config,
                    ),
                    pn.Spacer(height=18),
                    styles={
                        "background": self.styles.color_white,
                        "border-color": self.styles.color_blue_300,
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
        """Generate the UI section for advanced options.

        This section typically includes controls for caching and threshold input.

        Returns
        -------
        pn.Row
            A Panel Row containing widgets for advanced settings.
        """
        logger.debug("view threshold metric")
        return pn.Row(
            pn.Spacer(width=18),  # padding to the left of the box
            pn.Column(
                pn.Spacer(height=20),  # padding at the top of the box
                # blue box:
                pn.Column(
                    pn.pane.Markdown(
                        "Advanced (optional)",
                        styles=self.styles.style_text_h3,
                        stylesheets=[self.styles.css_paragraph],
                    ),
                    pn.Row(
                        pn.Spacer(width=15),  # padding on the left inside of the blue box
                        pn.Column(
                            "Use Cache",
                            pn.widgets.Switch.from_param(
                                self.param.use_caches,
                                name="",
                                stylesheets=[self.styles.css_switch],
                            ),
                            pn.Spacer(height=10),  # padding for the bottom of the blue box
                        ),
                        pn.Column(
                            pn.Spacer(height=10),  # padding to make threshold align with usecache
                            pn.widgets.TextInput.from_param(
                                self.param.threshold,
                                name="Target Threshold",
                                width=175,
                                stylesheets=[self.styles.css_dropdown],
                                disabled=not (self.threshold_visible),
                            ),
                        ),
                        sizing_mode="stretch_width",
                        styles={
                            "background": self.styles.color_blue_100,
                            "border-color": self.styles.color_blue_300,
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
        """Generate the Tabulator widget for displaying results.

        Returns
        -------
        pn.widgets.Tabulator
            A Panel Tabulator widget configured to display `self.results_df`.

        Notes
        -----
        This is a separate method to facilitate dynamic updates of the
        DataFrame-based widget.
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
            stylesheets=[self.styles.css_tabulator_table],
            disabled=True,
            formatters=formatters,
            widths=column_widths,
        )

    def view_results_row(self) -> pn.Row:
        """Generate the UI section for displaying test results.

        This section contains the Tabulator widget for `self.results_df`.

        Returns
        -------
        pn.Row
            A Panel Row containing the test results table.
        """
        return pn.Row(
            pn.Spacer(width=10),  # padding to the left of the section
            pn.Column(
                pn.Row(
                    pn.Spacer(width=10),  # padding to the left of Test Results title
                    pn.pane.Markdown(
                        "Test Results",
                        styles=self.styles.style_text_h3,
                        stylesheets=[self.styles.css_paragraph],
                    ),
                ),
                self._view_df_tabulator,
            ),
        )

    def view_run_row(self) -> pn.Row:
        """Generate the UI row containing the 'Run Analysis' button.

        Returns
        -------
        pn.Row
            A Panel Row with the main action button for running the analysis.
        """
        return pn.Row(
            pn.layout.HSpacer(),
            self.run_analysis_button,
            pn.Spacer(width=15),
            sizing_mode="stretch_width",
        )

    def view_title_row(self) -> pn.Column:
        """Generate the UI section for the application title and description.

        This typically includes the task type (e.g., Object Detection),
        the workflow objective (e.g., Model Evaluation), and a brief
        description.

        Returns
        -------
        pn.Column
            A Panel Column containing title and descriptive text elements.
        """
        return pn.Column(
            pn.Spacer(height=10),
            pn.pane.Markdown(
                self.task.replace("_", " ").title(),
                styles=self.styles.style_text_body2,
                stylesheets=[self.styles.css_paragraph],
            ),
            pn.pane.Markdown(
                self.title,
                styles=self.styles.style_text_h2,
                stylesheets=[self.styles.css_paragraph],
            ),
            pn.pane.Markdown(
                "Configure your test setup to begin the analysis. You can view your results below",
                styles=self.styles.style_text_body2,
                stylesheets=[self.styles.css_paragraph],
            ),
        )

    def panel(self) -> pn.Column:
        """Construct the main Panel layout for the entire dashboard.

        This method assembles all the individual view components (header,
        status bar, configuration sections, results, etc.) into a cohesive
        Panel application.

        Returns
        -------
        pn.Column
            A Panel Column representing the complete dashboard UI.
        """
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
                        styles=self.styles.style_text_h2,
                        stylesheets=[self.styles.css_paragraph],
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
            styles={"background": self.styles.color_main_bg},
            width=self.styles.app_width,
            min_height=1600,
            sizing_mode="stretch_height",
        )
