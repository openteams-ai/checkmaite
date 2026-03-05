import datetime as dt
import json
import os
from pathlib import Path
from unittest import mock

import pytest
import torch
from PIL import Image

from checkmaite.core.object_detection.models import TorchvisionODModel
from checkmaite.ui.dashboards.dataset_analysis_dashboard import DatasetAnalysisDashboard

REPORT_PATH = "/report/path"
REPORT_LINK = "<a href='report-link'>Report Link</a>"
TEST_DATA_DIR = Path(__file__).parents[2] / "data_for_tests"


@pytest.fixture
def yolo_dataset(artifact_dir):
    """Generate yolo dataset one image and two classes"""
    ## Set up dataset
    classes = ["cat", "dog"]
    num_images_per_class = 3
    img_shape = (64, 128)

    root_dir = Path(artifact_dir) / "temp_yolo_dataset"
    split = "test"
    os.makedirs(root_dir / split, exist_ok=True)
    for class_name in classes:
        class_dir = root_dir / split / class_name
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_images_per_class):
            img = Image.new("RGB", img_shape, color=(i, i, i))
            img.save(class_dir / f"{i}_{class_name}.jpg")

    return root_dir


@pytest.mark.unsupported
@pytest.mark.filterwarnings(r"ignore:.*?more than \d+ detections in a single image:UserWarning")
@pytest.mark.filterwarnings("ignore:All samples look discrete with so few data points:UserWarning")
@pytest.mark.filterwarnings(r"ignore:.*?did not meet the recommended \d+ occurrences:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Image must be larger than \d+x\d+:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Bounding box .*? is out of bounds:UserWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
def test_dataset_analysis_dashboard_od_real_data(json_config_da_od, artifact_dir, tmp_path):
    """Test running of the DA dashboard for object detection."""

    app = DatasetAnalysisDashboard(
        task="object_detection",
        output_dir=artifact_dir,
    )

    # trigger the visualization to detect errors
    app.panel()

    # load in the config values
    app.output_test_stages = json_config_da_od

    ## Set up dataset
    # for OD - use sample dataset in the test suite
    coco_dataset_dir = TEST_DATA_DIR / "coco_resized_val2017"

    app.dataset_1_selector.value = "COCO dataset"
    app.dataset_1_directory.value = str(coco_dataset_dir)
    app.dataset_1_metadata_path.value = str(coco_dataset_dir.joinpath("instances_val2017_resized_6.json"))

    app.dataset_2_selector.value = "COCO dataset"
    app.dataset_2_directory.value = str(coco_dataset_dir)
    app.dataset_2_metadata_path.value = str(coco_dataset_dir.joinpath("instances_val2017_resized_6.json"))

    # Set up model
    model_name = "ssdlite320_mobilenet_v3_large"
    model_wrapper = TorchvisionODModel(model_name=model_name)
    # Until issue 303 is implemented, dashboard will instantiate models by providing JSON config file
    # with same stem name (different extension)
    # Ref: https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/303
    config_path = f"{tmp_path}/my_model.json"
    pickle_path = f"{tmp_path}/my_model.pt"

    # save metadata and state_dict to disk
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump({"index2label": model_wrapper.index2label}, f)
    _ = torch.save(model_wrapper.model.state_dict(), pickle_path)

    visualized_model_name = {value: key for key, value in app.model_label_map.items()}[model_name]
    app.model_widgets["Model 1 type"]["model_selector"].value = visualized_model_name
    app.model_widgets["Model 1 type"]["model_weights_path"].value = pickle_path
    app.model_widgets["Model 1 type"]["model_config_path"].value = config_path

    # trigger the mocked run
    app.run_analysis_button.clicks += 1

    title_ymd_h = dt.datetime.now().strftime("%Y%m%d_%H")
    # ensure the results table was populated
    assert title_ymd_h in app.results_df["Gradient Report"][0].replace(" ", "_").lower()

    ## test report name generation
    report_title = app._construct_report_filename()
    assert title_ymd_h in report_title


@pytest.mark.unsupported
@pytest.mark.filterwarnings(r"ignore:Unsupported placeholder:Warning")
def test_dataset_analysis_dashboard_od_model_widget_mechanics(baseline_eval_config_od, artifact_dir):
    """Test running of the DA dashboard for object detection.
    The actual run (_run_all_tests) is mocked for speed.
    UI requires gradient.
    """

    def _clear_app(app):
        """Clear the widgets"""
        app.model_widgets["Model 1 type"]["model_selector"].value = "Select Model type"
        app.model_widgets["Model 1 type"]["model_weights_path"].value = ""
        app.model_widgets["Model 1 type"]["model_config_path"].value = ""
        app.dataset_1_directory.value = ""
        app.dataset_1_metadata_path.value = ""
        app.loaded_models = {}
        app.loaded_datasets = {}

    def _set_coco_dataset_1(app):
        coco_dataset_dir = TEST_DATA_DIR / "coco_resized_val2017"

        app.dataset_1_selector.value = "COCO dataset"
        app.dataset_1_directory.value = str(coco_dataset_dir)
        app.dataset_1_metadata_path.value = str(coco_dataset_dir.joinpath("instances_val2017_resized_6.json"))

    def _set_visdrone_dataset_1(app):
        dataset_dir = TEST_DATA_DIR / "visdrone_dataset"

        app.dataset_1_selector.value = "Visdrone dataset"
        app.dataset_1_directory.value = str(dataset_dir)
        app.dataset_1_metadata_path.value = ""

    # Set up model
    model_name = "ssdlite320_mobilenet_v3_large"
    model_wrapper = TorchvisionODModel(model_name=model_name)
    config_path = f"{artifact_dir}/my_model.json"
    pickle_path = f"{artifact_dir}/my_model.pt"
    # save metadata and state_dict to disk
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump({"index2label": model_wrapper.index2label}, f)
    _ = torch.save(model_wrapper.model.state_dict(), pickle_path)

    def _set_model_1(app, model_name):
        visualized_model_name = {value: key for key, value in app.model_label_map.items()}[model_name]
        app.model_widgets["Model 1 type"]["model_selector"].value = visualized_model_name

    def _set_model_1_weights(app, path):
        app.model_widgets["Model 1 type"]["model_weights_path"].value = path

    def _set_model_1_config(app, path):
        app.model_widgets["Model 1 type"]["model_config_path"].value = path

    app = DatasetAnalysisDashboard(
        task="object_detection",
        output_dir=artifact_dir,
    )

    # trigger the visualization to detect errors
    app.panel()

    # load in the config values
    app.output_test_stages = baseline_eval_config_od

    app.use_cache = True
    app.run_analysis_button.disabled = False

    # No model type selected
    # warn and skip
    # can't fully test until https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/346 is resolved
    _clear_app(app)
    app.load_models_from_widgets()
    assert "invalid type" in app.status_source.current_value
    assert not app.loaded_models

    # torchvision
    # no weights path
    # load default weights
    _clear_app(app)
    _set_coco_dataset_1(app)
    _set_model_1(app, model_name)
    app.load_models_from_widgets()
    app.run_analysis_button.clicks += 1
    assert app.loaded_models

    # torchvision
    # weights provided
    # no config path
    # warn and skip
    _clear_app(app)
    _set_coco_dataset_1(app)
    _set_model_1(app, model_name)
    _set_model_1_weights(app, pickle_path)
    app.load_models_from_widgets()
    app.run_analysis_button.clicks += 1
    assert not app.loaded_models

    # torchvision
    # weights provided
    # config provided
    # load custom weights and config
    _clear_app(app)
    _set_coco_dataset_1(app)
    _set_model_1(app, model_name)
    _set_model_1_weights(app, pickle_path)
    _set_model_1_config(app, config_path)
    app.load_models_from_widgets()
    app.run_analysis_button.clicks += 1
    # testing can be improved by checking app.status_source.current_value
    assert "Report saved to" in app.status_source.current_value
    assert app.loaded_models

    # visdrone
    # weights/output dir provided
    # download to output dir and execute
    _clear_app(app)
    _set_visdrone_dataset_1(app)
    _set_model_1(app, "resnet18")
    _set_model_1_weights(app, str(artifact_dir))
    _set_model_1_config(app, config_path)
    app.load_models_from_widgets()
    app.run_analysis_button.clicks += 1
    assert "Report saved to" in app.status_source.current_value
    assert app.loaded_models


@pytest.mark.unsupported
@pytest.mark.filterwarnings(r"ignore:.*?did not meet the recommended \d+ occurrences:UserWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
@pytest.mark.filterwarnings(
    "ignore:Precision loss occurred in moment calculation due to catastrophic cancellation:RuntimeWarning"
)
def test_dataset_analysis_dashboard_ic_real_data(json_config_da_ic, artifact_dir, yolo_dataset, tmp_path):
    """Test running of the DA dashboard for image_classification. UI requires gradient."""
    app = DatasetAnalysisDashboard(
        task="image_classification",
        output_dir=artifact_dir,
    )

    # trigger the visualization to detect errors
    app.panel()

    # load in the config values
    app.output_test_stages = json_config_da_ic

    app.dataset_1_selector.value = "YOLO dataset"
    app.dataset_1_directory.value = str(yolo_dataset)
    app.dataset_1_metadata_path.value = "test"

    app.dataset_2_selector.value = "YOLO dataset"
    app.dataset_2_directory.value = str(yolo_dataset)
    app.dataset_2_metadata_path.value = "test"

    # Set up model
    from checkmaite.core.image_classification.models import TorchvisionICModel

    model_name = "resnext50_32x4d"
    model_wrapper = TorchvisionICModel(model_name=model_name)

    # Until issue 303 is implemented, dashboard will instantiate models by providing JSON config file
    # with same stem name (different extension)
    # Ref: https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/303
    config_path = f"{tmp_path}/my_model.json"
    pickle_path = f"{tmp_path}/my_model.pt"

    # save metadata and state_dict to disk
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump({"index2label": model_wrapper.index2label}, f)
    _ = torch.save(model_wrapper.model.state_dict(), pickle_path)

    visualized_model_name = {value: key for key, value in app.model_label_map.items()}[model_name]
    app.model_widgets["Model 1 type"]["model_selector"].value = visualized_model_name
    app.model_widgets["Model 1 type"]["model_weights_path"].value = pickle_path
    app.model_widgets["Model 1 type"]["model_config_path"].value = config_path

    # trigger the mocked run
    app.run_analysis_button.clicks += 1

    title_ymd_h = dt.datetime.now().strftime("%Y%m%d_%H")
    # ensure the results table was populated
    assert title_ymd_h in app.results_df["Gradient Report"][0].replace(" ", "_").lower()

    ## test report name generation
    report_title = app._construct_report_filename()
    assert title_ymd_h in report_title


@pytest.mark.real_data
def test_dataset_analysis_dashboard_od_mockrun_only(tmp_cache_path, tmp_path):
    """Test running of the DA dashboard for object detection.
    The actual run (_run_all_tests) is mocked for speed.
    """

    with mock.patch.object(DatasetAnalysisDashboard, "_run_all_tests") as _run_all_tests_mocked:
        _run_all_tests_mocked.return_value = REPORT_LINK
        app = DatasetAnalysisDashboard(task="object_detection", output_dir=tmp_cache_path)

        # trigger the visualization to detect errors
        app.panel()

        ## Test loading simple config
        od_config = {
            "task": "object_detection",
            "shift": {
                "TYPE": "DatasetShiftTestStage",
            },
        }

        # load in the config values
        app.output_test_stages = od_config

        ## Set up dataset
        # for OD - use sample dataset in the test suite
        coco_dataset_dir = TEST_DATA_DIR / "coco_dataset"
        app.dataset_1_selector.value = "COCO dataset"
        app.dataset_1_directory.value = str(coco_dataset_dir)
        app.dataset_1_metadata_path.value = str(coco_dataset_dir.joinpath("ann_file.json"))

        app.dataset_2_selector.value = "COCO dataset"
        app.dataset_2_directory.value = str(coco_dataset_dir)
        app.dataset_2_metadata_path.value = str(coco_dataset_dir.joinpath("ann_file.json"))

        # Set up model
        model_name = "ssdlite320_mobilenet_v3_large"
        model_wrapper = TorchvisionODModel(model_name=model_name)

        # Until issue 303 is implemented, dashboard will instantiate models by providing JSON config file
        # with same stem name (different extension)
        # Ref: https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/303
        config_path = f"{tmp_path}/my_model.json"
        pickle_path = f"{tmp_path}/my_model.pt"

        # save metadata and state_dict to disk
        with open(config_path, "w") as f:
            json.dump({"index2label": model_wrapper.index2label}, f)
        _ = torch.save(model_wrapper.model.state_dict(), pickle_path)

        visualized_model_name = {value: key for key, value in app.model_label_map.items()}[model_name]
        app.model_widgets["Model 1 type"]["model_selector"].value = visualized_model_name
        app.model_widgets["Model 1 type"]["model_weights_path"].value = pickle_path
        app.model_widgets["Model 1 type"]["model_config_path"].value = config_path

        # trigger the mocked run
        app.run_analysis_button.clicks += 1

        # ensure the results table was populated
        assert len(app.results_df) > 0
        assert REPORT_LINK in app.results_df["Gradient Report"][0]

        ## test report name generation
        report_title = app._construct_report_filename()
        assert dt.datetime.now().strftime("%Y%m%d_%H") in report_title


@pytest.mark.parametrize("local", [True, False])
def test_dataset_analysis_dashboard_od_full_mock(
    local, monkeypatch, fake_od_model_default, fake_od_dataset_default, tmp_cache_path
):
    """Test running of the DA dashboard for object detection.
    The actual run (_run_all_tests) is mocked for speed.
    """

    def _run_all_tests_mocked(self):
        return REPORT_PATH if self.local else REPORT_LINK

    def load_models_from_widgets_mocked(self):
        self.loaded_models = {"fake_model_1": fake_od_model_default}
        return True

    def load_datasets_from_widgets_mocked(self):
        self.loaded_datasets = {"fake_dataset_1": fake_od_dataset_default}
        return True

    monkeypatch.setattr(DatasetAnalysisDashboard, "_run_all_tests", _run_all_tests_mocked)
    monkeypatch.setattr(DatasetAnalysisDashboard, "load_models_from_widgets", load_models_from_widgets_mocked)
    monkeypatch.setattr(DatasetAnalysisDashboard, "load_datasets_from_widgets", load_datasets_from_widgets_mocked)

    app = DatasetAnalysisDashboard(task="object_detection", output_dir=tmp_cache_path, local=local)

    # trigger the visualization to detect errors
    app.panel()

    ## Test loading simple config
    od_config = {
        "task": "object_detection",
        "shift": {
            "TYPE": "DatasetShiftTestStage",
        },
    }

    # load in the config values
    app.output_test_stages = od_config

    # trigger the mocked run
    app.run_analysis_button.clicks += 1

    # ensure the results table was populated
    assert app.results_df["Gradient Report"][0] == REPORT_PATH if local else REPORT_LINK

    ## test report name generation
    report_title = app._construct_report_filename()
    assert dt.datetime.now().strftime("%Y%m%d_%H") in report_title
