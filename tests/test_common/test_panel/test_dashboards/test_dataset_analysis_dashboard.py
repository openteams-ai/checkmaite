import datetime as dt
import json
import os
from pathlib import Path
from unittest import mock

import pytest
import torch

import jatic_ri
from jatic_ri import PACKAGE_DIR
from jatic_ri._common._panel.dashboards.dataset_analysis_dashboard import DatasetAnalysisDashboard
from jatic_ri.object_detection.models import TorchvisionODModel

REPORT_PATH = "/report/path"
REPORT_LINK = "<a href='report-link'>Report Link</a>"


@pytest.mark.real_data
@pytest.mark.filterwarnings(r"ignore:.*?more than \d+ detections in a single image:UserWarning")
@pytest.mark.filterwarnings("ignore:All samples look discrete with so few data points:UserWarning")
@pytest.mark.filterwarnings(r"ignore:.*?did not meet the recommended \d+ occurrences:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Image must be larger than \d+x\d+:UserWarning")
@pytest.mark.filterwarnings(r"ignore:Bounding box .*? is out of bounds:UserWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
def test_dataset_analysis_dashboard_od_real_data(json_config_da_od, artifact_dir):
    """Test running of the DA dashboard for object detection."""
    # See https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/379
    json_config_da_od = json_config_da_od.copy()
    del json_config_da_od["reallabel"]

    app = DatasetAnalysisDashboard(
        task="object_detection",
        output_dir=artifact_dir,
    )

    # trigger the visualization to detect errors
    app.panel()

    # load in the config values
    app.config_file.value = json.dumps(json_config_da_od)

    ## Set up dataset
    # for OD - use sample dataset in the test suite
    coco_dataset_dir = PACKAGE_DIR.parent.parent.joinpath(
        Path("tests/testing_utilities/example_data/coco_resized_val2017")
    )
    app.dataset_1_selector.value = "Coco dataset"
    app.dataset_1_split_path.value = str(coco_dataset_dir)
    app.dataset_1_metadata_path.value = str(coco_dataset_dir.joinpath("instances_val2017_resized_6.json"))

    app.dataset_2_selector.value = "Coco dataset"
    app.dataset_2_split_path.value = str(coco_dataset_dir)
    app.dataset_2_metadata_path.value = str(coco_dataset_dir.joinpath("instances_val2017_resized_6.json"))

    # Set up model
    model_name = "ssdlite320_mobilenet_v3_large"
    model_wrapper = TorchvisionODModel(model_name=model_name)
    # Until issue 303 is implemented, dashboard will instantiate models by providing JSON config file
    # with same stem name (different extension)
    # Ref: https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/303
    config_path = "subdir/my_model.json"
    pickle_path = "subdir/my_model.pt"

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
@pytest.mark.filterwarnings(r"ignore:Unsupported placeholder:Warning")
def test_dataset_analysis_dashboard_od_model_widget_mechanics(baseline_eval_config_od, artifact_dir):
    """Test running of the DA dashboard for object detection.
    The actual run (_run_all_tests) is mocked for speed.
    """

    def _clear_app(app):
        """Clear the widgets"""
        app.model_widgets["Model 1 type"]["model_selector"].value = "Select Model type"
        app.model_widgets["Model 1 type"]["model_weights_path"].value = ""
        app.model_widgets["Model 1 type"]["model_config_path"].value = ""
        app.dataset_1_split_path.value = ""
        app.dataset_1_metadata_path.value = ""
        app.loaded_models = {}
        app.loaded_datasets = {}

    def _set_coco_dataset_1(app):
        coco_dataset_dir = jatic_ri.PACKAGE_DIR.parent.parent.joinpath(
            Path("tests/testing_utilities/example_data/coco_resized_val2017")
        )
        app.dataset_1_selector.value = "Coco dataset"
        app.dataset_1_split_path.value = str(coco_dataset_dir)
        app.dataset_1_metadata_path.value = str(coco_dataset_dir.joinpath("instances_val2017_resized_6.json"))

    def _set_visdrone_dataset_1(app):
        dataset_dir = jatic_ri.PACKAGE_DIR.parent.parent.joinpath(
            Path("tests/testing_utilities/example_data/visdrone_dataset")
        )
        app.dataset_1_selector.value = "Visdrone dataset"
        app.dataset_1_split_path.value = str(dataset_dir)
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
    app.config_file.value = json.dumps(baseline_eval_config_od)

    app.use_cache = True
    app.run_analysis_button.disabled = False

    # No model type selected
    # warn and skip
    # can't fully test until https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/346 is resolved
    _clear_app(app)
    app.load_models_from_widgets()
    assert "invalid type" in app.status_text
    assert not app.loaded_models

    # torchvision
    # no weights path
    # load default weights
    _clear_app(app)
    _set_coco_dataset_1(app)
    _set_model_1(app, model_name)
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
    app.run_analysis_button.clicks += 1
    # testing can be improved by checking status_text
    assert "Report saved to" in app.status_text
    assert app.loaded_models

    # visdrone
    # no weights/output dir
    # download to cwd and execute
    _clear_app(app)
    _set_visdrone_dataset_1(app)
    _set_model_1(app, "resnet18")
    app.run_analysis_button.clicks += 1
    assert "Report saved to" in app.status_text
    assert app.loaded_models

    # visdrone
    # weights/output dir provided
    # download to output dir and execute
    _clear_app(app)
    _set_visdrone_dataset_1(app)
    _set_model_1(app, "resnet18")
    _set_model_1_weights(app, str(artifact_dir))
    _set_model_1_config(app, config_path)
    app.run_analysis_button.clicks += 1
    assert "Report saved to" in app.status_text
    assert app.loaded_models


@pytest.mark.real_data
@pytest.mark.filterwarnings(r"ignore:.*?did not meet the recommended \d+ occurrences:UserWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
@pytest.mark.filterwarnings(
    "ignore:Precision loss occurred in moment calculation due to catastrophic cancellation:RuntimeWarning"
)
def test_dataset_analysis_dashboard_ic_real_data(json_config_da_ic, artifact_dir):
    """Test running of the DA dashboard for image_classification."""
    app = DatasetAnalysisDashboard(
        task="image_classification",
        output_dir=artifact_dir,
    )

    # trigger the visualization to detect errors
    app.panel()

    # load in the config values
    app.config_file.value = json.dumps(json_config_da_ic)

    ## Set up dataset
    # for IC - create a fake dataset
    from PIL import Image

    classes = ["cat", "dog"]
    num_images_per_class = 3
    img_shape = (64, 128)

    root_dir = Path("temp_yolo_dataset").resolve()
    split = "test"
    os.makedirs(root_dir / split, exist_ok=True)
    for class_name in classes:
        class_dir = root_dir / split / class_name
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_images_per_class):
            img = Image.new("RGB", img_shape, color=(i, i, i))
            img.save(class_dir / f"{i}_{class_name}.jpg")

    app.dataset_1_selector.value = "Yolo dataset"
    app.dataset_1_split_path.value = str(root_dir.joinpath(split))
    app.dataset_1_metadata_path.value = str(root_dir.joinpath("ann_file.json"))

    app.dataset_2_selector.value = "Yolo dataset"
    app.dataset_2_split_path.value = str(root_dir)
    app.dataset_2_metadata_path.value = str(root_dir.joinpath("ann_file.json"))

    # Set up model
    from jatic_ri.image_classification.models import TorchvisionICModel

    model_name = "resnext50_32x4d"
    model_wrapper = TorchvisionICModel(model_name=model_name)

    # Until issue 303 is implemented, dashboard will instantiate models by providing JSON config file
    # with same stem name (different extension)
    # Ref: https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/303
    config_path = "subdir/my_model.json"
    pickle_path = "subdir/my_model.pt"

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
def test_dataset_analysis_dashboard_od_mockrun_only(tmp_cache_path):
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
        app.config_file.value = json.dumps(od_config)

        ## Set up dataset
        # for OD - use sample dataset in the test suite
        coco_dataset_dir = PACKAGE_DIR.parent.parent.joinpath(Path("tests/testing_utilities/example_data/coco_dataset"))
        app.dataset_1_selector.value = "Coco dataset"
        app.dataset_1_split_path.value = str(coco_dataset_dir)
        app.dataset_1_metadata_path.value = str(coco_dataset_dir.joinpath("ann_file.json"))

        app.dataset_2_selector.value = "Coco dataset"
        app.dataset_2_split_path.value = str(coco_dataset_dir)
        app.dataset_2_metadata_path.value = str(coco_dataset_dir.joinpath("ann_file.json"))

        # Set up model
        model_name = "ssdlite320_mobilenet_v3_large"
        model_wrapper = TorchvisionODModel(model_name=model_name)

        # Until issue 303 is implemented, dashboard will instantiate models by providing JSON config file
        # with same stem name (different extension)
        # Ref: https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/303
        config_path = "subdir/my_model.json"
        pickle_path = "subdir/my_model.pt"

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
    app.config_file.value = json.dumps(od_config)

    # trigger the mocked run
    app.run_analysis_button.clicks += 1

    # ensure the results table was populated
    assert app.results_df["Gradient Report"][0] == REPORT_PATH if local else REPORT_LINK

    ## test report name generation
    report_title = app._construct_report_filename()
    assert dt.datetime.now().strftime("%Y%m%d_%H") in report_title
