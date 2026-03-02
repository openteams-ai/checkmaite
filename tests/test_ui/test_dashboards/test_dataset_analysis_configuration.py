"""Tests for the Dataset Analysis Configuration app for both object detection
and image classification.

Object detection includes Reallabel, Survivor, and the Dataeval capability
Image Classification includes Survivor and the Dataeval capability
"""

import json

import holoviews  # noqa (fixes issue with lazy holoviews import in Panel)
import pytest

from jatic_ri.ui.dashboards.combined_app import FullApp


def _reset_da_config_app(app: FullApp):
    """Reset everything on the DA config app landing page
    to false. This protects these tests against changes to default
    behavior. This works for both the IC and OD usecase
    """
    # ensure all the toggles are False
    app.pipeline._state.bias.value = False
    app.pipeline._state.shift.value = False
    app.pipeline._state.cleaning.value = False
    app.pipeline._state.feasibility.value = False
    # app.pipeline._state.show_survivor_config = False
    # app.pipeline._state.show_reallabel_config = False
    # clear output_test_stages
    app.pipeline._state.output_test_stages = {}


@pytest.mark.parametrize("task", ["object_detection", "image_classification"])
def test_route_da_bias_shift_cleaning(task):
    """Test route when bias, shift, and/or cleaning are selected

    ROUTE:
    LandingPage -> DAConfigurationLandingPage -> DatasetAnalysisDashboard

    Selecting bias, shift, and/or cleaning should not affect the route
    """
    workflow = "dataset_analysis"
    local = True
    # instantiate the pipeline
    app = FullApp(task=task, local=local, workflow=workflow)
    app.panel()

    # go to me od/ic landing page
    if task == "object_detection":
        app.pipeline._state.od_button.clicks += 1
    elif task == "image_classification":
        app.pipeline._state.ic_button.clicks += 1

    assert app.pipeline._state.__class__.__name__ == "DAConfigurationLandingPage"

    # reset the app (ensure this test is not affected by changing defaults)
    _reset_da_config_app(app)
    # with everything False, the next stage should be "DatasetAnalysisDashboard"
    assert app.pipeline._next_stage == "DatasetAnalysisDashboard"

    # toggle bias to true
    app.pipeline._state.bias.value = True
    # with bias selected, the next stage should still be "DatasetAnalysisDashboard"
    assert app.pipeline._next_stage == "DatasetAnalysisDashboard"

    # toggle shift to true
    app.pipeline._state.shift.value = True
    # with shift selected, the next stage should still be "DatasetAnalysisDashboard"
    assert app.pipeline._next_stage == "DatasetAnalysisDashboard"

    # toggle cleaning to true
    app.pipeline._state.cleaning.value = True
    # with cleaning selected, the next stage should still be "DatasetAnalysisDashboard"
    assert app.pipeline._next_stage == "DatasetAnalysisDashboard"

    # go to next stage
    app.pipeline._state.ready = True  # to avoid bug where non-display pipelines remain in unready state
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the final page by checking the class name
    assert app.pipeline._state.__class__.__name__ == "DatasetAnalysisDashboard"

    final_output = app.pipeline._state.output_test_stages
    assert len(final_output) == 4  # task, bias, shift, and cleaning should be present
    assert "bias" in final_output.keys()
    assert "shift" in final_output.keys()
    assert "cleaning" in final_output.keys()
    assert "task" in final_output.keys()


# NOTE: The Unsupported Capabilities have been removed from the Panel UI due to the complexity of reconstructing the pipeline
# graph for each combination of capabilities.  So this is commented out completely instead of @pytest.mark.unsupported
#
# def test_route_da_od_reallabel():
#     """Test route when only reallabel is selected.
#     Reallabel is OD only.

#     ROUTE:
#     LandingPage -> DAConfigurationLandingPage -> Configure Reallabel -> DatasetAnalysisDashboard
#     """
#     task = "object_detection"
#     workflow = "dataset_analysis"
#     local = True
#     # instantiate the pipeline
#     app = FullApp(task=task, local=local, workflow=workflow)
#     app.panel()

#     # go to da od landing page
#     app.pipeline._state.od_button.clicks += 1
#     assert app.pipeline._state.__class__.__name__ == "DAConfigurationLandingPage"

#     # reset the app (ensure this test is not affected by changing defaults)
#     _reset_da_config_app(app)

#     # toggle reallabel to true
#     app.pipeline._state.show_reallabel_config = True

#     # with only reallabel true, the next stage should be reallabel
#     assert app.pipeline._next_stage == "Configure Reallabel"

#     # go to next stage
#     app.pipeline.next_button.clicks += 1
#     # ensure we actually went to the final page by checking the class name
#     assert app.pipeline._state.__class__.__name__ == "RealLabelApp"

#     # go to next stage
#     app.pipeline.next_button.clicks += 1

#     # ensure we actually went to the final page by checking the class name
#     assert app.pipeline._state.__class__.__name__ == "DatasetAnalysisDashboard"

#     final_output = app.pipeline._state.output_test_stages
#     assert len(final_output) == 2  # task and reallabel should be present
#     assert "reallabel_test_stage" in final_output.keys()

# NOTE: The Unsupported Capabilities have been removed from the Panel UI due to the complexity of reconstructing the pipeline
# graph for each combination of capabilities.  So this is commented out completely instead of @pytest.mark.unsupported
#
# def test_route_da_od_reallabel_survivor():
#     """Test route when only reallabel and survivor are selected
#     Reallabel is OD only.

#     ROUTE:
#     LandingPage -> DAConfigurationLandingPage -> Configure Reallabel -> ConfigureSurvivorOD -> DatasetAnalysisDashboard
#     """
#     task = "object_detection"
#     workflow = "dataset_analysis"
#     local = True
#     # instantiate the pipeline
#     app = FullApp(task=task, local=local, workflow=workflow)
#     app.panel()

#     # go to da od landing page
#     app.pipeline._state.od_button.clicks += 1
#     assert app.pipeline._state.__class__.__name__ == "DAConfigurationLandingPage"

#     # reset the app (ensure this test is not affected by changing defaults)
#     _reset_da_config_app(app)

#     # toggle reallabel and reallabel to true
#     app.pipeline._state.show_reallabel_config = True
#     app.pipeline._state.show_survivor_config = True

#     # with reallabel true, the next stage should be reallabel
#     assert app.pipeline._next_stage == "Configure Reallabel"

#     # go to next stage
#     app.pipeline.next_button.clicks += 1
#     # ensure we actually went to the correct page by checking the class name
#     assert app.pipeline._state.__class__.__name__ == "RealLabelApp"

#     # go to next stage
#     app.pipeline.next_button.clicks += 1
#     # ensure we actually went to the correct page by checking the class name
#     assert app.pipeline._state.__class__.__name__ == "SurvivorAppOD"

#     # go to next stage
#     app.pipeline.next_button.clicks += 1
#     # ensure we actually went to the correct page by checking the class name
#     assert app.pipeline._state.__class__.__name__ == "DatasetAnalysisDashboard"

#     final_output = app.pipeline._state.output_test_stages
#     assert len(final_output) == 3  # task, survivor and reallabel should be present
#     assert "reallabel_test_stage" in final_output.keys()
#     assert "survivor_test_stage" in final_output.keys()

#     # convert the on-disk formatted configs into instantiated test stages
#     app.pipeline._state.load_pipeline(app.pipeline._state.output_test_stages)

#     assert len(app.pipeline._state.test_stages) == len(app.pipeline._state.output_test_stages) - 1

# @pytest.mark.parametrize("task", ["object_detection", "image_classification"])
# def test_route_da_od_survivor(task):
#     """Test route when only survivor is selected.

#     ROUTE:
#     LandingPage -> DAConfigurationLandingPage -> ConfigureSurvivorOD -> DatasetAnalysisDashboard
#     """
#     workflow = "dataset_analysis"
#     local = True

#     # instantiate the pipeline
#     app = FullApp(task=task, local=local, workflow=workflow)
#     app.panel()

#     # go to me od/ic landing page
#     state = app.pipeline._state
#     button = getattr(state, f"{app.suffix.lower()}_button")
#     button.clicks += 1
#     assert app.pipeline._state.__class__.__name__ == "DAConfigurationLandingPage"

#     # reset the app (ensure this test is not affected by changing defaults)
#     _reset_da_config_app(app)

#     # toggle survivor to true
#     app.pipeline._state.show_survivor_config = True

#     # with only survivor true, the next stage should be survivor
#     assert app.pipeline._next_stage == f"Configure Survivor{app.suffix}"

#     # go to next stage
#     app.pipeline.next_button.clicks += 1
#     # ensure we actually went to the correct page by checking the class name
#     assert app.pipeline._state.__class__.__name__ == f"SurvivorApp{app.suffix}"

#     # go to next stage
#     app.pipeline.next_button.clicks += 1
#     # ensure we actually went to the correct page by checking the class name
#     assert app.pipeline._state.__class__.__name__ == "DatasetAnalysisDashboard"

#     final_output = app.pipeline._state.output_test_stages
#     assert len(final_output) == 2  # task and survivor should be present
#     assert "survivor_test_stage" in final_output.keys()


def test_route_da_config_load_od(json_config_da_od):
    """
    Test route when loading a configuration from a file.
    This should load the configuration and go to the final page.

    ROUTE:
    LandingPage -> DatasetAnalysisDashboard
    """
    task = "object_detection"
    workflow = "dataset_analysis"
    local = True

    # instantiate the pipeline
    app = FullApp(task=task, local=local, workflow=workflow)
    app.panel()

    # loading data into the file dropper should automatically send us to the final page
    app.pipeline._state.file_dropper.value = {"loaded_config": json.dumps(json_config_da_od)}

    # ensure we actually went to the correct page by checking the class name
    assert app.pipeline._state.__class__.__name__ == "DatasetAnalysisDashboard"

    assert len(app.pipeline._state.output_test_stages) == len(json_config_da_od)

    # convert the on-disk formatted configs into instantiated test stages
    app.pipeline._state.load_pipeline(app.pipeline._state.output_test_stages)

    assert len(json_config_da_od) - 1 == len(app.pipeline._state.test_stages)


def test_route_da_config_load_ic(json_config_da_ic):
    """
    Test route when loading a configuration from a file.
    This should load the configuration and go to the final page.

    ROUTE:
    LandingPage -> DatasetAnalysisDashboard
    """
    task = "image_classification"
    workflow = "dataset_analysis"
    local = True

    # instantiate the pipeline
    app = FullApp(task=task, local=local, workflow=workflow)
    app.panel()

    # loading data into the file dropper should automatically send us to the final page
    app.pipeline._state.file_dropper.value = {"loaded_config": json.dumps(json_config_da_ic)}

    # ensure we actually went to the correct page by checking the class name
    assert app.pipeline._state.__class__.__name__ == "DatasetAnalysisDashboard"

    assert len(app.pipeline._state.output_test_stages) == len(json_config_da_ic)

    # convert the on-disk formatted configs into instantiated test stages
    app.pipeline._state.load_pipeline(app.pipeline._state.output_test_stages)

    assert len(json_config_da_ic) - 1 == len(app.pipeline._state.test_stages)


def test_da_workflow_change():
    """Test that changing the workflow from DA to ME works correctly."""
    task = "object_detection"
    workflow = "dataset_analysis"
    local = True

    # instantiate the pipeline
    app = FullApp(task=task, local=local, workflow=workflow)
    app.panel()

    assert app.pipeline._next_stage == "DAConfigurationLandingPage"

    app.pipeline._state.me_da_toggle.value = "model_evaluation"

    assert app.pipeline._next_stage == "MEConfigurationLandingPage"
