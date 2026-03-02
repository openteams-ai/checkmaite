import json

import pytest

from jatic_ri.ui._common.model_evaluation_configuration import MEConfigurationLandingPage
from jatic_ri.ui.dashboards.combined_app import FullApp


def _reset_me_config_app(app: MEConfigurationLandingPage):
    """Reset everything on the ME config app landing page
    to false. This protects these tests against changes to default
    behavior.
    """
    # ensure all the toggles are False
    app.pipeline._state.baseline_eval.value = False
    app.pipeline._state.show_xaitk_config = False
    app.pipeline._state.show_nrtk_config = False
    # clear output_test_stages
    app.pipeline._state.output_test_stages = {}


@pytest.mark.parametrize("local", [True, False])
def test_route_me_od_none(local):
    """Test the route to the final page of the pipeline

    ROUTE:
    LandingPage -> MEConfigurationLandingPage -> ModelEvaluationTestbed

    No selections means it goes straight to final page
    """
    task = "object_detection"
    workflow = "model_evaluation"
    # instantiate the pipeline
    app = FullApp(task=task, local=local, workflow=workflow)
    app.panel()

    # go to da od landing page
    app.pipeline._state.od_button.clicks += 1
    assert app.pipeline._state.__class__.__name__ == "MEConfigurationLandingPage"

    # reset the app (ensure this test is not affected by changing defaults)
    _reset_me_config_app(app)

    # with everything False, the next stage should be "ModelEvaluationTestbed"
    assert app.pipeline._next_stage == "ModelEvaluationTestbed"

    # go to next stage
    app.pipeline._state.ready = True  # to avoid bug where non-display pipelines remain in unready state
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the final page by checking the class name
    assert app.pipeline._state.__class__.__name__ == "ModelEvaluationTestbed"

    final_output = app.pipeline._state.output_test_stages
    assert len(final_output) == 1  # only "task" should be present
    assert "task" in final_output.keys()
    assert final_output["task"] == task
    assert app.pipeline._state.local == local


@pytest.mark.parametrize("task", ["object_detection", "image_classification"])
def test_route_me_eval(task):
    """Test route when only baseline evaluation is selected

    ROUTE:
    LandingPage -> MEConfigurationLandingPage -> ModelEvaluationTestbed

    Selecting bias, shift, and/or cleaning should not affect the route
    """
    workflow = "model_evaluation"
    local = True
    # instantiate the pipeline
    app = FullApp(task=task, local=local, workflow=workflow)
    app.panel()

    # go to da od/ic landing page
    state = app.pipeline._state
    button = getattr(state, f"{app.suffix.lower()}_button")
    button.clicks += 1
    assert app.pipeline._state.__class__.__name__ == "MEConfigurationLandingPage"

    # reset the app (ensure this test is not affected by changing defaults)
    _reset_me_config_app(app)

    # toggle baseline_evaluate to true
    app.pipeline._state.baseline_eval.value = True
    # with bias selected, the next stage should still be "ModelEvaluationTestbed"
    assert app.pipeline._next_stage == "ModelEvaluationTestbed"

    # go to next stage
    app.pipeline._state.ready = True  # to avoid bug where non-display pipelines remain in unready state
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the final page by checking the class name
    assert app.pipeline._state.__class__.__name__ == "ModelEvaluationTestbed"

    final_output = app.pipeline._state.output_test_stages
    assert len(final_output) == 2  # task and baseline eval should be present
    assert "baseline_evaluate" in final_output.keys()


@pytest.mark.parametrize("task", ["object_detection", "image_classification"])
def test_route_me_nrtk_only(task):
    """Test route when only nrtk is selected

    ROUTE:
    LandingPage -> MEConfigurationLandingPage -> ConfigureNRTK{suffix} -> ModelEvaluationTestbed
    """
    workflow = "model_evaluation"
    local = True

    # instantiate the pipeline
    app = FullApp(task=task, local=local, workflow=workflow)
    app.panel()

    # go to me od/ic landing page
    state = app.pipeline._state
    button = getattr(state, f"{app.suffix.lower()}_button")
    button.clicks += 1
    assert app.pipeline._state.__class__.__name__ == "MEConfigurationLandingPage"

    # reset the app (ensure this test is not affected by changing defaults)
    _reset_me_config_app(app)

    # toggle nrtk to true
    app.pipeline._state.show_nrtk_config = True

    # with only nrtk true, the next stage should be nrtk
    assert app.pipeline._next_stage == f"Configure NRTK{app.suffix}"

    # go to next stage
    app.pipeline._state.ready = True  # to avoid bug where non-display pipelines remain in unready state
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the correct page by checking the class name
    assert app.pipeline._state.__class__.__name__ == f"NRTKApp{app.suffix}"
    # add perturber factory
    app.pipeline._state.add_button.clicks += 1
    # add perturber factory
    app.pipeline._state.add_button.clicks += 1

    # go to next stage
    app.pipeline._state.ready = True  # to avoid bug where non-display pipelines remain in unready state
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the correct page by checking the class name
    assert app.pipeline._state.__class__.__name__ == "ModelEvaluationTestbed"

    final_output = app.pipeline._state.output_test_stages
    assert len(final_output) == 3  # task and 2 nrtk should be present
    assert f"NRTKApp{app.suffix}_0" in final_output.keys()

    # convert the on-disk formatted configs into instantiated test stages
    app.pipeline._state.load_pipeline(app.pipeline._state.output_test_stages)

    assert len(app.pipeline._state.test_stages) == len(app.pipeline._state.output_test_stages) - 1


@pytest.mark.parametrize("task", ["object_detection", "image_classification"])
def test_route_me_nrtk_xaitk(task):
    """Test route when both nrtk and xaitk are selected
    ROUTE:
    LandingPage -> MEConfigurationLandingPage -> ConfigureNRTK{suffix} -> ConfigureXAITK{suffix} -> ModelEvaluationTestbed
    """
    workflow = "model_evaluation"
    local = True
    # instantiate the pipeline
    app = FullApp(task=task, local=local, workflow=workflow)
    app.panel()
    # go to me od/ic landing page
    state = app.pipeline._state
    button = getattr(state, f"{app.suffix.lower()}_button")
    button.clicks += 1
    assert app.pipeline._state.__class__.__name__ == "MEConfigurationLandingPage"
    # reset the app (ensure this test is not affected by changing defaults)
    _reset_me_config_app(app)
    # toggle nrtk to true
    app.pipeline._state.show_nrtk_config = True
    app.pipeline._state.show_xaitk_config = True
    # with only nrtk true, the next stage should be nrtk
    assert app.pipeline._next_stage == f"Configure NRTK{app.suffix}"
    # go to next stage
    app.pipeline._state.ready = True  # to avoid bug where non-display pipelines remain in unready state
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the correct page by checking the class name
    assert app.pipeline._state.__class__.__name__ == f"NRTKApp{app.suffix}"
    # add perturber factory
    app.pipeline._state.add_button.clicks += 1
    # go to next stage
    app.pipeline._state.ready = True  # to avoid bug where non-display pipelines remain in unready state
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the correct page by checking the class name
    assert app.pipeline._state.__class__.__name__ == f"XAITKApp{app.suffix}"
    # go to next stage
    app.pipeline._state.ready = True  # to avoid bug where non-display pipelines remain in unready state
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the correct page by checking the class name
    assert app.pipeline._state.__class__.__name__ == "ModelEvaluationTestbed"
    final_output = app.pipeline._state.output_test_stages
    assert len(final_output) == 3  # task, nrtk and xaitk should be present
    assert f"NRTKApp{app.suffix}_0" in final_output.keys()
    assert f"XAITKApp{app.suffix}_0" in final_output.keys()
    # convert the on-disk formatted configs into instantiated test stages
    app.pipeline._state.load_pipeline(app.pipeline._state.output_test_stages)
    assert len(app.pipeline._state.test_stages) == len(app.pipeline._state.output_test_stages) - 1


@pytest.mark.parametrize("task", ["object_detection", "image_classification"])
def test_route_me_xaitk_only(task):
    """Test route when only xaitk is selected
    ROUTE:
    LandingPage -> MEConfigurationLandingPage -> ConfigureXAITK{suffix} -> ModelEvaluationTestbed
    """
    workflow = "model_evaluation"
    local = True
    # instantiate the pipeline
    app = FullApp(task=task, local=local, workflow=workflow)
    app.panel()
    # go to me od/ic landing page
    state = app.pipeline._state
    button = getattr(state, f"{app.suffix.lower()}_button")
    button.clicks += 1
    assert app.pipeline._state.__class__.__name__ == "MEConfigurationLandingPage"
    # reset the app (ensure this test is not affected by changing defaults)
    _reset_me_config_app(app)
    # toggle nrtk to true
    app.pipeline._state.show_xaitk_config = True
    # go to next stage
    app.pipeline._state.ready = True  # to avoid bug where non-display pipelines remain in unready state
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the correct page by checking the class name
    assert app.pipeline._state.__class__.__name__ == f"XAITKApp{app.suffix}"
    # go to next stage
    app.pipeline._state.ready = True  # to avoid bug where non-display pipelines remain in unready state
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the correct page by checking the class name
    assert app.pipeline._state.__class__.__name__ == "ModelEvaluationTestbed"
    final_output = app.pipeline._state.output_test_stages
    assert len(final_output) == 2  # task, xaitk should be present
    assert f"XAITKApp{app.suffix}_0" in final_output.keys()


def test_route_me_config_load_od(json_config_me_od):
    """
    Test route when loading a configuration from a file.
    This should load the configuration and go to the final page.

    ROUTE:
    LandingPage -> ModelEvaluationTestbed
    """
    task = "object_detection"
    workflow = "model_evaluation"
    local = True
    # instantiate the pipeline
    app = FullApp(task=task, local=local, workflow=workflow)
    app.panel()

    # loading data into the file dropper should automatically send us to the final page
    app.pipeline._state.file_dropper.value = {"loaded_config": json.dumps(json_config_me_od)}

    # ensure we actually went to the correct page by checking the class name
    assert app.pipeline._state.__class__.__name__ == "ModelEvaluationTestbed"

    assert len(app.pipeline._state.output_test_stages) == len(json_config_me_od)

    # convert the on-disk formatted configs into instantiated test stages
    app.pipeline._state.load_pipeline(app.pipeline._state.output_test_stages)

    assert len(json_config_me_od) - 1 == len(app.pipeline._state.test_stages)


def test_route_me_config_load_ic(json_config_me_ic):
    """
    Test route when loading a configuration from a file.
    This should load the configuration and go to the final page.

    ROUTE:
    LandingPage -> ModelEvaluationTestbed
    """
    task = "image_classification"
    workflow = "model_evaluation"
    local = True
    # instantiate the pipeline
    app = FullApp(task=task, local=local, workflow=workflow)
    app.panel()

    # loading data into the file dropper should automatically send us to the final page
    app.pipeline._state.file_dropper.value = {"loaded_config": json.dumps(json_config_me_ic)}

    # ensure we actually went to the correct page by checking the class name
    assert app.pipeline._state.__class__.__name__ == "ModelEvaluationTestbed"

    assert len(app.pipeline._state.output_test_stages) == len(json_config_me_ic)

    # convert the on-disk formatted configs into instantiated test stages
    app.pipeline._state.load_pipeline(app.pipeline._state.output_test_stages)

    assert len(json_config_me_ic) - 1 == len(app.pipeline._state.test_stages)


def test_me_workflow_change():
    """Test that changing the workflow from ME to DA works correctly."""
    task = "object_detection"
    workflow = "model_evaluation"
    local = True

    # instantiate the pipeline
    app = FullApp(task=task, local=local, workflow=workflow)
    app.panel()

    assert app.pipeline._next_stage == "MEConfigurationLandingPage"

    app.pipeline._state.me_da_toggle.value = "dataset_analysis"

    assert app.pipeline._next_stage == "DAConfigurationLandingPage"
