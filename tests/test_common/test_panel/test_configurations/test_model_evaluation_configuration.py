"""Test suite for Model Evaluation Configuration app.

The differences between the object detection and image classification tasks 
is limited to the importing of the xaitk and nrtk individual apps. For this 
reason, IC is only tested in the case where all apps are included. 
"""
from io import StringIO
import json

import panel as pn
import pytest

from jatic_ri._common._panel.configurations.model_evaluation_configuration import ModelEvaluationConfigApp


def _reset_me_config_app(app: ModelEvaluationConfigApp):
    """reset everything on the ME config app landing page 
    to false. This protects these tests against changes to default
    behavior.
    """
    # ensure all the toggles are False
    app.pipeline._state.baseline_eval.value = False
    app.pipeline._state.show_xaitk_config = False
    app.pipeline._state.show_nrtk_config = False
    # clear output_test_stages
    app.pipeline._state.output_test_stages = {}


def test_model_evaluation_configuration_pipeline():
    """test ME OD configuration pipeline app for basic 
    instantiation and visualization"""
    # instantiate the pipeline
    app = ModelEvaluationConfigApp()
    # trigger visualization (test even though we can't visualize here)
    app.panel()


@pytest.mark.parametrize("local", [True, False])
def test_me_config_dynamic_stages_local(local):
    """Ensure local setting makes it to the final page
    No selections means it goes straight to finalize
    """
    task = 'object_detection'
    # instantiate the pipeline
    app = ModelEvaluationConfigApp(task=task, local=local)

    # reset the app
    _reset_me_config_app(app)

    # with everything False, the next stage should be "Finalize"
    assert app.pipeline._next_stage == 'Finalize'

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the final page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'FinalPage'

    final_output = app.pipeline._state.output_test_stages
    assert len(final_output) == 1
    assert 'task' in final_output.keys()
    assert final_output['task'] == task
    assert app.pipeline._state.local == local


def test_me_config_dynamic_stages_baseline_evaluate_only():
    """Test the dynamic nature of the pipeline"""
    task = 'object_detection'
    # instantiate the pipeline
    app = ModelEvaluationConfigApp(task=task)

    # reset the app
    _reset_me_config_app(app)
    # toggle baseline_evaluate to true
    app.pipeline._state.baseline_eval.value = True

    # with only bias true, the next stage should be "Finalize"
    assert app.pipeline._next_stage == 'Finalize'

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the final page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'FinalPage'

    # click the download button and convert contents back to dict
    string_io_output = app.pipeline._state.writeout_button.callback().read()
    content = json.loads(string_io_output)
    # should only contain task and bias entries
    assert len(content) == 2
    assert 'baseline_evaluate' in content.keys()


def test_me_config_dynamic_stages_nrtk_not_xaitk():
    """Test the dynamic nature of the pipeline"""
    task = 'object_detection'
    # instantiate the pipeline
    app = ModelEvaluationConfigApp(task=task)

    # reset the app
    _reset_me_config_app(app)
    # toggle nrtk to true
    app.pipeline._state.show_nrtk_config = True

    # with only nrtk true, the next stage should be nrtk
    assert app.pipeline._next_stage == 'Configure NRTK'

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the final page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'NRTKApp'

    # nrtk app requires explicitly adding a configuration
    app.pipeline._state.add_button.clicks += 1

    # go to next stage
    app.pipeline.next_button.clicks += 1

    # click the download button and convert contents back to dict
    string_io_output = app.pipeline._state.writeout_button.callback().read()
    content = json.loads(string_io_output)
    # should only contain task and nrtk entries
    assert len(content) == 2
    assert 'NRTKApp_0' in content.keys()


def test_me_config_dynamic_stages_xrtk_not_nrtk():
    """Test the dynamic nature of the pipeline"""
    task = 'object_detection'
    # instantiate the pipeline
    app = ModelEvaluationConfigApp(task=task)

    # reset the app
    _reset_me_config_app(app)
    # toggle xaitk to true
    app.pipeline._state.show_xaitk_config = True

    # with only xaitk true, the next stage should be xaitk
    assert app.pipeline._next_stage == 'Configure XAITK'

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the xaitk page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'XAITKApp'

    # go to next stage
    app.pipeline.next_button.clicks += 1

    # click the download button and convert contents back to dict
    string_io_output = app.pipeline._state.writeout_button.callback().read()
    content = json.loads(string_io_output)
    # should only contain task and xaitk entries
    assert len(content) == 2
    assert 'XAITKApp_0' in content.keys()


@pytest.mark.parametrize("task", ["object_detection", "image_classification"])
def test_me_config_dynamic_stages_nrtk_xaitk_and_baseline(task):
    """Test the dynamic nature of the pipeline for both OD and IC"""
    # instantiate the pipeline
    app = ModelEvaluationConfigApp(task=task)

    # reset the app
    _reset_me_config_app(app)
    # toggle nrtk and xaitk to true
    app.pipeline._state.show_nrtk_config = True
    app.pipeline._state.show_xaitk_config = True
    app.pipeline._state.baseline_eval.value = True

    # with only bias true, the next stage should be 'nrtk'
    assert app.pipeline._next_stage == 'Configure NRTK'

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the nrtk page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'NRTKApp'

    # nrtk app requires explicitly adding a configuration
    app.pipeline._state.add_button.clicks += 1

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the xaitk page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'XAITKApp'

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the final page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'FinalPage'

    # click the download button and convert contents back to dict
    string_io_output = app.pipeline._state.writeout_button.callback().read()
    content = json.loads(string_io_output)

    assert len(content) == 4
    assert 'NRTKApp_0' in content.keys()
    assert 'XAITKApp_0' in content.keys()
    assert 'baseline_evaluate' in content.keys()
