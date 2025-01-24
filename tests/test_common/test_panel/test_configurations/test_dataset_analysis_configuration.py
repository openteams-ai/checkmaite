"""Tests for the Dataset Analysis Configuration app for both object detection 
and image classification. 

Object detection includes Reallabel, Survivor, and the Dataeval tools
Image Classification includes Survivor and the Dataeval tools
"""
import json

from io import StringIO
import panel as pn
import pytest

from jatic_ri._common._panel.configurations.dataset_analysis_configuration import DatasetAnalysisConfigApp


def _reset_da_config_app(app: DatasetAnalysisConfigApp):
    """reset everything on the DA config app landing page 
    to false. This protects these tests against changes to default
    behavior. This works for both the IC and OD usecase
    """
    # ensure all the toggles are False
    app.pipeline._state.bias.value = False
    app.pipeline._state.shift.value = False
    app.pipeline._state.linting.value = False
    app.pipeline._state.feasibility.value = False
    app.pipeline._state.show_survivor_config = False
    app.pipeline._state.show_reallabel_config = False
    # clear output_test_stages
    app.pipeline._state.output_test_stages = {}


def test_dataset_analysis_configuration_pipeline():
    """test DA OD configuration pipeline app for basic 
    instantiation and visualization"""
    # instantiate the pipeline
    app = DatasetAnalysisConfigApp()
    # trigger visualization (test even though we can't visualize here)
    app.panel()


def test_da_config_dynamic_stages_final_only():
    """Test the dynamic nature of the pipeline"""
    task = 'object_detection'
    # instantiate the pipeline
    app = DatasetAnalysisConfigApp(task=task)

    # reset the app
    _reset_da_config_app(app)

    # with everything False, the next stage should be "Finalize"
    assert app.pipeline._next_stage == 'Finalize'

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the final page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'FinalPage'

    # click the download button and convert contents back to dict
    string_io_output = app.pipeline._state.writeout_button.callback().read()
    content = json.loads(string_io_output)
    # the only thing in it is the task
    assert len(content) == 1
    assert 'task' in content.keys()
    assert content['task'] == task


def test_da_config_dynamic_stages_bias_only():
    """Test the dynamic nature of the pipeline"""
    task = 'object_detection'
    # instantiate the pipeline
    app = DatasetAnalysisConfigApp(task=task)

    # reset the app
    _reset_da_config_app(app)
    # toggle bias to true
    app.pipeline._state.bias.value = True

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
    assert 'bias' in content.keys()


def test_da_config_dynamic_stages_reallabel_not_survivor_OD():
    """Test the dynamic nature of the pipeline"""
    task = 'object_detection'
    # instantiate the pipeline
    app = DatasetAnalysisConfigApp(task=task)

    # reset the app
    _reset_da_config_app(app)
    # toggle reallabel to true
    app.pipeline._state.show_reallabel_config = True

    # with only reallabel true, the next stage should be reallabel
    assert app.pipeline._next_stage == 'Configure Reallabel'

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the final page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'RealLabelApp'

    # go to next stage
    app.pipeline.next_button.clicks += 1

    # click the download button and convert contents back to dict
    string_io_output = app.pipeline._state.writeout_button.callback().read()
    content = json.loads(string_io_output)
    # should only contain task and reallabel entries
    assert len(content) == 2
    assert 'reallabel_test_stage' in content.keys()


@pytest.mark.parametrize("task", ["object_detection", "image_classification"])
def test_da_config_dynamic_stages_survivor_not_reallabel(task):
    """Test the dynamic nature of the pipeline"""
    # instantiate the pipeline
    app = DatasetAnalysisConfigApp(task=task)

    # reset the app
    _reset_da_config_app(app)
    # toggle survivor to true
    app.pipeline._state.show_survivor_config = True

    # with only survivor true, the next stage should be survivor
    assert app.pipeline._next_stage == 'Configure Survivor'

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the survivor page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'SurvivorApp'

    # go to next stage
    app.pipeline.next_button.clicks += 1

    # click the download button and convert contents back to dict
    string_io_output = app.pipeline._state.writeout_button.callback().read()
    content = json.loads(string_io_output)
    # should only contain task and reallabel entries
    assert len(content) == 2
    assert 'survivor_test_stage' in content.keys()


def test_da_config_dynamic_stages_reallabel_survivor_and_linting():
    """Test the dynamic nature of the pipeline"""
    task = 'object_detection'
    # instantiate the pipeline
    app = DatasetAnalysisConfigApp(task=task)

    # reset the app
    _reset_da_config_app(app)
    # toggle reallabel and survivor to true
    app.pipeline._state.show_reallabel_config = True
    app.pipeline._state.show_survivor_config = True
    app.pipeline._state.linting.value = True

    # with only bias true, the next stage should be 'reallabel'
    assert app.pipeline._next_stage == 'Configure Reallabel'

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the reallabel page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'RealLabelApp'

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the survivor page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'SurvivorApp'

    # go to next stage
    app.pipeline.next_button.clicks += 1
    # ensure we actually went to the final page by checking the class name
    assert app.pipeline._state.__class__.__name__ == 'FinalPage'

    # click the download button and convert contents back to dict
    string_io_output = app.pipeline._state.writeout_button.callback().read()
    content = json.loads(string_io_output)
    # should only contain task and reallabel entries
    assert len(content) == 4
    assert 'reallabel_test_stage' in content.keys()
    assert 'survivor_test_stage' in content.keys()
    assert 'linting' in content.keys()


def test_da_config_dynamic_stages_ic():
    """Test the dynamic nature of the pipeline"""
    task = 'image_classification'
    # instantiate the pipeline
    app = DatasetAnalysisConfigApp(task=task)
    
    # ensure reallabel is not included
    assert 'Configure Reallabel' not in app.pipeline._stages


def test_da_config_dynamic_stages_od():
    """Test the dynamic nature of the pipeline"""
    task = 'object_detection'
    # instantiate the pipeline
    app = DatasetAnalysisConfigApp(task=task)
    
    # ensure reallabel is not included
    assert 'Configure Reallabel' in app.pipeline._stages
