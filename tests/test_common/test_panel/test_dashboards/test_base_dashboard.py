import os
import pytest

from jatic_ri._common._panel.dashboards.base_dashboard import BaseDashboard


@pytest.fixture(scope="session") 
def pipeline_config_od(
    reallabel_config_od, 
    survivor_config_od, 
    nrtk_config_od, 
    xaitk_config_od, 
    feasibility_config_od, 
    bias_config_od, 
    linting_config_od, 
    shift_config_od, 
    baseline_eval_config_od,
    ):
    return {
        'task': 'object_detection',
        'reallabel_config': reallabel_config_od,
        'survivor_config': survivor_config_od, 
        'nrtk_config': nrtk_config_od,
        'xaitk_config': xaitk_config_od, 
        'feasibility_config': feasibility_config_od,
        'bias_config': bias_config_od,
        'linting_config': linting_config_od,
        'shift_config': shift_config_od,
        'baseline_eval_config': baseline_eval_config_od,
    }

@pytest.fixture(scope="session") 
def pipeline_config_ic(
    survivor_config_ic, 
    nrtk_config_ic, 
    xaitk_config_ic, 
    feasibility_config_ic, 
    bias_config_ic, 
    linting_config_ic, 
    shift_config_ic, 
    baseline_eval_config_ic,
    ):
    return {
        'task': 'image_classification',
        'survivor_config': survivor_config_ic, 
        'nrtk_config': nrtk_config_ic,
        'xaitk_config': xaitk_config_ic, 
        'feasibility_config': feasibility_config_ic,
        'bias_config': bias_config_ic,
        'linting_config': linting_config_ic,
        'shift_config': shift_config_ic,
        'baseline_eval_config': baseline_eval_config_ic,
    }


@pytest.mark.parametrize("pipeline_config", ['pipeline_config_od', 'pipeline_config_ic'])
def test_basedashboard_multi_model(pipeline_config, request):
    """Test basic functionality for both OD and IC"""
    config = request.getfixturevalue(pipeline_config)
    task = config['task']

    app = BaseDashboard()
    app.task = task

    # test adding models
    total_models = len(app.model_widgets)
    app.add_model_button_callback(None)
    assert len(app.model_widgets) == total_models + 1

    # ensure multi-model mode is deselected
    app.multi_model_visible = False

    # test loading pipeline from config
    assert app.load_pipeline(configs=config)
    assert "Configuration file loaded" in app.status_text.value
    # the default configs all include multi-model test stages
    assert app.multi_model_visible

    # test mismatch of task
    app.task = 'something else'
    assert not app.load_pipeline(configs=config)
    assert "Mismatch between dashboard type" in app.status_text.value

    # test missing task key in config dict
    del config['task']
    success = app.load_pipeline(configs=config)
    assert not success
    assert "Task must be specified" in app.status_text.value


def test_basedashboard_single_model(nrtk_config_ic):
    """Test triggering of multi-model mode. 
    In the UI, triggering this mode adds an "Add model" button
    to allow users to add another model (i.e. visualizes another
    set of model widgets)
    """
    # task here is arbitrary, but must match in several locations below
    task = 'image_classification'

    config = {
        'task': 'image_classification',
        'nrtk1': nrtk_config_ic,
    }
    app = BaseDashboard()
    app.task = task

    # ensure multi-model mode is deselected
    app.multi_model_visible = False

    # test loading pipeline from config
    assert app.load_pipeline(configs=config)
    assert "Configuration file loaded" in app.status_text.value
    # the default configs all include multi-model test stages
    assert not app.multi_model_visible


def test_basedashboard_output_dir():
    """Test the output dir gets generated"""
    temp_dir = 'deleteme'
    assert not os.path.isdir(temp_dir)
    app = BaseDashboard(output_dir=temp_dir)
    assert os.path.isdir(temp_dir)
    os.rmdir(temp_dir)
