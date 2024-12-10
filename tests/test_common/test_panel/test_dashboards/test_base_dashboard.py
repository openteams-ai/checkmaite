import os
import pytest

from jatic_ri._common._panel.dashboards.base_dashboard import BaseDashboard


def test_basedashboard_od(
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
    """Test basic functionality"""
    app = BaseDashboard()

    # test adding models
    total_models = len(app.model_widgets)
    app.add_model_button_callback(None)
    assert len(app.model_widgets) == total_models + 1

    # test loading pipeline from config
    pipeline_config = {
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
    # test missing task in config
    success = app.load_pipeline(configs=pipeline_config)
    assert not success
    assert "Task must be specified" in app.status_text

    # test mismatch of task
    app.task = 'something else'
    pipeline_config['task'] = 'object_detection'
    assert not app.load_pipeline(configs=pipeline_config)
    assert "Mismatch between dashboard type" in app.status_text

    app.task = "object_detection"
    assert app.load_pipeline(configs=pipeline_config)
    assert "Configuration file loaded" in app.status_text


def test_basedashboard_output_dir():
    """Test the output dir gets generated"""
    temp_dir = 'deleteme'
    assert not os.path.isdir(temp_dir)
    app = BaseDashboard(output_dir=temp_dir)
    assert os.path.isdir(temp_dir)
    os.rmdir(temp_dir)
