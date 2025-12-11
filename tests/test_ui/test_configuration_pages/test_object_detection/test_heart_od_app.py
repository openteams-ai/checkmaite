import pytest

from jatic_ri.ui.configuration_pages.object_detection.heart_od_app import HeartODApp


@pytest.mark.unsupported
def test_add_test_stage_to_json() -> None:
    # run through visualization even though it can't be seen this way
    heart_od_app: HeartODApp = HeartODApp()
    heart_od_app.panel()

    # select widget values
    heart_od_app.patch_attack_config = True
    heart_od_app.strong_attack_config = True

    # trigger adding the settings as a stored test stage
    heart_od_app.add_test_stage_callback(None)

    assert len(heart_od_app.attack_stages) == 1
    assert len(heart_od_app.finished_factory_display) == 1
    assert heart_od_app.attack_stages[0]["CONFIG"]["attack_type"] == "Patch Attack"
    assert heart_od_app.attack_stages[0]["CONFIG"]["parameters"] == "Stronger Attack"

    # click the export button
    #    this calls `export_button_callback` which then calls
    #    `_run_export` and updates the `status_text`
    starting_stages = len(heart_od_app.output_test_stages)
    heart_od_app._run_export()
    # ensure _run_export is called
    assert len(heart_od_app.output_test_stages) != starting_stages
    assert heart_od_app.output_test_stages["heart-0"]["CONFIG"]["attack_type"] == "Patch Attack"
    assert heart_od_app.output_test_stages["heart-0"]["CONFIG"]["parameters"] == "Stronger Attack"


@pytest.mark.unsupported
def test_common_app_model_recognition() -> None:
    """Test the model recognition function of the heart base app"""
    heart_od_app: HeartODApp = HeartODApp()
    heart_od_app.panel()

    assert heart_od_app.task == "object_detection"
