"""Test heart_app"""

from jatic_ri.image_classification._panel.configurations.heart_ic_app import HeartICApp


def test_add_test_stage_to_json() -> None:
    # run through visualization even though it can't be seen this way
    heart_ic_app: HeartICApp = HeartICApp()
    heart_ic_app.panel()

    # select widget values
    heart_ic_app.patch_attack_config = True
    heart_ic_app.strong_attack_config = True

    # trigger adding the settings as a stored test stage
    heart_ic_app.add_test_stage_callback(None)

    assert len(heart_ic_app.attack_stages) == 1
    assert len(heart_ic_app.finished_factory_display) == 1
    assert heart_ic_app.attack_stages[0]["CONFIG"]["attack_type"] == "Patch Attack"
    assert heart_ic_app.attack_stages[0]["CONFIG"]["parameters"] == "Stronger Attack"

    # click the export button
    #    this calls `export_button_callback` which then calls
    #    `_run_export` and updates the `status_text`
    starting_stages = len(heart_ic_app.output_test_stages)
    heart_ic_app._run_export()
    # ensure _run_export is called
    assert len(heart_ic_app.output_test_stages) != starting_stages
    assert heart_ic_app.output_test_stages["heart-0"]["CONFIG"]["attack_type"] == "Patch Attack"
    assert heart_ic_app.output_test_stages["heart-0"]["CONFIG"]["parameters"] == "Stronger Attack"


def test_common_app_model_recognition() -> None:
    """Test the model recognition function of the heart base app"""
    heart_ic_app: HeartICApp = HeartICApp()
    heart_ic_app.panel()

    assert heart_ic_app.task == "image_classification"
