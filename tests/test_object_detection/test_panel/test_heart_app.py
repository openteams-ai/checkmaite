"""Test heart_app"""

import pytest

from jatic_ri.object_detection._panel.configurations.heart_app import HeartApp


@pytest.fixture(scope="function")  # noqa: PT003
def heart_app() -> HeartApp:
    return HeartApp()


def test_add_test_stage_to_json(heart_app) -> None:
    # run through visualization even though it can't be seen this way
    heart_app.panel()
    # select widget values
    heart_app.attack_select.value = "Patch Attack"
    heart_app.parameter_select.value = "Stronger Attack"

    # trigger adding the settings as a stored test stage
    heart_app.add_test_stage_callback(None)

    assert len(heart_app.test_stages) == 1
    assert len(heart_app.finished_factory_display) == 1
    assert heart_app.test_stages[0]["CONFIG"]["attack_type"] == "Patch Attack"
    assert heart_app.test_stages[0]["CONFIG"]["parameters"] == "Stronger Attack"

    # click the export button
    #    this calls `export_button_callback` which then calls
    #    `_run_export` and updates the `status_text`
    starting_stages = len(heart_app.output_test_stages)
    heart_app.export_button.clicks += 1
    # ensure _run_export is called
    assert len(heart_app.output_test_stages) != starting_stages


def test_clear_test_stage_callback(heart_app) -> None:
    """Test the callback to clear the test stages from the app"""
    # run through visualization even though it can't be seen this way
    heart_app.panel()
    # mock existing settings
    heart_app.finished_factory_display = ["config1", "config2"]
    heart_app.test_stages = ["stage1", "stage2"]
    # trigger callback to clear the test stages
    heart_app.clear_test_stage_callback(None)

    assert heart_app.finished_factory_display == []
    assert heart_app.test_stages == []


"""
TODO: Once the method for caching the benign and attacked image has been finalized,
include a test case for the test_attack_button_callback method.
"""
