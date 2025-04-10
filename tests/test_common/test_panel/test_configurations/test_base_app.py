"""Test base app"""

import pytest

from jatic_ri._common._panel.configurations.base_app import BaseApp


@pytest.mark.parametrize("local", [True, False])
def test_base_app_widgets(local) -> None:
    """This tests the basic functionality provided in the base class"""
    # instantiate the panel app
    base_app = BaseApp(local=local)
    # run through visualization
    # it can't be viewed this way, but it will allow us to catch some errors
    base_app.panel()

    initial_status_text = base_app.status_text
    # click the export button
    #    this calls `export_button_callback` which then calls
    #    `_run_export` and updates the `status_text`
    base_app.export_button.clicks += 1
    # ensure _run_export is called
    assert base_app.__class__.__name__ in base_app.output_test_stages
    # ensure status text changed
    assert base_app.status_text != initial_status_text

    # test the panel stage output
    task, config_dict, local_output = base_app.output()
    assert base_app.output_test_stages == config_dict
    assert base_app.task == task
    assert local == local_output
