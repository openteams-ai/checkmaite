""" Test IC XAITKApp"""

import panel as pn
import json

from smqtk_core.configuration import from_config_dict
from xaitk_saliency import GenerateImageClassifierBlackboxSaliency

from jatic_ri.image_classification._panel.configurations.xaitk_app import XAITKApp
from jatic_ri.image_classification.test_stages.impls.xaitk_test_stage import XAITKTestStage


def test_run_export_rise() -> None:
    """Test calling the XAITKApp's _run_export method"""
    xaitk_app = XAITKApp()
    # run through visualization
    # it can't be viewed this way, but it will allow us to catch some errors
    xaitk_app.panel()
    xaitk_app.stack_select.value = "RISE"
    xaitk_app.export_button.clicks = 1

    assert xaitk_app.status_text == "Configuration saved"
    assert "XAITKApp_0" in xaitk_app.output_test_stages
    test_stage = xaitk_app.output_test_stages["XAITKApp_0"]

    # Check if TYPE and CONFIG keys exist
    assert all(k in test_stage for k in ("TYPE", "CONFIG"))

    # Check if name, saliency_generator and id2label exists
    assert all(k in test_stage["CONFIG"] for k in ("name", "saliency_generator"))

    # Assert export config type
    assert test_stage["TYPE"] == "XAITKTestStage"

    # Test saliency generator config
    saliency_generator_config = test_stage["CONFIG"]["saliency_generator"]
    random_grid_stack_impl = "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise.RISEStack"

    assert saliency_generator_config["type"] == random_grid_stack_impl
    assert (
        all(
            k in saliency_generator_config[random_grid_stack_impl]
            for k in ("n", "s", "p1", "threads", "seed", "debiased")
        )
    )
    # Check PlugConfigurable compatibilty
    assert isinstance(
        from_config_dict(
            saliency_generator_config, GenerateImageClassifierBlackboxSaliency.get_impls()
        ), GenerateImageClassifierBlackboxSaliency
    )

    # Check output to XAITKTestStage
    json.dumps(xaitk_app.output_test_stages)
    xaitk_test_stage = XAITKTestStage(test_stage["CONFIG"])
    assert xaitk_test_stage.stage_name == test_stage["CONFIG"]["name"]

def test_run_export_mc_rise() -> None:
    """Test calling the XAITKApp's _run_export method"""
    xaitk_app = XAITKApp()
    # run through visualization
    # it can't be viewed this way, but it will allow us to catch some errors
    xaitk_app.panel()
    xaitk_app.stack_select.value = "MC-RISE"
    xaitk_app.export_button.clicks = 1

    assert xaitk_app.status_text == "Configuration saved"
    assert "XAITKApp_0" in xaitk_app.output_test_stages
    test_stage = xaitk_app.output_test_stages["XAITKApp_0"]

    # Check if TYPE and CONFIG keys exist
    assert all(k in test_stage for k in ("TYPE", "CONFIG"))

    # Check if name, saliency_generator and id2label exists
    assert all(k in test_stage["CONFIG"] for k in ("name", "saliency_generator"))

    # Assert export config type
    assert test_stage["TYPE"] == "XAITKTestStage"

    # Test saliency generator config
    saliency_generator_config = test_stage["CONFIG"]["saliency_generator"]
    mc_rise_stack_impl = "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.mc_rise.MCRISEStack"

    assert saliency_generator_config["type"] == mc_rise_stack_impl
    assert (
        all(
            k in saliency_generator_config[mc_rise_stack_impl]
            for k in ("n", "s", "p1", "threads", "seed", "fill_colors")
        )
    )
    # Check PlugConfigurable compatibilty
    assert isinstance(
        from_config_dict(
            saliency_generator_config, GenerateImageClassifierBlackboxSaliency.get_impls()
        ), GenerateImageClassifierBlackboxSaliency
    )

    # Check output to XAITKTestStage
    json.dumps(xaitk_app.output_test_stages)
    xaitk_test_stage = XAITKTestStage(test_stage["CONFIG"])
    assert xaitk_test_stage.stage_name == test_stage["CONFIG"]["name"]


def test_saliency_generation_rise() -> None:
    """
    Test saliency_gen_button_callback that generates
    the saliency map for the given sample image.
    """
    xaitk_app = XAITKApp()
    # run through visualization
    # it can't be viewed this way, but it will allow us to catch some errors
    xaitk_app.panel()
    xaitk_app.stack_select.value = "RISE"

    # reduce number of masks to 1 to save computation time
    xaitk_app.saliency_widget[0].objects[0].objects[0].value = 1

    xaitk_app.saliency_gen_button.clicks = 1
    assert xaitk_app.status_text == "Saliency generation test completed"

def test_saliency_generation_mc_rise() -> None:
    """
    Test saliency_gen_button_callback that generates
    the saliency map for the given sample image.
    """
    xaitk_app = XAITKApp()
    # run through visualization
    # it can't be viewed this way, but it will allow us to catch some errors
    xaitk_app.panel()
    xaitk_app.stack_select.value = "MC-RISE"

    # reduce number of masks to 1 to save computation time
    xaitk_app.saliency_widget[0].objects[0].objects[0].value = 1

    xaitk_app.saliency_gen_button.clicks = 1
    assert xaitk_app.status_text == "Saliency generation test completed"
