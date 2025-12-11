import json

from smqtk_core.configuration import from_config_dict
from xaitk_saliency.interfaces.gen_object_detector_blackbox_sal import GenerateObjectDetectorBlackboxSaliency

from jatic_ri.ui.configuration_pages.object_detection.xaitk_app import XAITKAppOD


def test_run_export() -> None:
    """Test calling the XAITKApp's _run_export method with default DRISE stack"""
    xaitk_app = XAITKAppOD()
    # run through visualization
    # it can't be viewed this way, but it will allow us to catch some errors
    xaitk_app.panel()

    xaitk_app._run_export()

    assert "XAITKAppOD_0" in xaitk_app.output_test_stages
    test_stage = xaitk_app.output_test_stages["XAITKAppOD_0"]

    # Check if TYPE and CONFIG keys exist
    assert all(k in test_stage for k in ("TYPE", "CONFIG"))

    # Check if name, saliency_generator and id2label exists
    assert all(k in test_stage["CONFIG"] for k in ("name", "saliency_generator"))

    # Assert export config type
    assert test_stage["TYPE"] == "XAITKTestStage"

    # Test saliency generator config
    saliency_generator_config = test_stage["CONFIG"]["saliency_generator"]
    drise_stack_impl = "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack"

    assert saliency_generator_config["type"] == drise_stack_impl
    assert all(k in saliency_generator_config[drise_stack_impl] for k in ("n", "s", "p1", "threads", "seed", "fill"))

    # DRISEStack expects a single value for size
    assert type(saliency_generator_config[drise_stack_impl]["s"]) is int

    # Check PlugConfigurable compatibilty
    assert isinstance(
        from_config_dict(saliency_generator_config, GenerateObjectDetectorBlackboxSaliency.get_impls()),
        GenerateObjectDetectorBlackboxSaliency,
    )

    # Check output to XAITKTestStage
    json.dumps(xaitk_app.output_test_stages)


def test_saliency_generation() -> None:
    """
    Test saliency_gen_button_callback that generates
    the saliency map for the given sample image.
    """
    xaitk_app = XAITKAppOD()
    # run through visualization
    # it can't be viewed this way, but it will allow us to catch some errors
    xaitk_app.panel()

    # reduce number of masks to 1 to save computation time
    xaitk_app.saliency_widget[0].objects[0].objects[0].value = 1

    xaitk_app.saliency_gen_button.clicks = 1
    assert xaitk_app.status_source.current_value == "Saliency generation test completed"


def test_random_grid_export() -> None:
    """Test parameters when setting stack to RandomGrid the XAITKApp's _run_export method"""
    xaitk_app = XAITKAppOD()
    xaitk_app.panel()
    xaitk_app.stack_select.value = "RandomGrid"
    xaitk_app._run_export()

    test_stage = xaitk_app.output_test_stages["XAITKAppOD_0"]
    # Test saliency generator config
    saliency_generator_config = test_stage["CONFIG"]["saliency_generator"]
    random_grid_stack_impl = "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.RandomGridStack"

    assert saliency_generator_config["type"] == random_grid_stack_impl
    assert all(
        k in saliency_generator_config[random_grid_stack_impl] for k in ("n", "s", "p1", "threads", "seed", "fill")
    )

    # RandomGridStack expects a list of 2 parammeters (dimensions of a square)
    assert len(saliency_generator_config[random_grid_stack_impl]["s"]) == 2
    # Check PlugConfigurable compatibilty
    assert isinstance(
        from_config_dict(saliency_generator_config, GenerateObjectDetectorBlackboxSaliency.get_impls()),
        GenerateObjectDetectorBlackboxSaliency,
    )
