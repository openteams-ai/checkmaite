""" Test XAITKApp"""

import panel as pn

from smqtk_core.configuration import from_config_dict
from xaitk_saliency import GenerateObjectDetectorBlackboxSaliency

from jatic_ri.object_detection._panel.configurations.xaitk_app import XaitkApp
from jatic_ri.object_detection.test_stages.impls.xaitk_test_stage import XAITKTestStage


def test_run_export() -> None:
    """Test calling the XaitkApp's _run_export method"""
    xaitk_app = XaitkApp()
    xaitk_app.export_button.clicks = 1

    assert xaitk_app.status_text == "Configuration saved"
    assert "XaitkApp_0" in xaitk_app.output_test_stages
    test_stage = xaitk_app.output_test_stages["XaitkApp_0"]

    # Check if TYPE and CONFIG keys exist
    assert all(k in test_stage for k in ("TYPE", "CONFIG"))

    # Check if name, saliency_generator and id2label exists
    assert all(k in test_stage["CONFIG"] for k in ("name", "saliency_generator", "id2label"))

    # Assert export config type
    assert test_stage["TYPE"] == "XAITKTestStage"

    # Test saliency generator config
    saliency_generator_config = test_stage["CONFIG"]["saliency_generator"]
    random_grid_stack_impl = "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.RandomGridStack"

    assert saliency_generator_config["type"] == random_grid_stack_impl
    assert (
        all(
            k in saliency_generator_config[random_grid_stack_impl]
            for k in ("n", "s", "p1", "threads", "seed", "fill")
        )
    )
    # Check PlugConfigurable compatibilty
    assert isinstance(
        from_config_dict(
            saliency_generator_config, GenerateObjectDetectorBlackboxSaliency.get_impls()
        ), GenerateObjectDetectorBlackboxSaliency
    )

    # Check output to XAITKTestStage
    xaitk_test_stage = XAITKTestStage(test_stage["CONFIG"])
    assert xaitk_test_stage.stage_name == test_stage["CONFIG"]["name"]
    assert xaitk_test_stage.id2label == test_stage["CONFIG"]["id2label"]


def test_saliency_generation() -> None:
    """
    Test saliency_gen_button_callback that generates
    the saliency map for the given sample image.
    """
    xaitk_app = XaitkApp()

    def _add_saliency_gen_config_widget() -> pn.Column:
        return pn.Column(
            pn.widgets.IntInput(
                name="Number of Masks",
                value=50,
                start=50,
                end=1200,
                step=50,
                stylesheets=[xaitk_app.widget_stylesheet],
            ),
            pn.widgets.Select(
                name="Occlusion Grid Size",
                options=["(7,7)", "(5,5)", "(10,10)"],
                stylesheets=[xaitk_app.widget_stylesheet],
            ),
            pn.pane.Markdown(
                f"""
                <style>
                * {{
                    color: {xaitk_app.color_light_gray};
                }}
                </style>
                [Documentation for XAITK Saliency Generation]
                (https://xaitk-saliency.readthedocs.io/en/latest/implementations.html#end-to-end-saliency-generation)
                """,
            ),
        )
    xaitk_app.saliency_widget = [
        pn.Card(
            _add_saliency_gen_config_widget(),
            title="Saliency Generation Parameters",
            header_color=xaitk_app.color_light_gray,
            width=xaitk_app.left_column_width,
        )
    ]
    xaitk_app.saliency_gen_button.clicks = 1
    assert xaitk_app.status_text == "Saliency generation test completed"
