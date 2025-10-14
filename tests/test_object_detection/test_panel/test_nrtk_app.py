"""Test base app"""

import pytest
from nrtk.impls.perturb_image.generic.PIL.enhance import BrightnessPerturber
from nrtk.impls.perturb_image.pybsm.pybsm_perturber import PybsmPerturber
from nrtk.impls.perturb_image_factory.generic.linspace_step import LinSpacePerturbImageFactory
from nrtk.impls.perturb_image_factory.generic.one_step import OneStepPerturbImageFactory
from nrtk.impls.perturb_image_factory.generic.step import StepPerturbImageFactory
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory

from jatic_ri.object_detection._panel.configurations.nrtk_app import NRTKAppOD
from jatic_ri.object_detection.test_stages.impls.nrtk_test_stage import NRTKTestStage


@pytest.mark.parametrize(
    ("perturber_type", "perturber_factory_type", "factory_args"),
    [
        (BrightnessPerturber, OneStepPerturbImageFactory, {"theta_key": "factor", "theta_value": 10.0}),
        (
            BrightnessPerturber,
            StepPerturbImageFactory,
            {"theta_key": "factor", "start": 1.0, "stop": 30.0, "step": 2.0, "to_int": True},
        ),
        (
            BrightnessPerturber,
            LinSpacePerturbImageFactory,
            {"theta_key": "factor", "start": 0.0, "stop": 30.0, "step": 3},
        ),
        (
            PybsmPerturber,
            CustomPybsmPerturbImageFactory,
            {"theta_keys": ["f", "D"], "thetas": [[0.014, 0.012], [0.001, 0.003]]},
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:invalid value encountered in (arccos|sqrt):RuntimeWarning")
def test_base_app_widgets(perturber_type, perturber_factory_type, factory_args) -> None:
    """This tests the basic functionality provided in the base class"""
    # instantiate the panel app
    nrtk_app = NRTKAppOD()
    # run through visualization
    # it can't be viewed this way, but it will allow us to catch some errors
    nrtk_app.panel()

    initial_status_text = nrtk_app.status_source.current_value

    nrtk_app._run_export()
    # ensure _run_export is called
    assert len(nrtk_app.output_test_stages) == 0
    # ensure status text changed
    assert nrtk_app.status_source.current_value != initial_status_text

    # test the panel stage output
    _, config_output, _ = nrtk_app.output()
    assert nrtk_app.output_test_stages == config_output

    # setup internals used in testing
    nrtk_app.perturber_select.value = perturber_type
    nrtk_app.factory_selector.value = perturber_factory_type
    nrtk_app.name_input.value = "TestFactory"

    if perturber_factory_type == OneStepPerturbImageFactory:
        nrtk_app.theta_key.value = factory_args["theta_key"]
        nrtk_app.theta_value.value = factory_args["theta_value"]
    elif perturber_factory_type == StepPerturbImageFactory:
        nrtk_app.theta_key.value = factory_args["theta_key"]
        nrtk_app.start.value = factory_args["start"]
        nrtk_app.stop.value = factory_args["stop"]
        nrtk_app.step.value = factory_args["step"]
        nrtk_app.to_int.value = factory_args["to_int"]
    elif perturber_factory_type == LinSpacePerturbImageFactory:
        nrtk_app.theta_key.value = factory_args["theta_key"]
        nrtk_app.start.value = factory_args["start"]
        nrtk_app.stop.value = factory_args["stop"]
        nrtk_app.step.value = factory_args["step"]
    elif perturber_factory_type == CustomPybsmPerturbImageFactory:
        nrtk_app.theta_keys_input.value = factory_args["theta_keys"]
        nrtk_app.thetas_input.value = factory_args["thetas"]

    # test factory config is built correctly
    factory_json = nrtk_app.build_factory_json()
    factory_type_string = f"{perturber_factory_type.__module__}.{perturber_factory_type.__name__}"
    assert factory_json["type"] == factory_type_string
    if perturber_factory_type == OneStepPerturbImageFactory:
        assert factory_json[factory_type_string]["theta_key"] == factory_args["theta_key"]
        assert factory_json[factory_type_string]["theta_value"] == factory_args["theta_value"]
    elif perturber_factory_type == CustomPybsmPerturbImageFactory:
        assert factory_json[factory_type_string]["theta_keys"] == factory_args["theta_keys"]
        assert factory_json[factory_type_string]["thetas"] == factory_args["thetas"]

    # test _run_export
    nrtk_app.add_test_stage_callback(None)
    nrtk_app._run_export()
    assert len(nrtk_app.output_test_stages) == 1
    test_stage = nrtk_app.output_test_stages["NRTKAppOD_0"]
    assert test_stage["TYPE"] == "NRTKTestStage"
    assert test_stage["CONFIG"]["name"] == "natural_robustness_TestFactory"
    assert test_stage["CONFIG"]["perturber_factory"] == factory_json

    # test output to NRTKTestStage
    nrtk_test_stage = NRTKTestStage(test_stage["CONFIG"])
    assert nrtk_test_stage.config.name == test_stage["CONFIG"]["name"]

    # test test_perturber_button_callback
    nrtk_app.test_perturber_button_callback(None)
    assert nrtk_app.status_source.current_value == "Finished Perturbing"

    # test clear_test_stage_callback
    nrtk_app.clear_test_stage_callback(None)
    nrtk_app._run_export()
    assert len(nrtk_app.output_test_stages) == 1
    assert len(nrtk_app.test_stages) == 0
