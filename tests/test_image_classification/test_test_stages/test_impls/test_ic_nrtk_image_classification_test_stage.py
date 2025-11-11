"""Test NRTKTestStage"""

# 3rd party imports
import pandas as pd
import pytest

# Local imports
from jatic_ri.image_classification.test_stages import NRTKTestStage

ARGS = {
    "name": "NRTKTestStage Example",
    "perturber_factory": {
        "type": "nrtk.impls.perturb_image_factory.generic.step.StepPerturbImageFactory",
        "nrtk.impls.perturb_image_factory.generic.step.StepPerturbImageFactory": {
            "perturber": "nrtk.impls.perturb_image.generic.cv2.blur.AverageBlurPerturber",
            "theta_key": "ksize",
            "start": 1,
            "stop": 10,
            "step": 1,
        },
    },
}


@pytest.mark.parametrize("use_stage_cache", [True, False])
def test_nrtk_test_stage(use_stage_cache, dummy_model_od, dummy_dataset_od, dummy_metric_od) -> None:
    """Test NRTKTestStage implementation"""

    test = NRTKTestStage(ARGS)
    run = test.run(
        use_stage_cache=use_stage_cache, models=[dummy_model_od], metrics=[dummy_metric_od], datasets=[dummy_dataset_od]
    )
    output = run.collect_report_consumables(threshold=10)

    example_args = output[0]

    assert all(required_key in example_args for required_key in ("deck", "layout_name", "layout_arguments"))

    assert example_args["layout_name"] == "NRTKEvaluation"
    assert example_args["layout_arguments"]["title"] == test.id
    assert isinstance(example_args["layout_arguments"]["data"], pd.DataFrame)
    assert example_args["layout_arguments"]["x_data_col"] == test.config.perturber_factory.theta_key
    assert example_args["layout_arguments"]["perturbation_type"] == "Average Blur Perturber"
