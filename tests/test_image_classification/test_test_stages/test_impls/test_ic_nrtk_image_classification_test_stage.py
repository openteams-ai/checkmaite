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
    # load the maite compliant model
    test.load_model(model=dummy_model_od, model_id="model_1")
    test.load_metric(metric=dummy_metric_od, metric_id=dummy_metric_od.return_key)
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dummy_dataset_od, dataset_id="dataset_1")
    test.run(use_stage_cache=use_stage_cache)
    output = test.collect_report_consumables()

    example_args = output[0]

    assert all(required_key in example_args for required_key in ("deck", "layout_name", "layout_arguments"))

    assert example_args["layout_name"] == "NRTKEvaluation"
    assert example_args["layout_arguments"]["title"] == test.name
    assert isinstance(example_args["layout_arguments"]["data"], pd.DataFrame)
    assert example_args["layout_arguments"]["x_data_col"] == test.config.perturber_factory.theta_key
    assert example_args["layout_arguments"]["y_data_col"] == test.metric_id
    assert example_args["layout_arguments"]["perturbation_type"] == "Average Blur Perturber"
    assert example_args["layout_arguments"]["model"] == test.model_id
