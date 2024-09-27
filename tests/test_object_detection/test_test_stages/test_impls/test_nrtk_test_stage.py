"""Test NRTKTestStage"""

from jatic_ri.object_detection.test_stages.impls.nrtk_test_stage import (
    NRTKTestStage,
)

ARGS = {
    "name": "NRTKTestStage Example",
    "perturber_factory": {
        "type": "nrtk.impls.perturb_image_factory.generic.step.StepPerturbImageFactory",
        "nrtk.impls.perturb_image_factory.generic.step.StepPerturbImageFactory": {
            "perturber": "nrtk.impls.perturb_image.generic.cv2.blur.GaussianBlurPerturber",
            "theta_key": "ksize",
            "start": 1,
            "stop": 10,
            "step": 1,
        },
    },
}


def test_nrtk_test_stage(dummy_model, dummy_dataset, dummy_metric) -> None:
    """Test NRTKTestStage implementation"""

    test = NRTKTestStage(ARGS)
    # load the maite compliant model
    test.load_model(model=dummy_model, model_id="model_1")
    test.load_metric(metric=dummy_metric, metric_id="metric_1")
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dummy_dataset, dataset_id="dataset_1")
    test.run()
    test.collect_report_consumables()


def test_nrtk_test_stage_no_cache(dummy_model, dummy_dataset, dummy_metric) -> None:
    """Test NRTKTestStage implementation"""

    test = NRTKTestStage(ARGS)
    # load the maite compliant model
    test.load_model(model=dummy_model, model_id="model_1")
    test.load_metric(metric=dummy_metric, metric_id="metric_1")
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dummy_dataset, dataset_id="dataset_1")
    test.run(use_cache=False)
    test.collect_report_consumables()
