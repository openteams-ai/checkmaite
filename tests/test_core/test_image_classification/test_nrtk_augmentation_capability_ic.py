import pytest

from jatic_ri.core.image_classification.nrtk_augmentation_capability import NrtkAugmentation, NrtkAugmentationConfig

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


@pytest.fixture
def test_config():
    return NrtkAugmentationConfig(name=ARGS["name"], perturber_factory=ARGS["perturber_factory"])


def test_run_and_collect(fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default, test_config):
    capability = NrtkAugmentation()

    outputs = capability.run(
        use_cache=False,
        models=[fake_ic_model_default],
        metrics=[fake_ic_metric_default],
        datasets=[fake_ic_dataset_default],
        config=test_config,
    )

    assert outputs.model_dump()  # smoke test

    assert outputs.collect_report_consumables(threshold=0.5)  # smoke test
