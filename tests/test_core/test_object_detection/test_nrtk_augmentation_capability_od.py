from copy import deepcopy

import pytest

from jatic_ri.core.object_detection.nrtk_augmentation_capability import NrtkAugmentation, NrtkAugmentationConfig

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


def test_run_and_collect(fake_od_model_default, fake_od_dataset_default, fake_od_metric_default, test_config):
    capability = NrtkAugmentation()

    copied_metric = deepcopy(fake_od_metric_default)
    copied_metric.metadata["id"] = "fake_metric"

    output = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[copied_metric],
        datasets=[fake_od_dataset_default],
        config=test_config,
    )

    assert output.model_dump()  # smoke test

    assert output.collect_report_consumables(threshold=0.5)  # smoke test

    assert output.collect_md_report(threshold=0.5)  # smoke test
