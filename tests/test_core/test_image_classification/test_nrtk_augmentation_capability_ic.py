import pytest

from jatic_ri.core.image_classification.nrtk_robustness_capability import NrtkRobustness, NrtkRobustnessConfig
from jatic_ri.core.report._gradient import HAS_GRADIENT

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
    return NrtkRobustnessConfig(name=ARGS["name"], perturber_factory=ARGS["perturber_factory"])


@pytest.fixture
def test_run(fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default, test_config):
    capability = NrtkRobustness()

    outputs = capability.run(
        use_cache=False,
        models=[fake_ic_model_default],
        metrics=[fake_ic_metric_default],
        datasets=[fake_ic_dataset_default],
        config=test_config,
    )

    assert outputs.model_dump()  # smoke test

    return outputs


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_run_and_collect_consumables(test_run):
    assert test_run.collect_report_consumables(threshold=0.5)  # smoke test


def test_run_and_collect_md(test_run):
    assert test_run.collect_md_report(threshold=0.5)  # smoke test
