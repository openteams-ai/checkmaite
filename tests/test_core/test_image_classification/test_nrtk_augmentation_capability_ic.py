import pytest

from jatic_ri.core.image_classification.nrtk_robustness_capability import NrtkRobustness, NrtkRobustnessConfig
from jatic_ri.core.report._gradient import HAS_GRADIENT

ARGS = {
    "name": "NRTKTestStage Example",
    "perturber_factory": {
        "type": "nrtk.impls.perturb_image_factory.PerturberStepFactory",
        "nrtk.impls.perturb_image_factory.PerturberStepFactory": {
            "perturber": "nrtk.impls.perturb_image.photometric.blur.AverageBlurPerturber",
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


def test_no_cache_hit_same_perturber_different_step(
    mocker, fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default
):
    """Changing perturber factory step (same perturber class) should produce different
    augmentation IDs and therefore no predict-cache hit."""

    config_a = NrtkRobustnessConfig(
        name="cache_test_a",
        perturber_factory={
            "type": "nrtk.impls.perturb_image_factory.PerturberStepFactory",
            "nrtk.impls.perturb_image_factory.PerturberStepFactory": {
                "perturber": "nrtk.impls.perturb_image.photometric.blur.AverageBlurPerturber",
                "theta_key": "ksize",
                "start": 1,
                "stop": 3,
                "step": 1,
            },
        },
    )

    # Different step *and* non-overlapping range so no individual perturber can cache-hit
    config_b = NrtkRobustnessConfig(
        name="cache_test_b",
        perturber_factory={
            "type": "nrtk.impls.perturb_image_factory.PerturberStepFactory",
            "nrtk.impls.perturb_image_factory.PerturberStepFactory": {
                "perturber": "nrtk.impls.perturb_image.photometric.blur.AverageBlurPerturber",
                "theta_key": "ksize",
                "start": 1,
                "stop": 4,
                "step": 1,
            },
        },
    )

    capability = NrtkRobustness()

    # Run A — populates the predict / evaluate caches
    capability.run(
        use_cache=True,
        models=[fake_ic_model_default],
        metrics=[fake_ic_metric_default],
        datasets=[fake_ic_dataset_default],
        config=config_a,
    )

    # Replace maite.tasks.predict so any real call raises
    mocker.patch(
        "maite.tasks.predict",
        side_effect=RuntimeError("maite.tasks.predict was called — no cache hit"),
    )

    # Run B — perturber hashing should produce different augmentation IDs,
    # so predict() must be invoked (triggering the RuntimeError).
    with pytest.raises(RuntimeError, match="maite.tasks.predict was called"):
        capability.run(
            use_cache=True,
            models=[fake_ic_model_default],
            metrics=[fake_ic_metric_default],
            datasets=[fake_ic_dataset_default],
            config=config_b,
        )

    # Run A again — should hit the cache this time and not raise error
    capability.run(
        use_cache=True,
        models=[fake_ic_model_default],
        metrics=[fake_ic_metric_default],
        datasets=[fake_ic_dataset_default],
        config=config_a,
    )


def test_no_cache_hit_different_perturber(
    mocker, fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default
):
    """Using a completely different perturber class should produce a different
    augmentation ID and therefore no predict-cache hit."""

    config_blur = NrtkRobustnessConfig(
        name="cache_test_blur",
        perturber_factory={
            "type": "nrtk.impls.perturb_image_factory.PerturberStepFactory",
            "nrtk.impls.perturb_image_factory.PerturberStepFactory": {
                "perturber": "nrtk.impls.perturb_image.photometric.blur.AverageBlurPerturber",
                "theta_key": "ksize",
                "start": 1,
                "stop": 3,
                "step": 1,
            },
        },
    )

    config_brightness = NrtkRobustnessConfig(
        name="cache_test_brightness",
        perturber_factory={
            "type": "nrtk.impls.perturb_image_factory.PerturberStepFactory",
            "nrtk.impls.perturb_image_factory.PerturberStepFactory": {
                "perturber": "nrtk.impls.perturb_image.photometric.enhance.BrightnessPerturber",
                "theta_key": "factor",
                "start": 1,
                "stop": 3,
                "step": 1,
            },
        },
    )

    capability = NrtkRobustness()

    # Run with AverageBlurPerturber — populates caches
    capability.run(
        use_cache=True,
        models=[fake_ic_model_default],
        metrics=[fake_ic_metric_default],
        datasets=[fake_ic_dataset_default],
        config=config_blur,
    )

    mocker.patch(
        "maite.tasks.predict",
        side_effect=RuntimeError("maite.tasks.predict was called — no cache hit"),
    )

    # Run with BrightnessPerturber — different class → different augmentation ID
    with pytest.raises(RuntimeError, match="maite.tasks.predict was called"):
        capability.run(
            use_cache=True,
            models=[fake_ic_model_default],
            metrics=[fake_ic_metric_default],
            datasets=[fake_ic_dataset_default],
            config=config_brightness,
        )

    capability.run(
        use_cache=True,
        models=[fake_ic_model_default],
        metrics=[fake_ic_metric_default],
        datasets=[fake_ic_dataset_default],
        config=config_blur,
    )
