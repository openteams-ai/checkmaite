import pytest

from checkmaite.core.image_classification.nrtk_robustness_capability import NrtkRobustness, NrtkRobustnessConfig
from checkmaite.core.report._gradient import HAS_GRADIENT
from tests.report_assertions import assert_inline_markdown_report

ARGS = {
    "name": "NRTKTestStage Example",
    "perturber_factory": {
        "type": "nrtk.impls.perturb_factory.PerturberStepFactory",
        "nrtk.impls.perturb_factory.PerturberStepFactory": {
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
    return NrtkRobustnessConfig(
        metric_key="fake_metric", name=ARGS["name"], perturber_factory=ARGS["perturber_factory"]
    )


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


@pytest.mark.optional
@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_run_and_collect_consumables(test_run):
    assert test_run.collect_report_consumables(threshold=0.5)  # smoke test


def test_run_and_collect_md(test_run):
    report = test_run.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=test_run.capability_id)


def test_no_cache_hit_same_perturber_different_step(
    mocker, fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default
):
    """Changing perturber factory step (same perturber class) should produce different
    augmentation IDs and therefore no predict-cache hit."""

    config_a = NrtkRobustnessConfig(
        metric_key="fake_metric",
        name="cache_test_a",
        perturber_factory={
            "type": "nrtk.impls.perturb_factory.PerturberStepFactory",
            "nrtk.impls.perturb_factory.PerturberStepFactory": {
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
        metric_key="fake_metric",
        name="cache_test_b",
        perturber_factory={
            "type": "nrtk.impls.perturb_factory.PerturberStepFactory",
            "nrtk.impls.perturb_factory.PerturberStepFactory": {
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

    # Replace the fundamental MAITE task so any real call raises.
    mocker.patch(
        "maite.tasks.evaluate",
        side_effect=RuntimeError("maite.tasks.evaluate was called — no cache hit"),
    )

    # Run B — perturber hashing should produce different augmentation IDs,
    # so evaluate() must execute (triggering the RuntimeError).
    with pytest.raises(RuntimeError, match="maite.tasks.evaluate was called"):
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
        metric_key="fake_metric",
        name="cache_test_blur",
        perturber_factory={
            "type": "nrtk.impls.perturb_factory.PerturberStepFactory",
            "nrtk.impls.perturb_factory.PerturberStepFactory": {
                "perturber": "nrtk.impls.perturb_image.photometric.blur.AverageBlurPerturber",
                "theta_key": "ksize",
                "start": 1,
                "stop": 3,
                "step": 1,
            },
        },
    )

    config_brightness = NrtkRobustnessConfig(
        metric_key="fake_metric",
        name="cache_test_brightness",
        perturber_factory={
            "type": "nrtk.impls.perturb_factory.PerturberStepFactory",
            "nrtk.impls.perturb_factory.PerturberStepFactory": {
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
        "maite.tasks.evaluate",
        side_effect=RuntimeError("maite.tasks.evaluate was called — no cache hit"),
    )

    # Run with BrightnessPerturber — different class → different augmentation ID
    with pytest.raises(RuntimeError, match="maite.tasks.evaluate was called"):
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


def test_step_factory_config_end_to_end(
    tmp_path, mocker, fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default
):
    """A step factory config runs, is reloaded from the cache, and gives a report and analytics rows."""
    from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend

    factory_config = {
        "type": "nrtk.impls.perturb_factory.PerturberStepFactory",
        "nrtk.impls.perturb_factory.PerturberStepFactory": {
            "perturber": "nrtk.impls.perturb_image.photometric.blur.AverageBlurPerturber",
            "theta_key": "ksize",
            "start": 1,
            "stop": 4,
            "step": 1,
        },
    }
    config = NrtkRobustnessConfig(metric_key="fake_metric", name="step_config", perturber_factory=factory_config)
    inputs = {
        "models": [fake_ic_model_default],
        "metrics": [fake_ic_metric_default],
        "datasets": [fake_ic_dataset_default],
        "config": config,
    }

    capability = NrtkRobustness()
    run = capability.run(**inputs)
    assert [point.theta for point in run.outputs.points] == [1.0, 2.0, 3.0]
    assert run.outputs.perturber == "AverageBlurPerturber"
    assert run.outputs.theta_key == "ksize"

    # The second run must come from the cache, with the factory restored from it.
    spy = mocker.spy(capability, "_run")
    cached = capability.run(**inputs)
    spy.assert_not_called()
    assert cached.run_uid == run.run_uid
    assert list(cached.config.perturber_factory.thetas) == [1, 2, 3]
    assert cached.config.perturber_factory.get_config()["theta_key"] == "ksize"
    assert cached.outputs.model_dump() == run.outputs.model_dump()

    report = cached.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=cached.capability_id)
    assert "ksize" in report.content

    store = AnalyticsStore(ParquetBackend(str(tmp_path / "store")))
    store.write([cached])
    rows = store.query_sql("SELECT theta_key, theta_value FROM nrtk_robustness ORDER BY theta_index")
    assert set(rows["theta_key"]) == {"ksize"}
    assert sorted(set(rows["theta_value"])) == [1.0, 2.0, 3.0]
