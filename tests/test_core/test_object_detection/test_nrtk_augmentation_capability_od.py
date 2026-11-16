from copy import deepcopy

import numpy as np
import pytest
import torch

from checkmaite.core.object_detection.nrtk_robustness_capability import NrtkRobustness, NrtkRobustnessConfig
from checkmaite.core.report._gradient import HAS_GRADIENT
from tests.conftest import FakeODMetric
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
def test_run(fake_od_model_default, fake_od_dataset_default, fake_od_metric_default, test_config):
    capability = NrtkRobustness()

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

    return output


@pytest.mark.optional
@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_run_and_collect_consumables(test_run):
    assert test_run.collect_report_consumables(threshold=0.5)  # smoke test


def test_run_and_collect_md(test_run):
    report = test_run.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=test_run.capability_id)


def test_no_cache_hit_same_perturber_different_step(
    mocker, fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
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
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
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
            models=[fake_od_model_default],
            metrics=[fake_od_metric_default],
            datasets=[fake_od_dataset_default],
            config=config_b,
        )

    # Run A again — should hit the cache this time and not raise error
    capability.run(
        use_cache=True,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
        config=config_a,
    )


def test_no_cache_hit_different_perturber(
    mocker, fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
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
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
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
            models=[fake_od_model_default],
            metrics=[fake_od_metric_default],
            datasets=[fake_od_dataset_default],
            config=config_brightness,
        )

    # Run with AverageBlurPerturber again — should hit cache and not raise
    capability.run(
        use_cache=True,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
        config=config_blur,
    )


def test_augment_id_changes_with_exact_nrtk_version(mocker):
    """A different NRTK version, even a patch release, gives a different augment_id,
    so predictions on images perturbed by another NRTK version are not reused."""
    from nrtk.impls.perturb_image.photometric.enhance import BrightnessPerturber

    from checkmaite.core._common import nrtk_robustness_capability as common

    perturber = BrightnessPerturber(factor=0.5)
    versions = {"nrtk": "1.1.0", "pybsm": "0.13.0"}
    mocker.patch.object(common.importlib.metadata, "version", side_effect=versions.__getitem__)
    first = common._augment_id(perturber)
    assert common._augment_id(perturber) == first
    versions["nrtk"] = "1.1.1"
    assert common._augment_id(perturber) != first


def test_augment_id_is_stable_for_pybsm_perturbers():
    """pyBSM perturbers hold simulator objects that print with their memory address;
    the ID must not depend on them, or the cache never hits."""
    pytest.importorskip("pybsm")
    from nrtk.impls.perturb_image.optical import PybsmPerturber

    from checkmaite.core._common import nrtk_robustness_capability as common

    assert common._augment_id(PybsmPerturber()) == common._augment_id(PybsmPerturber())
    assert common._augment_id(PybsmPerturber(altitude=9000)) != common._augment_id(PybsmPerturber(altitude=5000))


def test_augment_id_hashes_arrays_exactly_and_rejects_unknown_values():
    from checkmaite.core._common import nrtk_robustness_capability as common

    class _Perturber:
        def __init__(self, config):
            self.config = config

        def get_config(self):
            return self.config

    big = np.zeros(2000)
    changed = big.copy()
    changed[1000] = 1  # hidden in the middle of a truncated repr
    assert common._augment_id(_Perturber({"fill": big})) != common._augment_id(_Perturber({"fill": changed}))
    with pytest.raises(TypeError, match="Cannot build a cache ID"):
        common._augment_id(_Perturber({"sensor": object()}))


def _brightness_config(name: str, start: float, stop: float, num: int) -> NrtkRobustnessConfig:
    return NrtkRobustnessConfig(
        metric_key="fake_metric",
        name=name,
        perturber_factory={
            "type": "nrtk.impls.perturb_factory.PerturberLinspaceFactory",
            "nrtk.impls.perturb_factory.PerturberLinspaceFactory": {
                "perturber": "nrtk.impls.perturb_image.photometric.enhance.BrightnessPerturber",
                "theta_key": "factor",
                "start": start,
                "stop": stop,
                "num": num,
            },
        },
    )


def test_outputs_describe_the_sweep_and_include_the_baseline(
    fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
):
    run = NrtkRobustness().run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
        config=_brightness_config("sweep", 0.5, 1.5, 3),
    )
    outputs = run.outputs

    assert outputs.perturber == "BrightnessPerturber"
    assert outputs.theta_key == "factor"
    assert [point.theta for point in outputs.points] == [0.5, 1.0, 1.5]
    assert outputs.metric_key in outputs.baseline
    assert all(outputs.metric_key in point.metrics for point in outputs.points)


def test_baseline_is_the_score_on_the_unperturbed_dataset(
    fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
):
    """A brightness factor of 1.0 leaves the images unchanged, so it scores the same as the baseline."""
    from checkmaite.core.cached_tasks import evaluate

    run = NrtkRobustness().run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
        config=_brightness_config("identity", 1.0, 1.0, 1),
    )
    unperturbed, _, _ = evaluate(
        model=fake_od_model_default,
        dataset=fake_od_dataset_default,
        metric=fake_od_metric_default,
        batch_size=1,
        use_cache=False,
        task="object_detection",
    )

    metric_key = run.outputs.metric_key
    assert run.outputs.baseline[metric_key] == pytest.approx(float(unperturbed[metric_key]))
    assert run.outputs.points[0].metrics[metric_key] == pytest.approx(run.outputs.baseline[metric_key])


def test_report_and_records_show_the_baseline(fake_od_model_default, fake_od_dataset_default, fake_od_metric_default):
    run = NrtkRobustness().run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
        config=_brightness_config("report", 0.5, 1.5, 3),
    )
    metric_key = run.outputs.metric_key

    report = run.collect_md_report(threshold=0.5).content
    assert "Baseline (unperturbed)" in report
    assert "Change from baseline" in report

    primary = [record for record in run.extract() if record.is_primary]
    assert [record.theta_value for record in primary] == [0.5, 1.0, 1.5]
    baseline = float(run.outputs.baseline[metric_key])
    assert [record.baseline_value for record in primary] == pytest.approx([baseline] * 3)


def test_a_metric_with_one_number_needs_no_metric_key(fake_od_model_default, fake_od_dataset_default):
    metric = FakeODMetric(calculated_metrics={"map_50": torch.tensor(0.4)}, metric_metadata={"id": "single_number"})
    config = _brightness_config("single", 1.0, 1.0, 1).model_copy(update={"metric_key": None})

    run = NrtkRobustness().run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[metric],
        datasets=[fake_od_dataset_default],
        config=config,
    )

    assert run.outputs.metric_key == "map_50"


def test_an_ambiguous_metric_fails_after_the_baseline_and_before_the_sweep(
    mocker, fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
):
    from checkmaite.core._common import nrtk_robustness_capability as common

    evaluate = mocker.spy(common, "evaluate")
    config = _brightness_config("ambiguous", 0.5, 1.5, 3).model_copy(update={"metric_key": None})

    with pytest.raises(ValueError, match="several numeric results"):
        NrtkRobustness().run(
            use_cache=False,
            models=[fake_od_model_default],
            metrics=[fake_od_metric_default],
            datasets=[fake_od_dataset_default],
            config=config,
        )

    assert evaluate.call_count == 1  # only the baseline


def test_a_different_metric_key_is_a_different_run(
    fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
):
    def run_with(metric_key):
        return NrtkRobustness().run(
            models=[fake_od_model_default],
            metrics=[fake_od_metric_default],
            datasets=[fake_od_dataset_default],
            config=_brightness_config("keys", 1.0, 1.0, 1).model_copy(update={"metric_key": metric_key}),
        )

    first, second = run_with("fake_metric"), run_with("fake_metric_1")

    assert first.run_uid != second.run_uid
    assert (first.outputs.metric_key, second.outputs.metric_key) == ("fake_metric", "fake_metric_1")


class _SharedStateMetric:
    """Returns an array that later updates overwrite in place, as some metrics' internal state does."""

    def __init__(self, metric_id):
        self.metadata = {"id": metric_id}
        self.state = np.zeros(())
        self.updates = 0

    def reset(self):
        self.state[...] = 0

    def update(self, preds, targets, metadata=None):
        self.updates += 1
        self.state[...] = self.updates

    def compute(self):
        return {"score": self.state}


def test_results_are_not_changed_by_later_evaluations(fake_od_model_default, fake_od_dataset_default):
    run = NrtkRobustness().run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[_SharedStateMetric("shared-state")],
        datasets=[fake_od_dataset_default],
        config=_brightness_config("shared", 0.5, 1.5, 2).model_copy(update={"metric_key": None}),
    )

    scores = [float(run.outputs.baseline["score"]), *(float(p.metrics["score"]) for p in run.outputs.points)]
    assert len(set(scores)) == 3  # each evaluation kept its own result


class _UndefinedAfterBaselineMetric:
    """Defined on the unperturbed dataset, undefined (NaN) at every perturbation level."""

    def __init__(self):
        self.metadata = {"id": "undefined-after-baseline"}
        self.computes = 0

    def reset(self):
        pass

    def update(self, preds, targets, metadata=None):
        pass

    def compute(self):
        self.computes += 1
        return {"score": 0.5 if self.computes == 1 else float("nan")}


def test_undefined_levels_are_none_report_as_undefined_and_survive_the_cache(
    fake_od_model_default, fake_od_dataset_default
):
    def run():
        return NrtkRobustness().run(
            models=[fake_od_model_default],
            metrics=[_UndefinedAfterBaselineMetric()],
            datasets=[fake_od_dataset_default],
            config=_brightness_config("undefined", 0.5, 1.5, 2).model_copy(update={"metric_key": None}),
        )

    fresh, cached = run(), run()

    assert [point.metrics["score"] for point in fresh.outputs.points] == [None, None]
    assert cached.outputs.model_dump() == fresh.outputs.model_dump()
    assert "undefined" in cached.collect_md_report(threshold=0.5).content
    assert cached.extract() == []  # no level has a value to record
