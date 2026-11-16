import hashlib
import importlib
import importlib.metadata
import json
from typing import Any

import pytest

from checkmaite.core import capability_core
from checkmaite.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
    run_cache,
)
from checkmaite.core.report import InlineTextReport


class MockConfig(CapabilityConfigBase):
    pass


class MockOutputs(CapabilityOutputsBase):
    result: bool


class MockRun(CapabilityRunBase):
    config: MockConfig
    outputs: MockOutputs

    def collect_md_report(self, threshold: float) -> InlineTextReport:
        return InlineTextReport(
            media_type="text/plain",
            content=f"result={self.outputs.result}; threshold={threshold}",
            filename="report.txt",
        )


class MissingReportRun(CapabilityRunBase):
    config: MockConfig
    outputs: MockOutputs


class ParamCapability(Capability):
    task = "object_detection"

    _RUN_TYPE = MockRun

    def __init__(
        self,
        supports_models: Number,
        supports_datasets: Number,
        supports_metrics: Number,
    ) -> None:
        super().__init__()
        self._supports_models = supports_models
        self._supports_datasets = supports_datasets
        self._supports_metrics = supports_metrics

    @classmethod
    def _create_config(cls):
        return MockConfig()

    def _run(self, models, datasets, metrics, config, use_prediction_and_evaluation_cache):
        # The base class is expected to validate counts against supports_*.
        # We just return a success here if validation passes.
        return MockOutputs(result=True)

    @property
    def supports_models(self) -> Number:
        return self._supports_models

    @property
    def supports_datasets(self) -> Number:
        return self._supports_datasets

    @property
    def supports_metrics(self) -> Number:
        return self._supports_metrics

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        return [{"report": True}]


def _valid_len(card: Number) -> int:
    if card is Number.ZERO:
        return 0
    if card is Number.ONE:
        return 1
    if card is Number.TWO:
        return 2
    if card is Number.MANY:
        return 1  # minimal valid (>=1)
    raise RuntimeError(f"Unhandled card {card!r}")


def _invalid_len(card: Number) -> int:
    # Choose a single representative invalid length per cardinality
    if card is Number.ZERO:
        return 1  # any >0 is invalid
    if card is Number.ONE:
        return 0  # could also choose 2; 0 suffices
    if card is Number.TWO:
        return 1  # could also choose 3; 1 suffices
    if card is Number.MANY:
        return 0  # 0 invalid for MANY
    raise RuntimeError(f"Unhandled card {card!r}")


def _mk_models(n: int, fake_od_model_default) -> list[object]:
    return [fake_od_model_default for _ in range(n)]


def _mk_datasets(n: int, fake_od_dataset_default) -> list[object]:
    return [fake_od_dataset_default for _ in range(n)]


def _mk_metrics(n: int, fake_od_metric_default) -> list[object]:
    return [fake_od_metric_default for _ in range(n)]


def test_capability_run_default_collect_md_report_is_backward_compatible() -> None:
    run = MissingReportRun(
        capability_id="missing-report",
        config=MockConfig(),
        dataset_metadata=[],
        model_metadata=[],
        metric_metadata=[],
        outputs=MockOutputs(result=True),
    )

    with pytest.raises(NotImplementedError):
        run.collect_md_report(threshold=0.5)


@pytest.mark.parametrize("card", [Number.ZERO, Number.ONE, Number.TWO, Number.MANY])
def test_capability_collect_with_run(
    card: Number, fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
) -> None:
    # Use the same cardinality for all three axes to keep this test simple.
    stage = ParamCapability(supports_models=card, supports_datasets=card, supports_metrics=card)

    models = _mk_models(_valid_len(card), fake_od_model_default)
    datasets = _mk_datasets(_valid_len(card), fake_od_dataset_default)
    metrics = _mk_metrics(_valid_len(card), fake_od_metric_default)

    # Should not raise for valid lengths
    stage.run(models=models, datasets=datasets, metrics=metrics)

    report = stage.collect_report_consumables()
    assert report

    # Sanity-check properties reflect the parameterization
    assert stage.supports_models is card
    assert stage.supports_datasets is card
    assert stage.supports_metrics is card


@pytest.mark.parametrize("axis", ["models", "datasets", "metrics"])
@pytest.mark.parametrize("card", [Number.ZERO, Number.ONE, Number.TWO, Number.MANY])
def test_run_succeeds_with_valid_counts(
    card: Number, axis: str, fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
) -> None:
    # Configure only the tested axis with `card`; others ZERO.
    supports = {
        "supports_models": Number.ZERO,
        "supports_datasets": Number.ZERO,
        "supports_metrics": Number.ZERO,
    }
    supports[f"supports_{axis}"] = card
    stage = ParamCapability(**supports)

    models = _mk_models(_valid_len(supports["supports_models"]), fake_od_model_default)
    datasets = _mk_datasets(_valid_len(supports["supports_datasets"]), fake_od_dataset_default)
    metrics = _mk_metrics(_valid_len(supports["supports_metrics"]), fake_od_metric_default)

    stage.run(models=models, datasets=datasets, metrics=metrics)


@pytest.mark.parametrize("axis", ["models", "datasets", "metrics"])
@pytest.mark.parametrize("card", [Number.ZERO, Number.ONE, Number.TWO, Number.MANY])
def test_run_raises_with_invalid_counts(
    card: Number, axis: str, fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
) -> None:
    supports = {
        "supports_models": Number.ZERO,
        "supports_datasets": Number.ZERO,
        "supports_metrics": Number.ZERO,
    }
    supports[f"supports_{axis}"] = card
    stage = ParamCapability(**supports)

    # Valid counts for non-tested axes…
    models = _mk_models(_valid_len(supports["supports_models"]), fake_od_model_default)
    datasets = _mk_datasets(_valid_len(supports["supports_datasets"]), fake_od_dataset_default)
    metrics = _mk_metrics(_valid_len(supports["supports_metrics"]), fake_od_metric_default)

    # …replace the tested axis with an invalid length.
    if axis == "models":
        models = _mk_models(_invalid_len(card), fake_od_model_default)
    elif axis == "datasets":
        datasets = _mk_datasets(_invalid_len(card), fake_od_dataset_default)
    else:
        metrics = _mk_metrics(_invalid_len(card), fake_od_metric_default)

    with pytest.raises(TypeError):
        stage.run(models=models, datasets=datasets, metrics=metrics)


class FakeLibCapability(ParamCapability):
    cache_dependencies = ("fakelib",)


def _fake_library_version(monkeypatch: pytest.MonkeyPatch, fakelib_version: str) -> None:
    real_version = importlib.metadata.version

    def version(distribution_name: str) -> str:
        if distribution_name == "fakelib":
            return fakelib_version
        return real_version(distribution_name)

    monkeypatch.setattr(capability_core.importlib.metadata, "version", version)


def _run_uid(capability: Capability, dataset) -> str:
    return capability.run(datasets=[dataset]).run_uid


def _dataset_capability(cls: type[ParamCapability] = ParamCapability) -> ParamCapability:
    return cls(supports_models=Number.ZERO, supports_datasets=Number.ONE, supports_metrics=Number.ZERO)


def test_cache_versions_default_to_capability_version_only() -> None:
    assert ParamCapability.cache_versions() == {"capability": "1"}


def test_cache_versions_include_exact_version_of_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_library_version(monkeypatch, "1.2.3")

    assert FakeLibCapability.cache_versions() == {"capability": "1", "fakelib": "1.2.3"}


def test_run_records_versions_and_uid_matches_cache_key(
    monkeypatch: pytest.MonkeyPatch, fake_od_dataset_default
) -> None:
    _fake_library_version(monkeypatch, "1.2.3")
    capability = _dataset_capability(FakeLibCapability)

    run = capability.run(datasets=[fake_od_dataset_default])

    assert run.versions == {"capability": "1", "fakelib": "1.2.3"}
    assert run_cache.get(run.run_uid) == run


def test_run_uid_is_stable_for_the_same_library_version(
    monkeypatch: pytest.MonkeyPatch, fake_od_dataset_default
) -> None:
    capability = _dataset_capability(FakeLibCapability)

    _fake_library_version(monkeypatch, "1.2.3")
    first = _run_uid(capability, fake_od_dataset_default)
    assert _run_uid(capability, fake_od_dataset_default) == first


@pytest.mark.parametrize("new_version", ["1.2.4", "1.3.0", "2.2.3"])
def test_run_uid_changes_with_any_library_release(
    monkeypatch: pytest.MonkeyPatch, fake_od_dataset_default, new_version: str
) -> None:
    capability = _dataset_capability(FakeLibCapability)

    _fake_library_version(monkeypatch, "1.2.3")
    before = capability.run(datasets=[fake_od_dataset_default])

    _fake_library_version(monkeypatch, new_version)
    after = capability.run(datasets=[fake_od_dataset_default])

    assert after.run_uid != before.run_uid
    # The old cache entry is not reused: the run was recomputed with the new versions.
    assert after.versions["fakelib"] == new_version


def test_run_uid_changes_when_capability_version_is_bumped(fake_od_dataset_default) -> None:
    class BumpedCapability(ParamCapability):
        version = "2"

    before = _run_uid(_dataset_capability(), fake_od_dataset_default)
    after = _run_uid(_dataset_capability(BumpedCapability), fake_od_dataset_default)

    assert after != before


def test_compute_uid_depends_on_versions() -> None:
    kwargs: dict[str, Any] = {
        "capability_id": "cap",
        "config": MockConfig(),
        "dataset_metadata": [{"id": "d"}],
        "model_metadata": [],
        "metric_metadata": [],
    }

    same = MockRun.compute_uid(**kwargs, versions={"capability": "1", "lib": "1.2"})
    reordered = MockRun.compute_uid(**kwargs, versions={"lib": "1.2", "capability": "1"})
    bumped = MockRun.compute_uid(**kwargs, versions={"capability": "2", "lib": "1.2"})

    assert same == reordered
    assert same != bumped


def test_a_run_saved_without_versions_keeps_its_uid() -> None:
    """Runs saved before versions were recorded load with versions={} and the UID they had."""
    config = MockConfig()
    old_content = {
        "capability_id": "cap",
        "config": config.model_dump(mode="json"),
        "dataset_ids": ["d"],
        "model_ids": [],
        "metric_id": [],
    }
    old_uid = hashlib.sha256(json.dumps(old_content).encode("utf-8")).hexdigest()

    uid = MockRun.compute_uid(
        capability_id="cap",
        config=config,
        dataset_metadata=[{"id": "d"}],
        model_metadata=[],
        metric_metadata=[],
        versions={},
    )
    assert uid == old_uid


@pytest.mark.parametrize(
    ("module", "distribution"),
    [
        ("checkmaite.core.image_classification.dataeval_bias_capability.DataevalBias", "dataeval"),
        ("checkmaite.core.object_detection.dataeval_bias_capability.DataevalBias", "dataeval"),
        ("checkmaite.core.image_classification.dataeval_cleaning_capability.DataevalCleaning", "dataeval"),
        ("checkmaite.core.object_detection.dataeval_cleaning_capability.DataevalCleaning", "dataeval"),
        ("checkmaite.core.image_classification.dataeval_shift_capability.DataevalShift", "dataeval"),
        ("checkmaite.core.object_detection.dataeval_shift_capability.DataevalShift", "dataeval"),
        ("checkmaite.core.image_classification.dataeval_feasibility_capability.DataevalFeasibility", "dataeval"),
        ("checkmaite.core.object_detection.dataeval_feasibility_capability.DataevalFeasibility", "dataeval"),
        ("checkmaite.core.image_classification.dataeval_sufficiency_capability.DataevalSufficiency", "dataeval"),
        ("checkmaite.core.image_classification.nrtk_robustness_capability.NrtkRobustness", "nrtk"),
        ("checkmaite.core.object_detection.nrtk_robustness_capability.NrtkRobustness", "nrtk"),
        ("checkmaite.core.image_classification.xaitk_explainable_capability.XaitkExplainable", "xaitk-saliency"),
        ("checkmaite.core.object_detection.xaitk_explainable_capability.XaitkExplainable", "xaitk-saliency"),
    ],
)
def test_builtin_capabilities_declare_wrapped_library(module: str, distribution: str) -> None:
    module_name, class_name = module.rsplit(".", 1)
    capability_cls = getattr(importlib.import_module(module_name), class_name)

    assert capability_cls.cache_dependencies == (distribution,)
    assert capability_cls.cache_versions()[distribution] == importlib.metadata.version(distribution)


# Capabilities whose _run reads the prediction and evaluation caches through
# cached_tasks.evaluate, with the libraries those caches key on for them.
# NRTK Robustness builds its augmentation IDs from the exact NRTK version;
# the other keys hold only user-supplied IDs.
_READS_PREDICTION_CACHE = {
    "checkmaite.core.image_classification.maite_evaluation_capability.MaiteEvaluation": (),
    "checkmaite.core.object_detection.maite_evaluation_capability.MaiteEvaluation": (),
    "checkmaite.core.multiobject_tracking.maite_evaluation_capability.MaiteEvaluation": (),
    "checkmaite.core.image_classification.nrtk_robustness_capability.NrtkRobustness": ("nrtk",),
    "checkmaite.core.object_detection.nrtk_robustness_capability.NrtkRobustness": ("nrtk",),
}


@pytest.mark.parametrize(("module", "keyed_libraries"), _READS_PREDICTION_CACHE.items())
def test_cache_dependencies_are_part_of_the_lower_level_cache_keys(module: str, keyed_libraries: tuple) -> None:
    """A library in cache_dependencies must also be in the prediction cache key the capability reads.

    Otherwise a library upgrade changes the run UID, the capability reuses
    cached predictions computed before the upgrade, and the run records the
    new library version for results it was not computed with.
    """
    module_name, class_name = module.rsplit(".", 1)
    capability_cls = getattr(importlib.import_module(module_name), class_name)

    assert capability_cls.cache_dependencies == keyed_libraries


def test_maite_upgrade_keeps_the_maite_evaluation_run_uid(
    monkeypatch: pytest.MonkeyPatch, fake_od_model_default, fake_od_dataset_default, fake_od_metric_default
) -> None:
    from checkmaite.core.object_detection.maite_evaluation_capability import MaiteEvaluation

    real_version = importlib.metadata.version
    inputs = {
        "models": [fake_od_model_default],
        "datasets": [fake_od_dataset_default],
        "metrics": [fake_od_metric_default],
    }

    def run_with_maite(version: str):
        monkeypatch.setattr(
            capability_core.importlib.metadata,
            "version",
            lambda name: version if name == "maite" else real_version(name),
        )
        return MaiteEvaluation().run(**inputs)

    before, after = run_with_maite("0.9.5"), run_with_maite("0.10.3")

    assert after.run_uid == before.run_uid
    assert after.versions == {"capability": MaiteEvaluation.version}


def test_config_rejects_unknown_fields():
    """A removed or misspelled option must fail, not be silently ignored."""
    from pydantic import ValidationError

    from checkmaite.core._common.dataeval_sufficiency_capability import DataevalSufficiencyConfig

    with pytest.raises(ValidationError, match="target_metric_name"):
        DataevalSufficiencyConfig(target_metric_name="accuracy")
