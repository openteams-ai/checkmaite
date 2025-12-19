from typing import Any

import pytest

from jatic_ri.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
)


class MockConfig(CapabilityConfigBase):
    pass


class MockOutputs(CapabilityOutputsBase):
    result: bool


class MockRun(CapabilityRunBase):
    config: MockConfig
    outputs: MockOutputs


class ParamCapability(Capability):
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
