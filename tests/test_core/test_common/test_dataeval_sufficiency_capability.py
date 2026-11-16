from typing import Any

import numpy as np
import pytest
import torch
from pydantic.json_schema import GenerateJsonSchema

from checkmaite.core._common.dataeval_sufficiency_capability import (
    DataevalSufficiencyBase,
    DataevalSufficiencyConfig,
    _SufficiencyLimits,
)
from checkmaite.core.image_classification.metrics import accuracy_multiclass_torch_metric_factory
from tests.report_assertions import assert_inline_markdown_report


def do_smoke_run(dataset, monkeypatch):
    def _test_limits(cls):
        return _SufficiencyLimits(min_dataset_size=10, min_samples_per_class=5, min_metric_abs_diff_ratio=0.45)

    monkeypatch.setattr(DataevalSufficiencyBase, "_limits", classmethod(_test_limits))

    class _Sufficiency(DataevalSufficiencyBase):
        task = "image_classification"

    capability = _Sufficiency()

    def _get_model_and_preprocess_fns(num_classes, image_size):
        class Model(torch.nn.Module):
            def forward(self, x):
                output = torch.zeros(x.shape[0], num_classes, device=x.device)
                output[:, 0] = 1.0
                return output

        return Model(), lambda x: x, lambda x: x

    capability._get_model_and_preprocess_fns = _get_model_and_preprocess_fns

    def _get_training_strategy(*args, **kwargs):
        class TrainStrategy:
            def train(self, model, dataset, indices):
                pass

        return TrainStrategy()

    def _get_evaluation_strategy(*args, **kwargs):
        class EvalStrategy:
            def __init__(self):
                self.trial_id = 0

            def evaluate(self, model, dataset):
                self.trial_id += 1
                return {"accuracy": 0.2 * self.trial_id}

        return EvalStrategy()

    capability._get_training_strategy = _get_training_strategy
    capability._get_evaluation_strategy = _get_evaluation_strategy

    config = DataevalSufficiencyConfig(
        num_iters=2,
        batch_size=4,
        use_amp=False,
        sufficiency_schedule=[
            len(dataset) // 4,
            len(dataset) // 2,
            len(dataset),
        ],
    )

    metric = accuracy_multiclass_torch_metric_factory(num_classes=10)

    return capability.run(
        use_cache=False,
        datasets=[dataset],
        config=config,
        metrics=[metric],
    )  # smoke test


@pytest.fixture
def test_run_ic(fake_ic_dataset_default, monkeypatch) -> Any:
    return do_smoke_run(fake_ic_dataset_default, monkeypatch)


class _RuntimeTypeSchemaGenerator(GenerateJsonSchema):
    def handle_invalid_for_json_schema(self, schema, error_info):
        return {"not": {}, "description": f"Runtime-only value: {error_info}"}


def test_config_schema_requires_exactly_one_training_limit() -> None:
    schema = DataevalSufficiencyConfig.model_json_schema(schema_generator=_RuntimeTypeSchemaGenerator)

    assert schema["oneOf"] == [
        {
            "required": ["num_epochs"],
            "properties": {
                "num_epochs": {"type": "integer"},
                "num_iters": {"type": "null"},
            },
        },
        {
            "required": ["num_iters"],
            "properties": {
                "num_epochs": {"type": "null"},
                "num_iters": {"type": "integer"},
            },
        },
    ]


def test_sufficiency_output(fake_ic_dataset_default, monkeypatch):
    run_output = do_smoke_run(fake_ic_dataset_default, monkeypatch)
    output = run_output.outputs
    assert output.target_metric_name == "accuracy"
    assert output.target_dataset_size == 67
    np.testing.assert_allclose(
        output.sufficiency_table["step"],
        [
            len(fake_ic_dataset_default) // 4,
            len(fake_ic_dataset_default) // 2,
            len(fake_ic_dataset_default),
        ],
    )
    np.testing.assert_allclose(output.sufficiency_table["accuracy"], [0.2, 0.4, 0.6])


def test_collect_md_report_ic(test_run_ic):
    report = test_run_ic.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=test_run_ic.capability_id)


@pytest.mark.optional
@pytest.mark.skip(reason="gradient report generation is not implemented")
def test_collect_report_consumables_ic(test_run_ic):
    with pytest.warns(DeprecationWarning):
        consumables = test_run_ic.collect_report_consumables(threshold=0.5)
    assert consumables  # smoke test


class _DictMetric:
    def __init__(self, result):
        self.result = result
        self.metadata = {"id": "dict_metric"}

    def reset(self):
        pass

    def update(self, preds, targets, metadata):
        pass

    def compute(self):
        return self.result


class _ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 2)


_TINY_DATASET = [(torch.zeros(1, 2, 2), 0, {})] * 2


def _strategy(metrics, metric_key=None):
    from checkmaite.core._common.dataeval_sufficiency_capability import _DefaultEvaluationStrategy

    return _DefaultEvaluationStrategy(
        metrics=metrics, batch_size=2, device=torch.device("cpu"), num_workers=0, verbose=False, metric_key=metric_key
    )


def test_evaluation_fails_on_an_ambiguous_target_metric():
    strategy = _strategy([_DictMetric({"accuracy": 0.5, "f1_score": 0.4})])

    with pytest.raises(ValueError, match="Set metric_key"):
        strategy.evaluate(_ZeroModel(), _TINY_DATASET)


def test_evaluation_tracks_the_configured_key_and_ignores_extra_metrics():
    strategy = _strategy([_DictMetric({"accuracy": 0.5, "f1_score": 0.4}), _DictMetric({"recall": 0.3})], "f1_score")

    assert strategy.evaluate(_ZeroModel(), _TINY_DATASET) == {"accuracy": 0.5, "f1_score": 0.4, "recall": 0.3}
    assert strategy.resolved_metric_key == "f1_score"


def test_a_single_number_target_metric_needs_no_key_even_with_extra_metrics():
    strategy = _strategy([_DictMetric({"accuracy": 0.5}), _DictMetric({"recall": 0.3})])

    strategy.evaluate(_ZeroModel(), _TINY_DATASET)

    assert strategy.resolved_metric_key == "accuracy"
