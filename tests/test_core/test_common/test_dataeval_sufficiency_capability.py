from typing import Any

import numpy as np
import pytest
import torch

from checkmaite.core._common.dataeval_sufficiency_capability import (
    DataevalSufficiencyBase,
    DataevalSufficiencyConfig,
    _SufficiencyLimits,
)
from checkmaite.core.image_classification.metrics import accuracy_multiclass_torch_metric_factory


def do_smoke_run(dataset, monkeypatch):
    def _test_limits(cls):
        return _SufficiencyLimits(min_dataset_size=10, min_samples_per_class=5, min_metric_abs_diff_ratio=0.45)

    monkeypatch.setattr(DataevalSufficiencyBase, "_limits", classmethod(_test_limits))

    capability = DataevalSufficiencyBase()

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
    md = test_run_ic.collect_md_report(threshold=0.5)
    assert md  # smoke test


@pytest.mark.skip(reason="gradient report generation is not implemented")
def test_collect_report_consumables_ic(test_run_ic):
    with pytest.warns(DeprecationWarning):
        consumables = test_run_ic.collect_report_consumables(threshold=0.5)
    assert consumables  # smoke test
