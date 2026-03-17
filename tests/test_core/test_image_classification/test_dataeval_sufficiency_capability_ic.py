from typing import Any

import numpy as np
import pytest

from checkmaite.core._common.dataeval_sufficiency_capability import _SufficiencyLimits
from checkmaite.core.image_classification.dataeval_sufficiency_capability import (
    DataevalSufficiency,
    DataevalSufficiencyConfig,
)
from checkmaite.core.image_classification.metrics import accuracy_multiclass_torch_metric_factory


def do_smoke_run(dataset, monkeypatch):
    def _test_limits(cls):
        return _SufficiencyLimits(min_dataset_size=10, min_samples_per_class=5, min_metric_abs_diff_ratio=0.45)

    monkeypatch.setattr(DataevalSufficiency, "_limits", classmethod(_test_limits))

    capability = DataevalSufficiency()

    config = DataevalSufficiencyConfig(
        num_iters=2,
        batch_size=4,
        use_amp=False,
        sufficiency_schedule=[
            len(dataset) // 4,
            len(dataset) // 2,
            len(dataset),
        ],
        sufficiency_num_runs=2,
    )

    metric = accuracy_multiclass_torch_metric_factory(num_classes=10)

    return capability.run(
        use_cache=False,
        datasets=[dataset],
        config=config,
        metrics=[metric],
    )  # smoke test


@pytest.fixture
def test_run_ic(fake_ic_dataset_ten_unique_classes, monkeypatch) -> Any:
    return do_smoke_run(fake_ic_dataset_ten_unique_classes, monkeypatch)


def test_sufficiency_output(fake_ic_dataset_ten_unique_classes, monkeypatch):
    run_output = do_smoke_run(fake_ic_dataset_ten_unique_classes, monkeypatch)
    output = run_output.outputs
    assert output.target_metric_name == "accuracy"
    assert output.target_dataset_size is None
    np.testing.assert_allclose(
        output.sufficiency_table["step"],
        [
            len(fake_ic_dataset_ten_unique_classes) // 4,
            len(fake_ic_dataset_ten_unique_classes) // 2,
            len(fake_ic_dataset_ten_unique_classes),
        ],
    )
    np.testing.assert_allclose(output.sufficiency_table["accuracy"], [0.1, 0.1, 0.1])


def test_collect_md_report_ic(test_run_ic):
    md = test_run_ic.collect_md_report(threshold=0.5)
    assert md  # smoke test
