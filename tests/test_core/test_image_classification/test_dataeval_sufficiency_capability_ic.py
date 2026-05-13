import numpy as np

from checkmaite.core._common.dataeval_sufficiency_capability import _SufficiencyLimits
from checkmaite.core.image_classification.dataeval_sufficiency_capability import (
    DataevalSufficiency,
    DataevalSufficiencyConfig,
    _DefaultTrainingStrategy,
)
from checkmaite.core.image_classification.metrics import accuracy_multiclass_torch_metric_factory


def do_smoke_run(dataset, monkeypatch):
    def _test_limits(cls):
        return _SufficiencyLimits(min_dataset_size=10, min_samples_per_class=5, min_metric_abs_diff_ratio=0.45)

    monkeypatch.setattr(DataevalSufficiency, "_limits", classmethod(_test_limits))

    def _test_training_strategy(self, config):
        return _DefaultTrainingStrategy(
            num_epochs=config.num_epochs,
            num_iters=config.num_iters,
            batch_size=config.batch_size,
            device=config.device,
            verbose=config.verbose,
            use_amp=config.use_amp,
            num_workers=0,
        )

    monkeypatch.setattr(DataevalSufficiency, "_get_training_strategy", _test_training_strategy)

    capability = DataevalSufficiency()

    config = DataevalSufficiencyConfig(
        num_iters=1,
        batch_size=4,
        use_amp=False,
        sufficiency_schedule=[
            len(dataset) // 4,
            len(dataset) // 2,
            len(dataset),
        ],
        sufficiency_num_runs=1,
    )

    metric = accuracy_multiclass_torch_metric_factory(num_classes=10)

    return capability.run(
        use_cache=False,
        datasets=[dataset],
        config=config,
        metrics=[metric],
    )  # smoke test


def test_sufficiency_output_and_md_report(fake_ic_dataset_ten_unique_classes, monkeypatch):
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

    md = run_output.collect_md_report(threshold=0.5)
    assert md  # smoke test
