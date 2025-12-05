from jatic_ri.core.image_classification.dataeval_feasability_capability import (
    DataevalFeasibility,
)


def test_run_and_collect(fake_ic_dataset_default):
    capability = DataevalFeasibility()

    output = capability.run(use_cache=False, datasets=[fake_ic_dataset_default])

    assert output.outputs.ber == 0.7
    assert output.outputs.ber_lower == 0.49013621203813906

    assert output.collect_report_consumables(threshold=0.5)  # smoke test
