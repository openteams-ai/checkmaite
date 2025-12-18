import pytest

from jatic_ri.core.object_detection.dataeval_feasability_capability import DataevalFeasibility


@pytest.mark.skip(reason="OD feasibility test stage is not available until MAITE>=0.8.0 is supported")
def test_run_and_collect(fake_od_model_default, fake_od_dataset_default):
    capability = DataevalFeasibility()

    output = capability.run(use_cache=False, datasets=[fake_od_dataset_default], models=[fake_od_model_default])

    assert output.model_dump()  # smoke test

    assert output.collect_report_consumables(threshold=0.5)  # smoke test

    assert output.collect_md_report(threshold=0.5)  # smoke test
