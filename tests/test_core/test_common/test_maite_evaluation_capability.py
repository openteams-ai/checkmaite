import pytest

from checkmaite.core.image_classification.maite_evaluation_capability import MaiteEvaluation
from checkmaite.core.report._gradient import HAS_GRADIENT


@pytest.fixture
def test_run_ic(fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default):
    capability = MaiteEvaluation()
    output = capability.run(
        datasets=[fake_ic_dataset_default], metrics=[fake_ic_metric_default], models=[fake_ic_model_default]
    )
    assert output.model_dump()  # smoke test

    return output


def test_collect_md_report_ic(test_run_ic):
    assert test_run_ic.collect_md_report(threshold=0.5)  # smoke test


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_collect_report_consumables_ic(test_run_ic):
    assert test_run_ic.collect_report_consumables(threshold=0.5)  # smoke test
