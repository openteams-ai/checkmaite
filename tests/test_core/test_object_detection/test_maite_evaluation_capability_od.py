import pytest

from jatic_ri.core.object_detection.maite_evaluation_capability import MaiteEvaluation
from jatic_ri.core.object_detection.metrics import multiclass_map50_torch_metric_factory
from jatic_ri.core.report._gradient import HAS_GRADIENT


@pytest.fixture
def test_run(fake_od_model_default, fake_od_dataset_default, fake_od_metric_default):
    capability = MaiteEvaluation()

    output = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        metrics=[fake_od_metric_default],
        datasets=[fake_od_dataset_default],
    )

    assert output.model_dump()  # smoke test
    return output


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_collect_report_consumables(test_run):
    assert test_run.collect_report_consumables(threshold=0.5)  # smoke test


def test_collect_md_report(test_run):
    assert test_run.collect_md_report(threshold=0.5)  # smoke test


def test_multiclass(fake_od_model_default, fake_od_dataset_default):
    capability = MaiteEvaluation()

    metric = multiclass_map50_torch_metric_factory()

    output = capability.run(
        use_cache=False, models=[fake_od_model_default], metrics=[metric], datasets=[fake_od_dataset_default]
    )

    assert output.outputs.class_metrics is not None
