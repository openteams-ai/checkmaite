import pytest

from jatic_ri.core.image_classification.dataeval_cleaning_capability import DataevalCleaning
from jatic_ri.core.report._gradient import HAS_GRADIENT


def ignore_degenerate_data_warnings(test_fn):
    for filter in [
        "ignore:invalid value encountered in scalar divide:RuntimeWarning",
        "ignore:Precision loss occurred in moment calculation due to catastrophic cancellation:RuntimeWarning",
    ]:
        test_fn = pytest.mark.filterwarnings(filter)(test_fn)

    return test_fn


# Pytest does not support applying decorators to fixtures directly
# so we can't transform fake_ic_dataset_default fixture here.
@ignore_degenerate_data_warnings
def test_run_and_collect(fake_ic_dataset_default):
    capability = DataevalCleaning()

    output = capability.run(use_cache=False, datasets=[fake_ic_dataset_default])

    assert output.model_dump()  # smoke test


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
@ignore_degenerate_data_warnings
def test_collect_reports(fake_ic_dataset_default):
    capability = DataevalCleaning()
    output = capability.run(use_cache=False, datasets=[fake_ic_dataset_default])
    assert output.collect_report_consumables(threshold=0.5)  # smoke test


@ignore_degenerate_data_warnings
def test_collect_md_report(fake_ic_dataset_default):
    capability = DataevalCleaning()
    output = capability.run(use_cache=False, datasets=[fake_ic_dataset_default])
    assert output.collect_md_report(threshold=0.5)  # smoke test
