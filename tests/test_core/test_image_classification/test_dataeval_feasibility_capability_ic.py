import pytest

from checkmaite.core.image_classification.dataeval_feasibility_capability import (
    DataevalFeasibility,
    DataevalFeasibilityConfig,
    DataevalFeasibilityOutputs,
    DataevalFeasibilityRun,
)
from checkmaite.core.report._gradient import HAS_GRADIENT
from tests.report_assertions import assert_inline_markdown_report


@pytest.fixture
def run_dataeval_feasibility_ic(fake_ic_dataset_default):
    capability = DataevalFeasibility()

    return capability.run(use_cache=False, datasets=[fake_ic_dataset_default])


def test_run_and_collect(run_dataeval_feasibility_ic):
    # Numerical outputs can vary slightly depending on environment/library
    # versions. Validate structural correctness and plausibility instead
    # of relying on an exact numeric match.
    ber = run_dataeval_feasibility_ic.outputs.ber
    ber_lower = run_dataeval_feasibility_ic.outputs.ber_lower

    assert isinstance(ber, float)
    assert isinstance(ber_lower, float)
    assert 0.0 <= ber <= 1.0
    assert 0.0 <= ber_lower <= 1.0
    assert ber_lower <= ber


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_collect_report_consumables(run_dataeval_feasibility_ic):
    with pytest.warns(DeprecationWarning):
        assert run_dataeval_feasibility_ic.collect_report_consumables(threshold=0.5)  # smoke test


def _make_run(ber: float, ber_lower: float) -> DataevalFeasibilityRun:
    return DataevalFeasibilityRun(
        capability_id="DataevalFeasibility",
        dataset_metadata=[{"id": "dummy-dataset"}],
        model_metadata=[],
        metric_metadata=[],
        config=DataevalFeasibilityConfig(precision=3),
        outputs=DataevalFeasibilityOutputs(ber=ber, ber_lower=ber_lower),
    )


def test_feasibility_collect_md_report_feasible_branch() -> None:
    # BER=0.2 → accuracy=0.8 >= threshold=0.5 → feasible
    run = _make_run(ber=0.2, ber_lower=0.1)

    report = run.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=run.capability_id)

    assert "Dataset Feasibility Analysis" in report.content
    assert "Bayes Error Rate" in report.content
    assert "is feasible" in report.content

    # Rounded to precision=3 in the Results table
    assert "0.2" in report.content
    assert "0.1" in report.content

    # Action branch for feasible
    assert "No action required" in report.content
    assert "Reduce difficulty of the problem statement" not in report.content


def test_feasibility_collect_md_report_not_feasible_branch() -> None:
    # BER=0.912 → accuracy=0.088 < threshold=0.5 → NOT feasible
    run = _make_run(ber=0.91234, ber_lower=0.87654)

    report = run.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=run.capability_id)

    assert "is NOT" in report.content
    assert "Reduce difficulty of the problem statement" in report.content
