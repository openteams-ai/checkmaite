"""Tests for shared capability-job result construction."""

import pytest
import ray.cloudpickle

from checkmaite.core.report import ArtifactReport, InlineTextReport
from checkmaite.jobs import CapabilityRunRef
from checkmaite.jobs._result import build_capability_run_ref
from tests.test_jobs.fakes import OversizedReportTinyCapability, ReportlessTinyCapability, TinyCapability, TinyConfig


def test_build_capability_run_ref_collects_typed_report() -> None:
    run = TinyCapability().run(config=TinyConfig(text="reported"), use_cache=False)

    ref = build_capability_run_ref(run, store_uri="memory://run", report_threshold=0.75)

    assert ref.report == InlineTextReport(
        media_type="text/markdown",
        content="reported:0.75",
        filename="tiny-report.md",
    )


def test_build_capability_run_ref_returns_placeholder_for_oversized_inline_report() -> None:
    run = OversizedReportTinyCapability().run(config=TinyConfig(text="oversized"), use_cache=False)

    with pytest.warns(RuntimeWarning, match="exceeded the inline report size limit"):
        ref = build_capability_run_ref(run, store_uri="memory://run", report_threshold=0.5)

    assert ref.store_uri == "memory://run"
    assert ref.report == InlineTextReport(
        media_type="text/markdown",
        content=(
            "# Report unavailable\n\n"
            "The generated report exceeded the inline report size limit. "
            "Analytics results remain available through `store_uri`."
        ),
        filename="report-too-large.md",
    )


def test_build_capability_run_ref_supports_run_without_report() -> None:
    run = ReportlessTinyCapability().run(config=TinyConfig(text="legacy"), use_cache=False)

    ref = build_capability_run_ref(run, store_uri="memory://run", report_threshold=0.5)

    assert ref.report is None


def test_capability_run_ref_round_trips_through_ray_cloudpickle() -> None:
    ref = CapabilityRunRef(
        run_uid="run-1",
        capability_id="capability-1",
        store_uri="memory://run-1",
        report=ArtifactReport(
            media_type="application/pdf",
            uri="s3://reports/run-1.pdf",
            filename="run-1.pdf",
        ),
    )

    restored = ray.cloudpickle.loads(ray.cloudpickle.dumps(ref))

    assert restored == ref
    assert isinstance(restored.report, ArtifactReport)
