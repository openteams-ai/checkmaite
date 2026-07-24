import pytest
from pydantic import ValidationError

from checkmaite.core.report import MAX_INLINE_REPORT_BYTES, ArtifactReport, InlineTextReport
from checkmaite.jobs import CapabilityRunRef


@pytest.mark.parametrize(
    ("report", "report_type"),
    [
        (
            InlineTextReport(media_type="text/markdown", content="# Results", filename="results.md"),
            InlineTextReport,
        ),
        (
            ArtifactReport(media_type="application/pdf", uri="s3://reports/results.pdf", filename="results.pdf"),
            ArtifactReport,
        ),
    ],
)
def test_capability_run_ref_serialization_round_trips_report_variants(report, report_type) -> None:
    ref = CapabilityRunRef(
        run_uid="run-1",
        capability_id="capability-1",
        store_uri="file:///analytics/run-1.parquet",
        report=report,
    )

    payload = ref.model_dump(mode="json")
    restored = CapabilityRunRef.model_validate(payload)

    assert payload["report"]["kind"] == report.kind
    assert isinstance(restored.report, report_type)
    assert restored == ref


@pytest.mark.parametrize(
    "report",
    [
        {"kind": "inline_text", "media_type": "text/markdown", "filename": "results.md"},
        {"kind": "artifact", "media_type": "application/pdf", "filename": "results.pdf"},
        {"kind": "unknown", "media_type": "text/plain", "filename": "results.txt"},
        ["not", "a", "report"],
    ],
)
def test_capability_run_ref_rejects_invalid_reports(report) -> None:
    with pytest.raises(ValidationError):
        CapabilityRunRef(
            run_uid="run-1",
            capability_id="capability-1",
            store_uri="file:///analytics/run-1.parquet",
            report=report,
        )


def test_capability_run_ref_requires_report() -> None:
    with pytest.raises(ValidationError):
        CapabilityRunRef(
            run_uid="run-1",
            capability_id="capability-1",
            store_uri="file:///analytics/run-1.parquet",
        )


@pytest.mark.parametrize(
    "report",
    [
        {"media_type": "", "content": "results", "filename": "results.md"},
        {"media_type": "text/markdown", "content": "", "filename": "results.md"},
        {"media_type": "text/markdown", "content": "results", "filename": "../results.md"},
        {"media_type": "text/markdown", "content": "results", "filename": "reports/results.md"},
        {"media_type": "text/markdown", "content": "results", "filename": r"reports\results.md"},
    ],
)
def test_inline_report_rejects_invalid_fields(report) -> None:
    with pytest.raises(ValidationError):
        InlineTextReport(**report)


def test_artifact_report_rejects_empty_uri_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ArtifactReport(media_type="application/pdf", uri="", filename="results.pdf")

    with pytest.raises(ValidationError):
        ArtifactReport.model_validate(
            {
                "media_type": "application/pdf",
                "uri": "s3://reports/results.pdf",
                "filename": "results.pdf",
                "content": "not valid for an artifact",
            }
        )


def test_inline_report_rejects_content_over_byte_limit() -> None:
    content = "é" * (MAX_INLINE_REPORT_BYTES // 2 + 1)

    with pytest.raises(ValidationError, match="return ArtifactReport instead"):
        InlineTextReport(media_type="text/plain", content=content, filename="results.txt")


def test_report_models_are_frozen() -> None:
    report = InlineTextReport(media_type="text/plain", content="results", filename="results.txt")

    with pytest.raises(ValidationError):
        report.content = "changed"
