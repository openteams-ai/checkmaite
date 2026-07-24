"""Shared assertions for typed capability reports."""

from checkmaite.core.report import InlineTextReport


def assert_inline_markdown_report(report: object, *, capability_id: str) -> InlineTextReport:
    """Assert the complete contract for a capability's inline Markdown report."""
    assert isinstance(report, InlineTextReport)
    assert report.media_type == "text/markdown"
    assert report.filename == f"{capability_id}.md"
    assert report.content.strip()
    return report
