"""Shared capability-job result construction."""

import warnings
from typing import Any

from pydantic import ValidationError

from checkmaite.core.capability_core import CapabilityRunBase
from checkmaite.core.report import CapabilityReport, InlineTextReport
from checkmaite.jobs.protocol import CapabilityRunRef


def _is_oversized_inline_report_error(exc: ValidationError) -> bool:
    """Return whether validation failed only because inline report content was too large."""
    errors = exc.errors(include_input=False)
    return bool(errors) and all(
        error["loc"] == ("content",) and "inline report content is" in error["msg"] for error in errors
    )


def _collect_md_report(run: CapabilityRunBase[Any, Any], threshold: float) -> CapabilityReport | None:
    """Collect a typed report, preserving compatibility with runs that do not implement reporting."""
    try:
        return run.collect_md_report(threshold=threshold)
    except NotImplementedError:
        return None
    except ValidationError as exc:
        if not _is_oversized_inline_report_error(exc):
            raise
        warnings.warn(
            "Generated report exceeded the inline report size limit; returning a placeholder report. "
            "Analytics results remain available through store_uri.",
            RuntimeWarning,
            stacklevel=2,
        )
        return InlineTextReport(
            media_type="text/markdown",
            filename="report-too-large.md",
            content=(
                "# Report unavailable\n\n"
                "The generated report exceeded the inline report size limit. "
                "Analytics results remain available through `store_uri`."
            ),
        )


def build_capability_run_ref(
    run: CapabilityRunBase[Any, Any],
    *,
    store_uri: str | None,
    report_threshold: float,
) -> CapabilityRunRef:
    """Build the typed result returned by every capability job backend."""
    return CapabilityRunRef(
        run_uid=run.run_uid,
        capability_id=run.capability_id,
        store_uri=store_uri,
        outputs_uri=None,
        report=_collect_md_report(run, threshold=report_threshold),
    )
