from __future__ import annotations

from collections.abc import Mapping
from typing import Any

JOB_SUBMISSION_CACHE_ERROR = (
    "use_cache=True is not supported in job submission mode. Workers run with use_cache=False "
    "because worker-local caches are ephemeral and are not shared with clients or other workers. "
    "Use backend job dedupe and the analytics store for durable job-submission reuse."
)


def _normalize_required_job_name(job_name: object) -> str:
    if not isinstance(job_name, str):
        raise TypeError("job_name must be a string or None")
    normalized = job_name.strip()
    if not normalized:
        raise ValueError("job_name must be non-empty when provided")
    if len(normalized.encode("utf-8")) > 256:
        raise ValueError("job_name must be at most 256 UTF-8 bytes")
    return normalized


def normalize_job_name(job_name: object | None) -> str | None:
    """Validate and normalize an optional bounded user-facing job label."""
    if job_name is None:
        return None
    return _normalize_required_job_name(job_name)


def resolve_job_name(job_name: object | None, capability_id: object) -> str:
    """Return a bounded user label, defaulting to the capability identifier."""
    return _normalize_required_job_name(str(capability_id) if job_name is None else job_name)


def prepare_job_submission_run_kwargs(run_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return capability kwargs with job-submission cache semantics enforced."""
    prepared = dict(run_kwargs)
    if bool(prepared.get("use_cache", False)):
        raise ValueError(JOB_SUBMISSION_CACHE_ERROR)
    prepared["use_cache"] = False
    return prepared
