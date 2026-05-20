from __future__ import annotations

from collections.abc import Mapping
from typing import Any

JOB_SUBMISSION_CACHE_ERROR = (
    "use_cache=True is not supported in job submission mode. Workers run with use_cache=False "
    "because worker-local caches are ephemeral and are not shared with clients or other workers. "
    "Use backend job dedupe and the analytics store for durable job-submission reuse."
)


def prepare_job_submission_run_kwargs(run_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return capability kwargs with job-submission cache semantics enforced."""
    prepared = dict(run_kwargs)
    if bool(prepared.get("use_cache", False)):
        raise ValueError(JOB_SUBMISSION_CACHE_ERROR)
    prepared["use_cache"] = False
    return prepared
