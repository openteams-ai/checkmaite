"""Shared capability execution for job-backend workers."""

from datetime import datetime, timezone
from typing import Any

from checkmaite.core.analytics_store import Provenance
from checkmaite.jobs._result import build_capability_run_ref
from checkmaite.jobs._store import (
    AnalyticsStoreConfig,
    build_analytics_store,
    resolve_artifact_store_config,
    write_run_and_get_store_uri,
)
from checkmaite.jobs._submission import prepare_job_submission_run_kwargs
from checkmaite.jobs.protocol import CapabilityRunRef, CapabilityType


def execute_capability_and_build_ref(
    capability: CapabilityType,
    run_kwargs: dict[str, Any],
) -> CapabilityRunRef:
    """Run a submitted capability, persist analytics, and finalize its report."""
    # TODO: Future work should support a remote/shared cache backend
    # (for example object storage) that workers can read from. At that point,
    # worker execution can safely opt into cache usage.
    prepared_kwargs = prepare_job_submission_run_kwargs(run_kwargs)

    report_threshold = float(prepared_kwargs.pop("report_threshold", 0.5))
    raw_store_config = prepared_kwargs.pop("_analytics_store")
    raw_artifact_store_config = prepared_kwargs.pop("_artifact_store", None)
    raw_provenance = prepared_kwargs.pop("_provenance", None)
    store_config = AnalyticsStoreConfig.model_validate(raw_store_config)
    if raw_artifact_store_config is None:
        raise RuntimeError("artifact_store configuration is required for Ray workers")
    artifact_store_config = resolve_artifact_store_config(raw_artifact_store_config)

    run = capability.run(**prepared_kwargs)

    store = build_analytics_store(store_config)
    provenance = Provenance.from_optional(raw_provenance).merge({"completed_at": datetime.now(timezone.utc)})
    store_uri = write_run_and_get_store_uri(store, run, provenance=provenance)

    return build_capability_run_ref(
        run,
        store_uri=store_uri,
        report_threshold=report_threshold,
        artifact_uri=artifact_store_config.uri,
        artifact_storage_options=artifact_store_config.storage_options,
        artifact_scope=provenance.job_id,
    )
