from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import ray
from ray.actor import ActorHandle

from checkmaite.core.analytics_store import AnalyticsStore, Provenance, ProvenanceLike
from checkmaite.core.capability_core import CapabilityRunBase
from checkmaite.jobs._result import build_capability_run_ref
from checkmaite.jobs._store import AnalyticsStoreConfig, build_analytics_store, write_run_and_get_store_uri
from checkmaite.jobs._submission import prepare_job_submission_run_kwargs
from checkmaite.jobs.protocol import CapabilityRunRef, CapabilityType

from .controller import WorkerStartupUnavailableError


def _get_worker_store(store_config: AnalyticsStoreConfig | dict[str, Any]) -> AnalyticsStore:
    return build_analytics_store(store_config)


def _write_run_and_collect_store_metadata(
    store: AnalyticsStore,
    run: CapabilityRunBase[Any, Any],
    *,
    provenance: ProvenanceLike | None = None,
) -> str | None:
    return write_run_and_get_store_uri(store, run, provenance=provenance)


def execute_capability_ref(
    capability: CapabilityType,
    run_kwargs: dict[str, Any],
    controller: ActorHandle | None = None,
    controller_token: str | None = None,
    startup_timeout_s: float = 5.0,
) -> CapabilityRunRef:
    """Notify the controller that scheduling finished, then execute a capability."""
    if controller is not None and controller_token is not None:
        runtime_context = ray.get_runtime_context()
        worker_info = {
            "node_id": str(runtime_context.get_node_id()),
            "pod_name": os.getenv("HOSTNAME"),
            "kubernetes_node_name": os.getenv("CHECKMAITE_KUBERNETES_NODE_NAME"),
        }
        try:
            decision = ray.get(
                controller.worker_started.remote(controller_token, worker_info),
                timeout=float(startup_timeout_s),
            )
        except Exception as exc:
            raise WorkerStartupUnavailableError("could not verify worker startup") from exc
        if decision is None:
            raise WorkerStartupUnavailableError("job controller could not verify worker startup")
        if not decision:
            raise RuntimeError("job controller rejected worker startup acknowledgement")

    # TODO: Future work should support a remote/shared cache backend
    # (for example object storage) that workers can read from. At that point,
    # worker execution can safely opt into cache usage.
    run_kwargs = prepare_job_submission_run_kwargs(run_kwargs)

    report_threshold = float(run_kwargs.pop("report_threshold", 0.5))
    raw_store_config = run_kwargs.pop("_analytics_store")
    raw_provenance = run_kwargs.pop("_provenance", None)

    run = capability.run(**run_kwargs)

    store = _get_worker_store(raw_store_config)
    provenance = Provenance.from_optional(raw_provenance).merge({"completed_at": datetime.now(timezone.utc)})
    store_uri = _write_run_and_collect_store_metadata(store, run, provenance=provenance)

    return build_capability_run_ref(
        run,
        store_uri=store_uri,
        report_threshold=report_threshold,
    )
