from __future__ import annotations

import os
from typing import Any

import ray
from ray.actor import ActorHandle

from checkmaite.jobs._worker import execute_capability_and_build_ref
from checkmaite.jobs.protocol import CapabilityRunRef, CapabilityType

from .controller import WorkerStartupUnavailableError


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

    return execute_capability_and_build_ref(capability, run_kwargs)
