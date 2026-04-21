from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from checkmaite.jobs._store import AnalyticsStoreConfig
from checkmaite.jobs.ray_backend import RayBackend

if TYPE_CHECKING:
    import maite.protocols.generic as gen

    from checkmaite.core.capability_core import CapabilityConfigBase
    from checkmaite.jobs.protocol import Backend, CapabilityRunRef, CapabilityType, Job

_active_backend: Backend | None = None


def _require_backend() -> Backend:
    if _active_backend is None:
        raise RuntimeError(
            "No active jobs backend. Call configure_backend(..., analytics_store=...) before submitting jobs."
        )

    return _active_backend


def configure_backend(
    kind: str = "ray",
    *,
    analytics_store: AnalyticsStoreConfig | dict[str, Any],
    **kwargs: Any,
) -> None:
    """Configure the active job backend.

    Reconfiguration is non-blocking for currently tracked jobs by default.
    For Ray, pass ``force_reinit=True`` to explicitly disconnect/reconnect
    the runtime when you need new address/runtime_env settings to apply.

    ``analytics_store`` is required and is forwarded to all worker tasks so
    writes target an explicit client-chosen durable location.
    """
    global _active_backend

    if _active_backend is not None:
        _active_backend.shutdown(wait=False)

    if kind == "ray":
        _active_backend = RayBackend(analytics_store=analytics_store, **kwargs)
        return

    raise ValueError(f"Unknown backend: {kind!r}")


def submit_capability(
    capability: CapabilityType,
    models: Sequence[gen.Model[Any, Any]] | None = None,
    datasets: Sequence[gen.Dataset[Any, Any, Any]] | None = None,
    metrics: Sequence[gen.Metric[Any, Any]] | None = None,
    config: CapabilityConfigBase | None = None,
    use_cache: bool = True,
    **kwargs: Any,
) -> Job[CapabilityRunRef]:
    """Submit a capability run as an asynchronous job."""
    run_kwargs: dict[str, Any] = {
        "models": models,
        "datasets": datasets,
        "metrics": metrics,
        "config": config,
        "use_cache": use_cache,
    }
    run_kwargs.update(kwargs)

    return _require_backend().submit_capability(capability, **run_kwargs)


def list_jobs() -> Sequence[Job[CapabilityRunRef]]:
    """List jobs tracked by the active backend."""
    if _active_backend is None:
        return []
    return _active_backend.list_jobs()


def get_job(job_id: str) -> Job[CapabilityRunRef]:
    """Fetch a tracked job by ID."""
    return _require_backend().get_job(job_id)


def shutdown_backend(wait: bool = True) -> None:
    """Shutdown the active backend, if configured."""
    global _active_backend

    if _active_backend is None:
        return

    _active_backend.shutdown(wait=wait)
    _active_backend = None
