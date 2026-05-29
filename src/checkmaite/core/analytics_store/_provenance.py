from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, TypeAlias

import pydantic


class Provenance(pydantic.BaseModel):
    """Flat execution metadata stored with analytics run history.

    All fields are optional so callers can provide only the context they know.
    The analytics store persists these values on ``RunRecord`` rows, where they
    remain scalar and SQL-queryable.

    Static identity/environment fields usually come from
    ``configure_provenance(...)`` or ``CHECKMAITE_PROVENANCE_*`` environment
    variables:

    - ``user_id``: user or service principal, e.g. ``"alice@company.com"``.
    - ``workspace_id``: project/workspace/scope, e.g. ``"ml-team-a"``.
    - ``environment``: runtime label, e.g. ``"databricks-prod"``.
    - ``executor``: logical executor, e.g. ``"notebook"`` or ``"ray"``.
    - ``cluster_id``: compute cluster identifier, e.g. ``"cluster-42"``.
    - ``request_id``: trace/request/workflow ID, e.g. ``"req-abc123"``.

    Per-run-event fields usually come from job backends or explicit
    ``store.write(..., provenance=...)`` calls:

    - ``job_id``: backend job identifier, e.g. ``"job-abc123"``.
    - ``backend``: job backend name, e.g. ``"ray"`` or ``"ray-simple"``.
    - ``submitted_at``: timezone-aware timestamp for when the job/run event was submitted.
    - ``completed_at``: timezone-aware timestamp for when the job/run event completed.
    - ``run_event_id``: history-event key. Job backends set this to ``job_id``;
      local writes leave it ``None`` unless callers opt into separate history
      rows by providing an explicit value such as ``"notebook-cell-7"``.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    user_id: str | None = None
    workspace_id: str | None = None
    job_id: str | None = None
    backend: str | None = None
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    environment: str | None = None
    executor: str | None = None
    cluster_id: str | None = None
    request_id: str | None = None
    run_event_id: str | None = None

    @pydantic.field_validator("submitted_at", "completed_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware timestamps and normalize them to UTC."""
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("submitted_at and completed_at must be timezone-aware datetimes")
        return value.astimezone(timezone.utc)

    @classmethod
    def from_optional(cls, provenance: ProvenanceLike | None) -> Provenance:
        """Normalize an optional provenance payload."""
        if provenance is None:
            return cls()
        if isinstance(provenance, cls):
            return provenance
        return cls.model_validate(provenance)

    def merge(self, *overrides: ProvenanceLike | None) -> Provenance:
        """Return this provenance overlaid by each non-None override.

        Later values win when they are not ``None``. ``None`` values are treated
        as unspecified so dynamic metadata can be merged without clearing static
        defaults accidentally.
        """
        merged = self.model_dump(mode="python", exclude_none=True)
        for override in overrides:
            override_provenance = Provenance.from_optional(override)
            merged.update(override_provenance.model_dump(mode="python", exclude_none=True))
        return Provenance.model_validate(merged)


ProvenanceLike: TypeAlias = Provenance | Mapping[str, Any]


_STATIC_PROVENANCE_ENV_VARS: dict[str, str] = {
    "user_id": "CHECKMAITE_PROVENANCE_USER_ID",
    "workspace_id": "CHECKMAITE_PROVENANCE_WORKSPACE_ID",
    "environment": "CHECKMAITE_PROVENANCE_ENVIRONMENT",
    "executor": "CHECKMAITE_PROVENANCE_EXECUTOR",
    "cluster_id": "CHECKMAITE_PROVENANCE_CLUSTER_ID",
    "request_id": "CHECKMAITE_PROVENANCE_REQUEST_ID",
}


def _load_environment_provenance() -> tuple[Provenance, frozenset[str]]:
    """Load static provenance defaults from the process environment."""
    values: dict[str, str] = {}
    frozen_fields: set[str] = set()

    for field_name, env_var_name in _STATIC_PROVENANCE_ENV_VARS.items():
        value = os.environ.get(env_var_name)
        if value:
            values[field_name] = value
            frozen_fields.add(field_name)

    return Provenance.model_validate(values), frozenset(frozen_fields)


_ENV_PROVENANCE, _FROZEN_PROVENANCE_FIELDS = _load_environment_provenance()
_DEFAULT_PROVENANCE = _ENV_PROVENANCE
_PROVENANCE_LOCK = threading.RLock()


def _apply_frozen_provenance(provenance: Provenance) -> Provenance:
    """Return provenance with env-frozen fields enforced."""
    if not _FROZEN_PROVENANCE_FIELDS:
        return provenance

    values = provenance.model_dump(mode="python", exclude_none=True)
    for field_name in _FROZEN_PROVENANCE_FIELDS:
        frozen_value = getattr(_ENV_PROVENANCE, field_name)
        current_value = getattr(provenance, field_name)
        if current_value is not None and current_value != frozen_value:
            env_var_name = _STATIC_PROVENANCE_ENV_VARS[field_name]
            raise ValueError(
                f"Provenance field {field_name!r} is set by environment variable "
                f"{env_var_name!r} and cannot be changed."
            )
        values[field_name] = frozen_value

    return Provenance.model_validate(values)


def configure_provenance(
    *,
    user_id: str | None = None,
    workspace_id: str | None = None,
    environment: str | None = None,
    executor: str | None = None,
    cluster_id: str | None = None,
    request_id: str | None = None,
) -> Provenance:
    """Replace process-wide default provenance values.

    Pass any subset of static identity/environment fields to configure
    session defaults for callers that use ``get_provenance_defaults()``. Calling
    with no arguments resets mutable defaults while preserving
    environment-provided frozen values.

    Per-run-event fields such as ``job_id``, ``backend``, timestamps, and
    ``run_event_id`` are intentionally excluded from process-wide defaults;
    pass them through ``AnalyticsStore.write(..., provenance=...)`` or let job
    backends populate them.
    """
    global _DEFAULT_PROVENANCE

    configured = _apply_frozen_provenance(
        Provenance(
            user_id=user_id,
            workspace_id=workspace_id,
            environment=environment,
            executor=executor,
            cluster_id=cluster_id,
            request_id=request_id,
        )
    )
    with _PROVENANCE_LOCK:
        _DEFAULT_PROVENANCE = configured
        return configured.model_copy()


def get_provenance_defaults() -> Provenance:
    """Return a copy of the current process-wide provenance defaults."""
    with _PROVENANCE_LOCK:
        return _DEFAULT_PROVENANCE.model_copy()


def reset_provenance() -> Provenance:
    """Clear mutable process-wide provenance defaults while preserving env-provided frozen values."""
    return configure_provenance()
