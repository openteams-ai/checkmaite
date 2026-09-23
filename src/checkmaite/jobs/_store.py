from __future__ import annotations

import os
from glob import has_magic
from typing import TYPE_CHECKING, Annotated, Any, Literal
from urllib.parse import urlsplit

from fsspec.utils import get_protocol
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend, ProvenanceLike

if TYPE_CHECKING:
    from checkmaite.core.capability_core import CapabilityRunBase


class AnalyticsStoreConfig(BaseModel):
    """Configuration describing where job workers persist structured analytics records."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["parquet"] = "parquet"
    uri: str
    storage_options: dict[str, Any] = Field(default_factory=dict)


LOCAL_ARTIFACT_STORE_PROTOCOLS = frozenset({"file", "local"})
REMOTE_ARTIFACT_STORE_PROTOCOLS = frozenset({"s3", "s3a", "gs", "gcs", "abfs", "az"})
SUPPORTED_ARTIFACT_STORE_PROTOCOLS = LOCAL_ARTIFACT_STORE_PROTOCOLS | REMOTE_ARTIFACT_STORE_PROTOCOLS


class ArtifactStoreConfig(BaseModel):
    """Configuration for a supported durable report-artifact filesystem."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    uri: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    storage_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("uri")
    @classmethod
    def _require_concrete_uri_prefix(cls, uri: str) -> str:
        parsed = urlsplit(uri)
        if has_magic(parsed.path):
            raise ValueError("artifact store uri path must be a concrete prefix without glob patterns")
        if parsed.query:
            raise ValueError("artifact store uri must not contain query credentials; use storage_options instead")
        protocol = get_protocol(uri)
        if protocol not in SUPPORTED_ARTIFACT_STORE_PROTOCOLS:
            supported = ", ".join(sorted(SUPPORTED_ARTIFACT_STORE_PROTOCOLS))
            raise ValueError(f"unsupported artifact store protocol {protocol!r}; supported protocols: {supported}")
        if protocol in LOCAL_ARTIFACT_STORE_PROTOCOLS:
            local_path = parsed.path if parsed.scheme else uri
            if not os.path.isabs(local_path):
                raise ValueError("local artifact store uri must use an absolute path shared by every Ray node")
        return uri


def resolve_artifact_store_config(
    config: ArtifactStoreConfig | dict[str, Any],
) -> ArtifactStoreConfig:
    """Validate and detach artifact configuration from caller-owned mutable data."""
    return ArtifactStoreConfig.model_validate(config).model_copy(deep=True)


def build_analytics_store(config: AnalyticsStoreConfig | dict[str, Any]) -> AnalyticsStore:
    """Build an analytics store from explicit client-provided configuration."""
    resolved = AnalyticsStoreConfig.model_validate(config)

    if resolved.backend == "parquet":
        return AnalyticsStore(
            ParquetBackend(
                resolved.uri,
                storage_options=resolved.storage_options,
            )
        )

    raise ValueError(f"Unsupported analytics backend {resolved.backend!r}")


def write_run_and_get_store_uri(
    store: AnalyticsStore,
    run: CapabilityRunBase[Any, Any],
    *,
    provenance: ProvenanceLike | None = None,
) -> str | None:
    """Persist a run and return its payload URI, or ``None`` for an empty result."""
    receipt = store.write_with_receipt([run], provenance=provenance)

    store_uri = receipt.resolve_run_uri(run.run_uid)
    if store_uri is not None:
        return store_uri

    # No new payload row may be written for this run_uid when capability records
    # are deduplicated across calls. In that case, infer the persisted location
    # from existing analytics-store metadata. A valid run can also extract no
    # analytics rows (for example, XAITK when a model returns no detections); it
    # has no payload URI but still has a report and is a successful result.
    try:
        return store.get_run_uri(run.run_uid)
    except ValueError:
        if not run.extract():
            return None
        raise
