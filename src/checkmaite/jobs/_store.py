from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend

if TYPE_CHECKING:
    from checkmaite.core.capability_core import CapabilityRunBase


class AnalyticsStoreConfig(BaseModel):
    """Configuration describing where job workers persist analytics records."""

    backend: Literal["parquet"] = "parquet"
    uri: str
    storage_options: dict[str, Any] = Field(default_factory=dict)


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
) -> str:
    """Persist a run in the analytics store and return a concrete payload URI."""
    receipt = store.write_with_receipt([run])

    store_uri = receipt.resolve_run_uri(run.run_uid)
    if store_uri is not None:
        return store_uri

    # No new payload row may be written for this run_uid when capability records
    # are deduplicated across calls. In that case, infer the persisted location
    # from existing analytics-store metadata.
    return store.get_run_uri(run.run_uid)
