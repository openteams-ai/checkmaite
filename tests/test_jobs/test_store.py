from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from checkmaite.core.analytics_store import AnalyticsStore
from checkmaite.jobs._store import AnalyticsStoreConfig, build_analytics_store, write_run_and_get_store_uri
from tests.test_jobs.fakes import TinyCapability, TinyConfig


def test_build_analytics_store_accepts_config_dict_and_config_model(tmp_path: Path) -> None:
    from_dict = build_analytics_store({"backend": "parquet", "uri": str(tmp_path / "dict-store")})
    from_model = build_analytics_store(AnalyticsStoreConfig(uri=str(tmp_path / "model-store")))

    assert isinstance(from_dict, AnalyticsStore)
    assert isinstance(from_model, AnalyticsStore)


def test_build_analytics_store_rejects_invalid_config(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        build_analytics_store({"backend": "unsupported", "uri": str(tmp_path / "store")})


def test_write_run_and_get_store_uri_returns_concrete_payload_uri(tmp_path: Path) -> None:
    store = build_analytics_store({"backend": "parquet", "uri": str(tmp_path / "store")})
    run = TinyCapability().run(config=TinyConfig(text="persisted"), use_cache=False)

    store_uri = write_run_and_get_store_uri(store, run)

    assert store_uri.endswith(".parquet")
    assert "tiny_jobs" in store_uri
    assert store.get_run_uri(run.run_uid) == store_uri


def test_write_run_and_get_store_uri_falls_back_to_existing_run_uri(tmp_path: Path) -> None:
    store = build_analytics_store({"backend": "parquet", "uri": str(tmp_path / "store")})
    capability = TinyCapability()
    run = capability.run(config=TinyConfig(text="deduped"), use_cache=False)

    first_uri = write_run_and_get_store_uri(store, run)
    second_uri = write_run_and_get_store_uri(store, run)

    assert second_uri == first_uri
