from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from checkmaite.core.analytics_store import AnalyticsStore, Provenance, StorageWriteReceipt
from checkmaite.core.analytics_store import _provenance as provenance_module
from checkmaite.jobs._store import AnalyticsStoreConfig, build_analytics_store, write_run_and_get_store_uri
from tests.test_jobs.fakes import EmptyTinyCapability, TinyCapability, TinyConfig, TinyDatasetCapability


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


def test_write_run_and_get_store_uri_returns_none_for_empty_analytics(tmp_path: Path) -> None:
    store = build_analytics_store({"backend": "parquet", "uri": str(tmp_path / "store")})
    run = EmptyTinyCapability().run(config=TinyConfig(text="empty"), use_cache=False)

    assert write_run_and_get_store_uri(store, run) is None
    assert store.list_tables() == []


def test_write_run_and_get_store_uri_falls_back_to_existing_run_uri(tmp_path: Path) -> None:
    store = build_analytics_store({"backend": "parquet", "uri": str(tmp_path / "store")})
    capability = TinyCapability()
    run = capability.run(config=TinyConfig(text="deduped"), use_cache=False)

    first_uri = write_run_and_get_store_uri(store, run)
    second_uri = write_run_and_get_store_uri(store, run)

    assert second_uri == first_uri


def test_write_run_and_get_store_uri_reraises_for_missing_nonempty_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = build_analytics_store({"backend": "parquet", "uri": str(tmp_path / "store")})
    run = TinyCapability().run(config=TinyConfig(text="missing"), use_cache=False)
    monkeypatch.setattr(store, "write_with_receipt", Mock(return_value=StorageWriteReceipt()))
    monkeypatch.setattr(store, "get_run_uri", Mock(side_effect=ValueError("missing payload")))

    with pytest.raises(ValueError, match="missing payload"):
        write_run_and_get_store_uri(store, run)


def test_job_store_write_uses_submitted_provenance_not_worker_process_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_ic_dataset_default,
) -> None:
    monkeypatch.setattr(provenance_module, "_ENV_PROVENANCE", Provenance(user_id="worker-user"))
    monkeypatch.setattr(provenance_module, "_FROZEN_PROVENANCE_FIELDS", frozenset({"user_id"}))

    store = build_analytics_store({"backend": "parquet", "uri": str(tmp_path / "store")})
    run = TinyDatasetCapability().run(
        datasets=[fake_ic_dataset_default],
        config=TinyConfig(text="client-provenance"),
        use_cache=False,
    )

    write_run_and_get_store_uri(store, run, provenance={"user_id": "client-user"})

    result = store.query_sql("SELECT DISTINCT user_id FROM runs")
    assert result["user_id"].to_list() == ["client-user"]
