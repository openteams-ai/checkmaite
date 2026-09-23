from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from checkmaite.core.analytics_store import AnalyticsStore, Provenance, StorageWriteReceipt
from checkmaite.core.analytics_store import _provenance as provenance_module
from checkmaite.jobs._store import (
    AnalyticsStoreConfig,
    ArtifactStoreConfig,
    build_analytics_store,
    resolve_artifact_store_config,
    write_run_and_get_store_uri,
)
from checkmaite.jobs.backends.ray import RayJobBackend
from checkmaite.jobs.backends.ray_simple import RaySimpleJobBackend
from tests.test_jobs.fakes import EmptyTinyCapability, TinyCapability, TinyConfig, TinyDatasetCapability


def test_artifact_store_config_is_independent_from_analytics_store(tmp_path: Path) -> None:
    analytics = AnalyticsStoreConfig(uri="memory://analytics")
    artifact_uri = str(tmp_path / "artifacts")
    artifacts = ArtifactStoreConfig(uri=artifact_uri, storage_options={"auto_mkdir": True})

    assert analytics.model_dump() == {
        "backend": "parquet",
        "uri": "memory://analytics",
        "storage_options": {},
    }
    assert artifacts.model_dump() == {
        "uri": artifact_uri,
        "storage_options": {"auto_mkdir": True},
    }
    with pytest.raises(ValidationError, match="artifact_uri"):
        AnalyticsStoreConfig.model_validate({"uri": "memory://analytics", "artifact_uri": "memory://artifacts"})


@pytest.mark.parametrize("uri", ["", "   "])
def test_artifact_store_config_rejects_empty_uri(uri: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactStoreConfig(uri=uri)


@pytest.mark.parametrize(
    "uri",
    [
        "file:///tmp/reports*",
        "s3://bucket/reports[production]",
    ],
)
def test_artifact_store_config_rejects_glob_prefix(uri: str) -> None:
    with pytest.raises(ValidationError, match="concrete prefix without glob patterns"):
        ArtifactStoreConfig(uri=uri)


def test_artifact_store_config_requires_query_credentials_in_storage_options() -> None:
    with pytest.raises(ValidationError, match="use storage_options instead"):
        ArtifactStoreConfig(uri="abfs://container/reports?sv=1&sig=secret")

    config = ArtifactStoreConfig(
        uri="abfs://container/reports",
        storage_options={"sas_token": "sv=1&sig=secret"},
    )
    assert config.storage_options == {"sas_token": "sv=1&sig=secret"}


def test_artifact_store_config_is_frozen_and_resolved_as_a_deep_snapshot(tmp_path: Path) -> None:
    config = ArtifactStoreConfig(
        uri=str(tmp_path / "artifacts"),
        storage_options={"client_kwargs": {"endpoint_url": "original"}},
    )

    with pytest.raises(ValidationError):
        config.uri = str(tmp_path / "changed")

    resolved = resolve_artifact_store_config(config)
    config.storage_options["client_kwargs"]["endpoint_url"] = "changed"

    assert resolved is not config
    assert resolved.storage_options == {"client_kwargs": {"endpoint_url": "original"}}


@pytest.mark.parametrize(
    "uri",
    ["memory://reports", "hdfs://namenode/reports", "https://example.test/reports", "adl://account/reports"],
)
def test_artifact_store_config_rejects_unsupported_protocol(uri: str) -> None:
    with pytest.raises(ValidationError, match="unsupported artifact store protocol"):
        ArtifactStoreConfig(uri=uri)


@pytest.mark.parametrize("uri", ["./report-artifacts", "report-artifacts", "file:report-artifacts"])
def test_artifact_store_config_rejects_relative_local_path(uri: str) -> None:
    with pytest.raises(ValidationError, match="absolute path shared by every Ray node"):
        ArtifactStoreConfig(uri=uri)


@pytest.mark.parametrize(
    ("backend_type", "extra_kwargs"),
    [
        (RayJobBackend, {"idempotency_scope": "scope"}),
        (RaySimpleJobBackend, {}),
    ],
)
def test_job_backends_require_artifact_store_before_initializing_ray(
    backend_type,
    extra_kwargs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_init = Mock()
    ray_shutdown = Mock()
    monkeypatch.setattr("ray.is_initialized", lambda: True)
    monkeypatch.setattr("ray.init", ray_init)
    monkeypatch.setattr("ray.shutdown", ray_shutdown)

    with pytest.raises(TypeError, match="artifact_store"):
        backend_type(
            analytics_store={"uri": "memory://analytics"},
            force_reinit=True,
            **extra_kwargs,
        )

    ray_init.assert_not_called()
    ray_shutdown.assert_not_called()


@pytest.mark.parametrize(
    ("backend_type", "extra_kwargs"),
    [
        (RayJobBackend, {"idempotency_scope": "scope"}),
        (RaySimpleJobBackend, {}),
    ],
)
@pytest.mark.parametrize(
    "invalid_uri",
    ["  ", "memory://reports", "./report-artifacts", "file:///tmp/reports*", "hdfs://namenode/reports"],
)
def test_job_backends_reject_invalid_artifact_uri_before_initializing_ray(
    backend_type,
    extra_kwargs,
    invalid_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_init = Mock()
    ray_shutdown = Mock()
    invalid_config = ArtifactStoreConfig.model_construct(uri=invalid_uri, storage_options={})
    monkeypatch.setattr("ray.is_initialized", lambda: True)
    monkeypatch.setattr("ray.init", ray_init)
    monkeypatch.setattr("ray.shutdown", ray_shutdown)

    with pytest.raises(ValidationError):
        backend_type(
            analytics_store={"uri": "memory://analytics"},
            artifact_store=invalid_config,
            force_reinit=True,
            **extra_kwargs,
        )

    ray_init.assert_not_called()
    ray_shutdown.assert_not_called()


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
