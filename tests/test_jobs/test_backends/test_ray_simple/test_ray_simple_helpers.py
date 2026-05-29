from __future__ import annotations

from pathlib import Path

import pytest

from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend
from checkmaite.jobs.backends.ray_simple.job_backend import (
    _collect_md_report,
    _execute_capability_ref,
    _get_worker_store,
    _write_run_and_collect_store_metadata,
)
from tests.test_jobs.fakes import TinyCapability, TinyConfig, TinyDatasetCapability


class NoReportRun:
    def collect_md_report(self, threshold: float):
        raise NotImplementedError


def test_collect_md_report_returns_empty_list_when_report_is_not_implemented() -> None:
    assert _collect_md_report(NoReportRun(), threshold=0.5) == []  # type: ignore[arg-type]


def test_get_worker_store_builds_analytics_store(tmp_path: Path) -> None:
    store = _get_worker_store({"backend": "parquet", "uri": str(tmp_path / "store")})

    assert isinstance(store, AnalyticsStore)


def test_write_run_and_collect_store_metadata_returns_payload_uri(tmp_path: Path) -> None:
    store = _get_worker_store({"backend": "parquet", "uri": str(tmp_path / "store")})
    run = TinyCapability().run(config=TinyConfig(text="metadata"), use_cache=False)

    store_uri = _write_run_and_collect_store_metadata(store, run)

    assert store_uri.endswith(".parquet")
    assert "tiny_jobs" in store_uri
    assert store.get_run_uri(run.run_uid) == store_uri


def test_execute_capability_ref_runs_capability_writes_store_and_returns_reference(tmp_path: Path) -> None:
    marker = tmp_path / "worker-started.txt"

    ref = _execute_capability_ref(
        TinyCapability(),
        {
            "config": TinyConfig(text="worker", start_marker_path=str(marker)),
            "use_cache": False,
            "report_threshold": 0.75,
            "_analytics_store": {"backend": "parquet", "uri": str(tmp_path / "store")},
        },
    )

    assert marker.read_text() == "started"
    assert ref.capability_id == TinyCapability().id
    assert ref.store_uri.endswith(".parquet")
    assert ref.summary["md_report"] == "worker:0.75"


def test_execute_capability_ref_writes_provenance_to_runs_table(tmp_path: Path, fake_ic_dataset_default) -> None:
    store_path = tmp_path / "store"

    _execute_capability_ref(
        TinyDatasetCapability(),
        {
            "datasets": [fake_ic_dataset_default],
            "config": TinyConfig(text="worker"),
            "use_cache": False,
            "_analytics_store": {"backend": "parquet", "uri": str(store_path)},
            "_provenance": {
                "user_id": "alice",
                "job_id": "job-1",
                "backend": "ray-simple",
                "run_event_id": "job-1",
            },
        },
    )

    result = AnalyticsStore(ParquetBackend(str(store_path))).query_sql(
        "SELECT user_id, job_id, backend, completed_at, run_event_id FROM runs"
    )

    assert result.to_dicts()[0]["user_id"] == "alice"
    assert result.to_dicts()[0]["job_id"] == "job-1"
    assert result.to_dicts()[0]["backend"] == "ray-simple"
    assert result.to_dicts()[0]["completed_at"] is not None
    assert result.to_dicts()[0]["run_event_id"] == "job-1"


def test_execute_capability_ref_rejects_cache_in_job_submission(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="use_cache=True is not supported"):
        _execute_capability_ref(
            TinyCapability(),
            {
                "config": TinyConfig(text="worker"),
                "use_cache": True,
                "_analytics_store": {"backend": "parquet", "uri": str(tmp_path / "store")},
            },
        )
