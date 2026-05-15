from __future__ import annotations

from pathlib import Path

from checkmaite.core.analytics_store import AnalyticsStore
from checkmaite.jobs.backends.ray_simple.job_backend import (
    _collect_md_report,
    _execute_capability_ref,
    _get_worker_store,
    _write_run_and_collect_store_metadata,
)
from tests.test_jobs.fakes import TinyCapability, TinyConfig


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
            "use_cache": True,
            "report_threshold": 0.75,
            "_analytics_store": {"backend": "parquet", "uri": str(tmp_path / "store")},
        },
    )

    assert marker.read_text() == "started"
    assert ref.capability_id == TinyCapability().id
    assert ref.store_uri.endswith(".parquet")
    assert ref.summary["md_report"] == "worker:0.75"
