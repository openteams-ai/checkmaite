from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend
from checkmaite.core.report import ArtifactReport
from checkmaite.jobs.backends.ray_simple.job_backend import _execute_capability_ref
from tests.test_jobs.fakes import (
    EmptyTinyCapability,
    OversizedReportTinyCapability,
    TinyCapability,
    TinyConfig,
    TinyDatasetCapability,
)

TEST_ARTIFACT_STORE_URI = str((Path.cwd() / ".checkmaite-test-artifacts").resolve())


def test_execute_capability_ref_runs_capability_writes_store_and_returns_reference(tmp_path: Path) -> None:
    marker = tmp_path / "worker-started.txt"

    ref = _execute_capability_ref(
        TinyCapability(),
        {
            "config": TinyConfig(text="worker", start_marker_path=str(marker)),
            "use_cache": False,
            "report_threshold": 0.75,
            "_analytics_store": {"backend": "parquet", "uri": str(tmp_path / "store")},
            "_artifact_store": {"uri": TEST_ARTIFACT_STORE_URI},
        },
    )

    assert marker.read_text() == "started"
    assert ref.capability_id == TinyCapability().id
    assert ref.store_uri.endswith(".parquet")
    assert ref.report.model_dump() == {
        "kind": "inline_text",
        "media_type": "text/markdown",
        "content": "worker:0.75",
        "filename": "tiny-report.md",
    }


def test_execute_capability_ref_requires_artifact_store_before_running_capability(tmp_path: Path) -> None:
    marker = tmp_path / "worker-started.txt"

    with pytest.raises(RuntimeError, match="artifact_store configuration is required"):
        _execute_capability_ref(
            OversizedReportTinyCapability(),
            {
                "config": TinyConfig(text="oversized", start_marker_path=str(marker)),
                "use_cache": False,
                "report_threshold": 0.5,
                "_analytics_store": {"backend": "parquet", "uri": str(tmp_path / "store")},
            },
        )

    assert not marker.exists()
    assert not (tmp_path / "store").exists()


def test_execute_capability_ref_rejects_process_local_artifact_store_before_running_capability(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "worker-started.txt"

    with pytest.raises(ValidationError, match="unsupported artifact store protocol 'memory'"):
        _execute_capability_ref(
            OversizedReportTinyCapability(),
            {
                "config": TinyConfig(text="oversized", start_marker_path=str(marker)),
                "use_cache": False,
                "_analytics_store": {"backend": "parquet", "uri": str(tmp_path / "store")},
                "_artifact_store": {"uri": "memory://reports"},
            },
        )

    assert not marker.exists()
    assert not (tmp_path / "store").exists()


def test_execute_capability_ref_publishes_oversized_report_artifact(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "report-artifacts"
    ref = _execute_capability_ref(
        OversizedReportTinyCapability(),
        {
            "config": TinyConfig(text="oversized"),
            "use_cache": False,
            "report_threshold": 0.5,
            "_analytics_store": {
                "backend": "parquet",
                "uri": str(tmp_path / "store"),
            },
            "_artifact_store": {"uri": str(artifact_dir)},
            "_provenance": {"job_id": "job-1"},
        },
    )

    assert isinstance(ref.report, ArtifactReport)
    assert Path(ref.report.uri.removeprefix("file://")).is_file()
    assert str(artifact_dir) in ref.report.uri


def test_execute_capability_ref_fails_when_oversized_report_publication_fails(tmp_path: Path) -> None:
    blocked_artifact_path = tmp_path / "blocked-artifact-path"
    blocked_artifact_path.write_text("not a directory")

    with pytest.raises(RuntimeError, match="configured artifact_store"):
        _execute_capability_ref(
            OversizedReportTinyCapability(),
            {
                "config": TinyConfig(text="oversized"),
                "use_cache": False,
                "_analytics_store": {"backend": "parquet", "uri": str(tmp_path / "store")},
                "_artifact_store": {"uri": str(blocked_artifact_path)},
                "_provenance": {"job_id": "job-1"},
            },
        )


def test_execute_capability_ref_completes_with_empty_analytics(tmp_path: Path) -> None:
    ref = _execute_capability_ref(
        EmptyTinyCapability(),
        {
            "config": TinyConfig(text="no rows"),
            "use_cache": False,
            "report_threshold": 0.5,
            "_analytics_store": {"backend": "parquet", "uri": str(tmp_path / "store")},
            "_artifact_store": {"uri": TEST_ARTIFACT_STORE_URI},
        },
    )

    assert ref.store_uri is None
    assert ref.report.content == "no rows:0.5"


def test_execute_capability_ref_writes_provenance_to_runs_table(tmp_path: Path, fake_ic_dataset_default) -> None:
    store_path = tmp_path / "store"

    _execute_capability_ref(
        TinyDatasetCapability(),
        {
            "datasets": [fake_ic_dataset_default],
            "config": TinyConfig(text="worker"),
            "use_cache": False,
            "_analytics_store": {"backend": "parquet", "uri": str(store_path)},
            "_artifact_store": {"uri": TEST_ARTIFACT_STORE_URI},
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
