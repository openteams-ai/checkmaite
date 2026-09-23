"""Tests for shared capability-job result construction."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import pytest
import ray.cloudpickle

from checkmaite.core.report import ArtifactReport, InlineTextReport
from checkmaite.jobs import CapabilityRunRef
from checkmaite.jobs._result import _artifact_path, _commit_staged_artifact, _publish_bytes, build_capability_run_ref
from tests.test_jobs.fakes import (
    ConfigurableReportTinyCapability,
    ConfigurableReportTinyConfig,
    OversizedReportTinyCapability,
    ReportlessTinyCapability,
    TinyCapability,
    TinyConfig,
)


def _run_with_artifact_report(uri: str):
    return ConfigurableReportTinyCapability().run(
        config=ConfigurableReportTinyConfig(
            text="reported",
            report_media_type="application/pdf",
            report_filename="report.pdf",
            report_uri=uri,
        ),
        use_cache=False,
    )


def test_commit_staged_artifact_uses_protocol_contract_instead_of_filesystem_type() -> None:
    class CustomObjectStore:
        def __init__(self) -> None:
            self.operations = []

        def cp_file(self, source, destination) -> None:
            self.operations.append(("copy", source, destination))

        def rm(self, path) -> None:
            self.operations.append(("remove", path))

    filesystem = CustomObjectStore()

    _commit_staged_artifact(filesystem, "temporary", "destination", protocol="s3")

    assert filesystem.operations == [
        ("copy", "temporary", "destination"),
        ("remove", "temporary"),
    ]


def test_commit_staged_artifact_rejects_unsupported_protocol() -> None:
    with pytest.raises(RuntimeError, match="artifact protocol 'hdfs'"):
        _commit_staged_artifact(object(), "temporary", "destination", protocol="hdfs")


def test_artifact_path_preserves_filesystem_root() -> None:
    path = _artifact_path(
        "/",
        run_uid="run-1",
        artifact_scope="job-1",
        filename="report.md",
        content=b"report",
    )

    assert path.startswith("/run-1/")


@pytest.mark.parametrize("filename", ["a?.md", "a*.md", "a[1].md"])
def test_publish_bytes_uses_glob_safe_keys(filename: str) -> None:
    uri = _publish_bytes(
        b"report",
        filename=filename,
        run_uid="glob-safe-run",
        artifact_scope=filename,
        artifact_uri="memory://checkmaite-report-tests/glob-safe",
        storage_options=None,
    )

    filesystem, path = fsspec.core.url_to_fs(uri)
    key = path.rsplit("/", 1)[-1]
    assert not set("*?[]").intersection(key)
    assert filename not in path
    assert filesystem.cat(path) == b"report"
    assert not any(".tmp-" in existing for existing in filesystem.find("/checkmaite-report-tests/glob-safe"))


@pytest.mark.parametrize("artifact_uri", ["memory://", "memory:///"])
def test_publish_bytes_preserves_scheme_only_remote_uri(artifact_uri: str) -> None:
    uri = _publish_bytes(
        b"scheme-only report",
        filename="report.md",
        run_uid="run-scheme-only",
        artifact_scope=None,
        artifact_uri=artifact_uri,
        storage_options=None,
    )

    assert uri.startswith("memory:///run-scheme-only/")
    filesystem, path = fsspec.core.url_to_fs(uri)
    assert filesystem.cat(path) == b"scheme-only report"


def test_publish_bytes_preserves_remote_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    filesystem = fsspec.filesystem("memory")

    def azure_url_to_fs(url, **storage_options):
        assert url == "abfs://container@account.dfs.core.windows.net/reports"
        assert storage_options == {"sas_token": "secret"}
        return filesystem, "/container/reports"

    monkeypatch.setattr(fsspec.core, "url_to_fs", azure_url_to_fs)

    uri = _publish_bytes(
        b"report",
        filename="report.md",
        run_uid="run-azure-authority",
        artifact_scope="job-azure-authority",
        artifact_uri="abfs://container@account.dfs.core.windows.net/reports",
        storage_options={"sas_token": "secret"},
    )

    assert uri.startswith("abfs://container@account.dfs.core.windows.net/reports/run-azure-authority/")
    assert "secret" not in uri
    published = filesystem.find("/container/reports/run-azure-authority")
    assert len(published) == 1
    assert filesystem.cat(published[0]) == b"report"


def test_publish_bytes_rejects_query_credentials() -> None:
    with pytest.raises(ValueError, match="storage_options"):
        _publish_bytes(
            b"report",
            filename="report.md",
            run_uid="run-query",
            artifact_scope=None,
            artifact_uri="abfs://container/reports?sig=secret",
            storage_options=None,
        )


def test_build_capability_run_ref_collects_typed_report() -> None:
    run = TinyCapability().run(config=TinyConfig(text="reported"), use_cache=False)

    ref = build_capability_run_ref(run, store_uri="memory://run", report_threshold=0.75)

    assert ref.report == InlineTextReport(
        media_type="text/markdown",
        content="reported:0.75",
        filename="tiny-report.md",
    )


def test_build_capability_run_ref_requires_artifact_store_for_oversized_inline_report() -> None:
    run = OversizedReportTinyCapability().run(config=TinyConfig(text="oversized"), use_cache=False)

    with pytest.raises(RuntimeError, match="configure artifact_store"):
        build_capability_run_ref(run, store_uri="memory://run", report_threshold=0.5)


def test_build_capability_run_ref_publishes_oversized_report() -> None:
    run = OversizedReportTinyCapability().run(config=TinyConfig(text="oversized"), use_cache=False)

    ref = build_capability_run_ref(
        run,
        store_uri="memory://run",
        report_threshold=0.5,
        artifact_uri="memory://checkmaite-report-tests/oversized",
    )

    assert isinstance(ref.report, ArtifactReport)
    filesystem, path = fsspec.core.url_to_fs(ref.report.uri)
    assert len(filesystem.cat(path)) > 256 * 1024
    assert run.run_uid in ref.report.uri


def test_build_capability_run_ref_fails_when_artifact_filesystem_is_unavailable() -> None:
    run = OversizedReportTinyCapability().run(config=TinyConfig(text="oversized"), use_cache=False)

    with pytest.raises(RuntimeError, match="configured artifact_store"):
        build_capability_run_ref(
            run,
            store_uri="memory://run",
            report_threshold=0.5,
            artifact_uri="unsupported-protocol://reports",
        )


def test_publish_bytes_cleans_up_interrupted_publication(monkeypatch) -> None:
    filesystem = fsspec.filesystem("memory")
    original_url_to_fs = fsspec.core.url_to_fs

    def memory_url_to_fs(url, **storage_options):
        _ = storage_options
        if url == "memory://interrupted-publication":
            return filesystem, "/interrupted-publication"
        return original_url_to_fs(url, **storage_options)

    def fail_commit(*_args, **_kwargs):
        raise OSError("interrupted")

    monkeypatch.setattr(fsspec.core, "url_to_fs", memory_url_to_fs)
    monkeypatch.setattr(filesystem, "cp_file", fail_commit)

    with pytest.raises(OSError, match="interrupted"):
        _publish_bytes(
            b"report",
            filename="a?.md",
            run_uid="run-a",
            artifact_scope="job-a",
            artifact_uri="memory://interrupted-publication",
            storage_options=None,
        )

    assert filesystem.find("/interrupted-publication") == []


def test_publish_bytes_reuses_verified_content_key() -> None:
    publish_kwargs = {
        "content": b"report",
        "filename": "report[]*?#%.md",
        "run_uid": "run-a",
        "artifact_scope": "job-a",
        "artifact_uri": "memory://checkmaite-report-tests/deduplicated",
        "storage_options": None,
    }

    first_uri = _publish_bytes(**publish_kwargs)
    second_uri = _publish_bytes(**publish_kwargs)
    filesystem, path = fsspec.core.url_to_fs(first_uri)

    assert second_uri == first_uri
    assert filesystem.cat(path) == b"report"
    assert filesystem.find("/checkmaite-report-tests/deduplicated") == [path]


def test_publish_bytes_concurrent_writers_share_verified_content_key() -> None:
    kwargs = {
        "content": b"concurrent report",
        "filename": "report.md",
        "run_uid": "run-concurrent",
        "artifact_scope": "job-concurrent",
        "artifact_uri": "memory://checkmaite-report-tests/concurrent",
        "storage_options": None,
    }

    with ThreadPoolExecutor(max_workers=8) as executor:
        uris = list(executor.map(lambda _: _publish_bytes(**kwargs), range(8)))

    assert len(set(uris)) == 1
    filesystem, path = fsspec.core.url_to_fs(uris[0])
    assert filesystem.cat(path) == b"concurrent report"
    assert filesystem.find("/checkmaite-report-tests/concurrent") == [path]


def test_publish_bytes_atomically_repairs_local_content_key(tmp_path: Path) -> None:
    kwargs = {
        "content": b"local report",
        "filename": "report.md",
        "run_uid": "run-local",
        "artifact_scope": "job-local",
        "artifact_uri": str(tmp_path / "artifacts"),
        "storage_options": None,
    }

    first_uri = _publish_bytes(**kwargs)
    filesystem, path = fsspec.core.url_to_fs(first_uri)
    filesystem.pipe(path, b"corrupt")
    second_uri = _publish_bytes(**kwargs)

    assert second_uri == first_uri
    assert filesystem.cat(path) == b"local report"
    assert filesystem.find(str(tmp_path / "artifacts")) == [path]


def test_build_capability_run_ref_scopes_artifacts_to_job_invocation() -> None:
    run = OversizedReportTinyCapability().run(config=TinyConfig(text="oversized"), use_cache=False)
    references = []
    for artifact_scope in ("job-a", "job-b"):
        ref = build_capability_run_ref(
            run,
            store_uri="memory://run",
            report_threshold=0.5,
            artifact_uri="memory://checkmaite-report-tests/scoped",
            artifact_scope=artifact_scope,
        )
        assert isinstance(ref.report, ArtifactReport)
        references.append(ref.report.uri)

    assert references[0] != references[1]


def test_publish_bytes_repairs_corrupt_content_key() -> None:
    kwargs = {
        "content": b"report",
        "filename": "report.md",
        "run_uid": "run-a",
        "artifact_scope": "job-a",
        "artifact_uri": "memory://checkmaite-report-tests/corrupt",
        "storage_options": None,
    }
    first_uri = _publish_bytes(**kwargs)
    filesystem, path = fsspec.core.url_to_fs(first_uri)
    filesystem.pipe(path, b"corrupt")

    second_uri = _publish_bytes(**kwargs)

    assert second_uri == first_uri
    assert filesystem.cat(path) == b"report"
    assert filesystem.find("/checkmaite-report-tests/corrupt") == [path]


def test_publish_bytes_does_not_damage_existing_key_when_atomic_replace_fails(monkeypatch) -> None:
    filesystem = fsspec.filesystem("memory")
    original_url_to_fs = fsspec.core.url_to_fs
    base_path = "/failed-atomic-replace"
    if filesystem.exists(base_path):
        filesystem.rm(base_path, recursive=True)

    def memory_url_to_fs(url, **storage_options):
        _ = storage_options
        if url == "memory://failed-atomic-replace":
            return filesystem, base_path
        return original_url_to_fs(url, **storage_options)

    monkeypatch.setattr(fsspec.core, "url_to_fs", memory_url_to_fs)
    kwargs = {
        "content": b"report",
        "filename": "report.md",
        "run_uid": "run-a",
        "artifact_scope": "job-a",
        "artifact_uri": "memory://failed-atomic-replace",
        "storage_options": None,
    }
    uri = _publish_bytes(**kwargs)
    _, path = fsspec.core.url_to_fs(uri)
    filesystem.pipe(path, b"corrupt")

    def fail_commit(*_args, **_kwargs):
        raise OSError("atomic replace failed")

    monkeypatch.setattr(filesystem, "cp_file", fail_commit)

    with pytest.raises(OSError, match="atomic replace failed"):
        _publish_bytes(**kwargs)

    assert filesystem.cat(path) == b"corrupt"
    assert filesystem.find(base_path) == [path]


def test_publish_bytes_does_not_overwrite_destination_when_inspection_fails(monkeypatch) -> None:
    filesystem = fsspec.filesystem("memory")
    original_url_to_fs = fsspec.core.url_to_fs
    original_info = filesystem.info
    base_path = "/inspection-failure"
    if filesystem.exists(base_path):
        filesystem.rm(base_path, recursive=True)

    def memory_url_to_fs(url, **storage_options):
        _ = storage_options
        if url == "memory://inspection-failure":
            return filesystem, base_path
        return original_url_to_fs(url, **storage_options)

    monkeypatch.setattr(fsspec.core, "url_to_fs", memory_url_to_fs)
    kwargs = {
        "content": b"report",
        "filename": "report.md",
        "run_uid": "run-a",
        "artifact_scope": "job-a",
        "artifact_uri": "memory://inspection-failure",
        "storage_options": None,
    }
    uri = _publish_bytes(**kwargs)
    _, path = fsspec.core.url_to_fs(uri)

    def fail_destination_info(info_path, **info_kwargs):
        if info_path == path:
            raise PermissionError("inspection denied")
        return original_info(info_path, **info_kwargs)

    monkeypatch.setattr(filesystem, "info", fail_destination_info)

    with pytest.raises(OSError, match="could not inspect artifact path"):
        _publish_bytes(**kwargs)

    with filesystem.open(path, "rb") as published:
        assert published.read() == b"report"


def test_publish_bytes_does_not_overwrite_destination_when_verification_read_fails(monkeypatch) -> None:
    filesystem = fsspec.filesystem("memory")
    original_url_to_fs = fsspec.core.url_to_fs
    original_open = filesystem.open
    base_path = "/verification-read-failure"
    if filesystem.exists(base_path):
        filesystem.rm(base_path, recursive=True)

    def memory_url_to_fs(url, **storage_options):
        _ = storage_options
        if url == "memory://verification-read-failure":
            return filesystem, base_path
        return original_url_to_fs(url, **storage_options)

    monkeypatch.setattr(fsspec.core, "url_to_fs", memory_url_to_fs)
    kwargs = {
        "content": b"report",
        "filename": "report.md",
        "run_uid": "run-a",
        "artifact_scope": "job-a",
        "artifact_uri": "memory://verification-read-failure",
        "storage_options": None,
    }
    uri = _publish_bytes(**kwargs)
    _, path = fsspec.core.url_to_fs(uri)

    def fail_destination_read(open_path, mode="rb", **open_kwargs):
        if open_path == path and mode == "rb":
            raise PermissionError("read denied")
        return original_open(open_path, mode, **open_kwargs)

    monkeypatch.setattr(filesystem, "open", fail_destination_read)

    with pytest.raises(OSError, match="could not verify artifact path"):
        _publish_bytes(**kwargs)

    with original_open(path, "rb") as published:
        assert published.read() == b"report"
    assert filesystem.find(base_path) == [path]


def test_build_capability_run_ref_preserves_producer_owned_artifact_uri() -> None:
    run = _run_with_artifact_report("s3://reports/report.pdf")

    ref = build_capability_run_ref(run, store_uri="memory://run", report_threshold=0.5)

    assert ref.report == ArtifactReport(
        media_type="application/pdf",
        filename="report.pdf",
        uri="s3://reports/report.pdf",
    )


def test_build_capability_run_ref_supports_run_without_report() -> None:
    run = ReportlessTinyCapability().run(config=TinyConfig(text="legacy"), use_cache=False)

    ref = build_capability_run_ref(run, store_uri="memory://run", report_threshold=0.5)

    assert ref.report is None


def test_capability_run_ref_round_trips_through_ray_cloudpickle() -> None:
    ref = CapabilityRunRef(
        run_uid="run-1",
        capability_id="capability-1",
        store_uri="memory://run-1",
        report=ArtifactReport(
            media_type="application/pdf",
            uri="s3://reports/run-1.pdf",
            filename="run-1.pdf",
        ),
    )

    restored = ray.cloudpickle.loads(ray.cloudpickle.dumps(ref))

    assert restored == ref
    assert isinstance(restored.report, ArtifactReport)
