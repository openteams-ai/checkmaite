"""Shared capability-job result construction."""

import hashlib
import os
import posixpath
from contextlib import suppress
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import fsspec
from fsspec.utils import get_protocol

from checkmaite.core.capability_core import CapabilityRunBase
from checkmaite.core.report import MAX_INLINE_REPORT_BYTES, ArtifactReport, CapabilityReport, InlineTextReport
from checkmaite.jobs._store import LOCAL_ARTIFACT_STORE_PROTOCOLS, REMOTE_ARTIFACT_STORE_PROTOCOLS
from checkmaite.jobs.protocol import CapabilityRunRef

_COPY_ARTIFACT_PROTOCOLS = REMOTE_ARTIFACT_STORE_PROTOCOLS | {"memory"}


def _artifact_path(
    base_path: str,
    *,
    run_uid: str,
    artifact_scope: str | None,
    filename: str,
    content: bytes,
) -> str:
    digest = hashlib.sha256(content).hexdigest()
    filename_digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]
    normalized_base = str(base_path).rstrip("/")
    if not normalized_base and str(base_path).startswith("/"):
        normalized_base = "/"
    components = [normalized_base, run_uid]
    if artifact_scope:
        components.append(hashlib.sha256(artifact_scope.encode("utf-8")).hexdigest()[:16])
    components.append(f"{digest}-{filename_digest}")
    return posixpath.join(*components)


def _join_artifact_uri(base_uri: str, relative_path: str) -> str:
    normalized_base = base_uri.rstrip("/")
    if normalized_base.endswith(":"):
        return f"{normalized_base}///{relative_path.lstrip('/')}"
    return f"{normalized_base}/{relative_path.lstrip('/')}"


def _commit_staged_artifact(
    filesystem: Any,
    temporary: str,
    destination: str,
    *,
    protocol: str,
) -> None:
    """Commit a staged file using the operation guaranteed for its protocol."""
    if protocol in LOCAL_ARTIFACT_STORE_PROTOCOLS:
        os.replace(temporary, destination)
        return
    if protocol in _COPY_ARTIFACT_PROTOCOLS:
        filesystem.cp_file(temporary, destination)
        filesystem.rm(temporary)
        return
    raise RuntimeError(f"artifact protocol {protocol!r} does not provide a supported commit operation")


def _publish_bytes(  # noqa: C901
    content: bytes,
    *,
    filename: str,
    run_uid: str,
    artifact_scope: str | None,
    artifact_uri: str,
    storage_options: dict[str, Any] | None,
) -> str:
    """Publish through staging to a verified, deterministic content key."""
    if urlsplit(artifact_uri).query:
        raise ValueError("artifact_uri query credentials must be provided through storage_options")
    protocol = get_protocol(artifact_uri)
    filesystem, base_path = fsspec.core.url_to_fs(artifact_uri, **dict(storage_options or {}))
    destination = _artifact_path(
        str(base_path),
        run_uid=run_uid,
        artifact_scope=artifact_scope,
        filename=filename,
        content=content,
    )
    if protocol in LOCAL_ARTIFACT_STORE_PROTOCOLS:
        published_uri = filesystem.unstrip_protocol(destination)
    else:
        normalized_base = str(base_path).rstrip("/") or ("/" if str(base_path).startswith("/") else ".")
        relative_path = posixpath.relpath(destination, normalized_base)
        published_uri = _join_artifact_uri(artifact_uri, relative_path)
    parent = posixpath.dirname(destination)
    if parent:
        filesystem.makedirs(parent, exist_ok=True)
    expected_digest = hashlib.sha256(content).digest()

    def path_matches(path: str, *, missing_ok: bool) -> bool | None:
        try:
            filesystem.info(path)
        except FileNotFoundError:
            return None if missing_ok else False
        except Exception as exc:
            raise OSError(f"could not inspect artifact path {path!r}") from exc

        try:
            digest = hashlib.sha256()
            with filesystem.open(path, "rb") as published:
                while chunk := published.read(1024 * 1024):
                    digest.update(chunk)
        except FileNotFoundError as exc:
            if missing_ok:
                try:
                    filesystem.info(path)
                except FileNotFoundError:
                    return None
                except Exception as inspect_exc:
                    raise OSError(f"could not inspect artifact path {path!r}") from inspect_exc
            raise OSError(f"artifact path {path!r} disappeared during verification") from exc
        except Exception as exc:
            raise OSError(f"could not verify artifact path {path!r}") from exc
        return digest.digest() == expected_digest

    if path_matches(destination, missing_ok=True) is True:
        return published_uri

    temporary = f"{destination}.tmp-{uuid4().hex}"

    def stage_content() -> None:
        with filesystem.open(temporary, "wb") as destination_stream:
            destination_stream.write(content)
        if path_matches(temporary, missing_ok=False) is not True:
            raise OSError("staged artifact failed content verification")

    try:
        stage_content()
        if path_matches(destination, missing_ok=True) is True:
            return published_uri

        try:
            _commit_staged_artifact(filesystem, temporary, destination, protocol=protocol)
        except Exception:
            if path_matches(destination, missing_ok=True) is True:
                return published_uri
            raise

        if path_matches(destination, missing_ok=True) is not True:
            raise OSError("published artifact failed content verification")
    finally:
        with suppress(Exception):
            filesystem.rm(temporary)
    return published_uri


def _publish_text_report(
    report: InlineTextReport,
    *,
    run_uid: str,
    artifact_scope: str | None,
    artifact_uri: str,
    storage_options: dict[str, Any] | None,
) -> ArtifactReport:
    durable_uri = _publish_bytes(
        report.content.encode("utf-8"),
        filename=report.filename,
        run_uid=run_uid,
        artifact_scope=artifact_scope,
        artifact_uri=artifact_uri,
        storage_options=storage_options,
    )
    return ArtifactReport(media_type=report.media_type, filename=report.filename, uri=durable_uri)


def _collect_md_report(
    run: CapabilityRunBase[Any, Any],
    threshold: float,
    *,
    artifact_uri: str | None = None,
    artifact_storage_options: dict[str, Any] | None = None,
    artifact_scope: str | None = None,
) -> CapabilityReport | None:
    """Collect a producer-finalized report and externalize oversized inline text."""
    try:
        report = run.collect_md_report(threshold=threshold)
    except NotImplementedError:
        return None

    # Report producers own the durability of ArtifactReport URIs and every
    # reference contained by InlineTextReport. The job backend deliberately does
    # not inspect or rewrite report content.
    if isinstance(report, ArtifactReport) or len(report.content.encode("utf-8")) <= MAX_INLINE_REPORT_BYTES:
        return report

    if artifact_uri is not None:
        try:
            return _publish_text_report(
                report,
                run_uid=run.run_uid,
                artifact_scope=artifact_scope,
                artifact_uri=artifact_uri,
                storage_options=artifact_storage_options,
            )
        except Exception as exc:
            raise RuntimeError("Could not publish oversized report to the configured artifact_store") from exc

    raise RuntimeError(
        "Generated report exceeds the inline report size limit; configure artifact_store to publish it durably"
    )


def build_capability_run_ref(
    run: CapabilityRunBase[Any, Any],
    *,
    store_uri: str | None,
    report_threshold: float,
    artifact_uri: str | None = None,
    artifact_storage_options: dict[str, Any] | None = None,
    artifact_scope: str | None = None,
) -> CapabilityRunRef:
    """Build the typed result returned by every capability job backend."""
    return CapabilityRunRef(
        run_uid=run.run_uid,
        capability_id=run.capability_id,
        store_uri=store_uri,
        outputs_uri=None,
        report=_collect_md_report(
            run,
            threshold=report_threshold,
            artifact_uri=artifact_uri,
            artifact_storage_options=artifact_storage_options,
            artifact_scope=artifact_scope,
        ),
    )
