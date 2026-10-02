#!/usr/bin/env python3
"""Measure uncompressed and gzip-compressed sizes of local container images."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess  # nosec B404 - required to invoke Docker without a shell
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 1024 * 1024


class _ByteCounter:
    """A write-only stream that counts bytes without retaining them."""

    def __init__(self) -> None:
        self.bytes_written = 0

    def write(self, data: bytes) -> int:
        """Count data written by gzip and report a successful write."""
        size = len(data)
        self.bytes_written += size
        return size

    def flush(self) -> None:
        """Satisfy the file-like interface used by gzip."""


@dataclass(frozen=True)
class ImageSize:
    """Exact size measurements and identity for one local image."""

    reference: str
    image_id: str
    platform: str
    uncompressed_bytes: int
    compressed_archive_bytes: int


def _docker_executable() -> str:
    executable = shutil.which("docker")
    if executable is None:
        raise FileNotFoundError("docker is not available on PATH")
    return executable


def _inspect_image(reference: str) -> dict[str, Any]:
    completed = subprocess.run(  # noqa: S603  # nosec B603 - fixed command, no shell
        [_docker_executable(), "image", "inspect", reference],
        check=True,
        capture_output=True,
        text=True,
    )
    inspected = json.loads(completed.stdout)
    if len(inspected) != 1:
        msg = f"expected one image for {reference!r}, got {len(inspected)}"
        raise RuntimeError(msg)
    return inspected[0]


def _compressed_archive_size(reference: str) -> int:
    """Return the size of `docker image save` compressed with gzip level 6."""
    command = [_docker_executable(), "image", "save", reference]
    process = subprocess.Popen(  # noqa: S603  # nosec B603 - fixed command, no shell
        command,
        stdout=subprocess.PIPE,
    )
    stream = process.stdout
    if stream is None:  # pragma: no cover - guaranteed by stdout=PIPE
        process.kill()
        raise RuntimeError("docker image save did not provide an output stream")

    counter = _ByteCounter()
    try:
        with (
            stream,
            gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=counter,
                mtime=0,
            ) as compressed,
        ):
            shutil.copyfileobj(stream, compressed, length=_CHUNK_SIZE)
    except BaseException:
        process.kill()
        process.wait()
        raise

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            command,
        )
    return counter.bytes_written


def measure_image(reference: str) -> ImageSize:
    """Measure one local image without writing an archive to disk."""
    inspected = _inspect_image(reference)
    print(f"Compressing {reference}...", file=sys.stderr, flush=True)
    compressed_bytes = _compressed_archive_size(reference)
    return ImageSize(
        reference=reference,
        image_id=str(inspected["Id"]),
        platform=f"{inspected['Os']}/{inspected['Architecture']}",
        uncompressed_bytes=int(inspected["Size"]),
        compressed_archive_bytes=compressed_bytes,
    )


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _render_table(measurements: list[ImageSize]) -> str:
    headings = ("IMAGE", "PLATFORM", "UNCOMPRESSED", "COMPRESSED ARCHIVE")
    rows = [
        (
            result.reference,
            result.platform,
            _human_size(result.uncompressed_bytes),
            _human_size(result.compressed_archive_bytes),
        )
        for result in measurements
    ]
    widths = [max(len(headings[index]), *(len(row[index]) for row in rows)) for index in range(len(headings))]
    lines = [
        "  ".join(heading.ljust(widths[index]) for index, heading in enumerate(headings)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows)
    return "\n".join(lines)


def _render_json(measurements: list[ImageSize]) -> str:
    report = {
        "schema_version": 1,
        "methods": {
            "uncompressed_bytes": "Docker Engine image inspect .Size",
            "compressed_archive_bytes": ("gzip level 6 with mtime 0 over the docker image save archive"),
        },
        "images": [asdict(result) for result in measurements],
    }
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Measure requested local images and print or save the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="local image references to measure")
    parser.add_argument(
        "--json",
        metavar="PATH",
        type=Path,
        help="also write exact-byte measurements as JSON; use - for stdout",
    )
    args = parser.parse_args()

    try:
        measurements = [measure_image(reference) for reference in args.images]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(f"error: command failed with status {error.returncode}", file=sys.stderr)
        if error.stderr:
            print(error.stderr.rstrip(), file=sys.stderr)
        return error.returncode or 1

    rendered_json = _render_json(measurements)
    if args.json == Path("-"):
        print(rendered_json, end="")
    else:
        print(_render_table(measurements))
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(rendered_json, encoding="utf-8")
            print(f"\nExact-byte report: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
