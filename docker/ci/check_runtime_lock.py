"""Verify that the private container lock tracks packages and versions in the root lock."""

from __future__ import annotations

import importlib
import sys
from collections import defaultdict
from pathlib import Path

# tomllib is in the standard library from Python 3.11 onward. The root lock
# provides tomli on Python 3.10.
tomllib = importlib.import_module("tomllib" if sys.version_info >= (3, 11) else "tomli")

_ROOT = Path(__file__).resolve().parents[2]
# The runtime package is the only package that exists solely for the container.
_CONTAINER_ONLY_PACKAGES = frozenset({"checkmaite-container-runtime"})


def _locked_versions(path: Path) -> dict[str, set[str]]:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    versions: dict[str, set[str]] = defaultdict(set)
    for package in document["package"]:
        # Record every name, including a local project without a static version.
        package_versions = versions[package["name"]]
        version = package.get("version")
        if version is not None:
            # Device-specific PyTorch wheels add a local version suffix while
            # remaining on the root lock's upstream release.
            package_versions.add(version.partition("+")[0])
    return dict(versions)


def main(root_lock: Path | None = None, runtime_lock: Path | None = None) -> int:
    """Fail when the container lock adds packages or changes shared upstream versions."""
    root_versions = _locked_versions(_ROOT / "uv.lock" if root_lock is None else root_lock)
    runtime_versions = _locked_versions(
        _ROOT / "docker/runtime/uv.lock" if runtime_lock is None else runtime_lock,
    )
    unexpected = runtime_versions.keys() - root_versions.keys() - _CONTAINER_ONLY_PACKAGES
    mismatches = {
        name: (root_versions[name], runtime_versions[name])
        for name in root_versions.keys() & runtime_versions.keys()
        if root_versions[name] != runtime_versions[name]
    }
    for name in sorted(unexpected):
        print(f"{name}: container-only package is absent from the root lock")
    for name, (root, runtime) in sorted(mismatches.items()):
        print(
            f"{name}: root={sorted(root)!r}, container={sorted(runtime)!r}",
        )
    if unexpected or mismatches:
        print("regenerate docker/runtime/uv.lock from the updated root dependency set")
        return 1
    print(f"verified {len(root_versions.keys() & runtime_versions.keys())} shared locked packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
