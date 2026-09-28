"""Verify advertised maite.protocols entry points with MAITE's static checker.

Must run inside the project environment (``uv run python scripts/statically_verify_maite_entrypoints.py``).
The harness invokes ``pyright`` on a temp file with no ``--pythonpath``, so a bare
``python scripts/...`` uses the wrong interpreter and fails all entries on import
resolution, not on types.
"""

from __future__ import annotations

import sys

try:
    from maite._internals.testing.project import statically_verify_exposed_component_entrypoints
    from maite._internals.testing.pyright import PYRIGHT_PATH
except ImportError as exc:
    print(
        "cannot import MAITE's static entrypoint verifier " f"(maite._internals.testing, pin is maite<0.10): {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


def main() -> int:
    if PYRIGHT_PATH is None:
        print("pyright not on PATH; install the dev group", file=sys.stderr)
        return 1
    results = statically_verify_exposed_component_entrypoints(package_name="checkmaite")
    if not results:
        print("no maite.protocols entry points found for checkmaite", file=sys.stderr)
        return 1
    failed = sorted(name for name, ok in results.items() if not ok)
    if failed:
        print("statically_verify failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    print(f"verified {len(results)} maite.protocols entry points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
