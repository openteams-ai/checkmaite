import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).parents[2] / "docker/ci/check_runtime_lock.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_runtime_lock", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_runtime_lock = _load_script()


def _write_lock(path: Path, packages: list[tuple[str, str | None]]) -> Path:
    entries = []
    for name, version in packages:
        entry = f'[[package]]\nname = "{name}"\n'
        if version is not None:
            entry += f'version = "{version}"\n'
        entries.append(entry)
    path.write_text("version = 1\n\n" + "\n".join(entries), encoding="utf-8")
    return path


def _check(tmp_path: Path, root: list[tuple[str, str | None]], runtime: list[tuple[str, str | None]]) -> int:
    return check_runtime_lock.main(
        _write_lock(tmp_path / "root.lock", root),
        _write_lock(tmp_path / "runtime.lock", runtime),
    )


@pytest.mark.parametrize(
    ("root", "runtime"),
    [
        pytest.param([("numpy", "2.2.6")], [("numpy", "2.2.6")], id="identical"),
        pytest.param(
            [("torch", "2.13.0")],
            [("torch", "2.13.0"), ("torch", "2.13.0+cpu"), ("torch", "2.13.0+cu130")],
            id="local-version-builds",
        ),
        pytest.param(
            [("ipython", "8.39.0"), ("ipython", "9.16.1")],
            [("ipython", "9.16.1"), ("ipython", "8.39.0")],
            id="multiple-versions",
        ),
        pytest.param([("numpy", "2.2.6"), ("coverage", "7.0.0")], [("numpy", "2.2.6")], id="root-only-package"),
        pytest.param(
            [("checkmaite", None)],
            [("checkmaite", None), ("checkmaite-container-runtime", "0.1.0")],
            id="versionless-project-and-runtime-package",
        ),
    ],
)
def test_accepts_consistent_locks(
    tmp_path: Path,
    root: list[tuple[str, str | None]],
    runtime: list[tuple[str, str | None]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _check(tmp_path, root, runtime) == 0
    assert "verified" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("root", "runtime", "message"),
    [
        pytest.param(
            [("numpy", "2.2.6")],
            [("numpy", "2.3.1")],
            "numpy: root=['2.2.6'], container=['2.3.1']",
            id="version-mismatch",
        ),
        pytest.param(
            [("torch", "2.12.0")],
            [("torch", "2.13.0+cpu")],
            "torch: root=['2.12.0'], container=['2.13.0']",
            id="local-build-of-different-release",
        ),
        pytest.param(
            [("ipython", "8.39.0"), ("ipython", "9.16.1")],
            [("ipython", "9.16.1")],
            "ipython: root=['8.39.0', '9.16.1'], container=['9.16.1']",
            id="missing-one-of-multiple-versions",
        ),
        pytest.param(
            [("checkmaite", None)],
            [("checkmaite", "0.4.0")],
            "checkmaite: root=[], container=['0.4.0']",
            id="version-present-on-one-side",
        ),
        pytest.param(
            [("numpy", "2.2.6")],
            [("numpy", "2.2.6"), ("pyyaml", "6.0.3")],
            "pyyaml: container-only package is absent from the root lock",
            id="container-only-package",
        ),
    ],
)
def test_rejects_inconsistent_locks(
    tmp_path: Path,
    root: list[tuple[str, str | None]],
    runtime: list[tuple[str, str | None]],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _check(tmp_path, root, runtime) == 1
    assert message in capsys.readouterr().out


def test_committed_locks_are_consistent() -> None:
    assert check_runtime_lock.main() == 0
