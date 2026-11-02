"""Integration tests for plugin discovery across multiple packages.

TestFakePlugins: uses mocked entry points to prove multi-repo works.
TestRealEntryPoints: installs a throwaway dist-info distribution on PYTHONPATH and imports
checkmaite in a subprocess, so discovery runs through real importlib.metadata entry points.

Note: The autouse fixture in conftest.py clears the plugin registry before each test.
Real plugin records are created at module import time (when __init__.py first runs) and
cannot be re-populated without re-importing in this process, hence the subprocess.
Capability-specific checks for checkmaite-plugins (e.g. HEART) belong in that repository.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from checkmaite.core import _plugins
from checkmaite.core._plugins import list_loaded_plugins

# ---------------------------------------------------------------------------
# Fake plugin tests (always run — proves multi-repo discovery works)
# ---------------------------------------------------------------------------


class FakeCapability:
    """Fake capability class for testing."""


class FakeConfig:
    """Fake config class for testing."""


def _fake_single_plugin_exports() -> Mapping[str, Any]:
    """Simulates a single-plugin repo returning one capability."""
    return {
        "__plugin_api_version__": "1.0.0",
        "FakeCapability": FakeCapability,
        "FakeConfig": FakeConfig,
    }


def _fake_mono_plugin_exports() -> Mapping[str, Any]:
    """Simulates a mono-repo returning multiple capabilities."""
    return {
        "__plugin_api_version__": "1.0.0",
        "AlphaCapability": type("AlphaCapability", (), {}),
        "BetaCapability": type("BetaCapability", (), {}),
        "GammaCapability": type("GammaCapability", (), {}),
    }


def _make_entry_point(name: str, package: str, func: object) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = func
    ep.dist = MagicMock()
    ep.dist.name = package
    return ep


class TestFakePlugins:
    """Tests that dynamically create fake plugin packages to prove multi-repo works."""

    def test_single_fake_plugin_loads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A single-plugin repo registers one entry point with a few symbols."""
        ep = _make_entry_point("debiaser", "checkmaite-plugin-debiaser", _fake_single_plugin_exports)
        monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

        module_globals: dict[str, object] = {}
        exported_symbols: list[str] = []
        _plugins.inject_plugin_exports(
            module_globals, exported_symbols, group="checkmaite.plugins.image_classification"
        )

        assert "FakeCapability" in module_globals
        assert "FakeConfig" in module_globals

        records = list_loaded_plugins(group="checkmaite.plugins.image_classification")
        assert len(records) == 1
        assert records[0].package_name == "checkmaite-plugin-debiaser"
        assert records[0].status == "loaded"

    def test_mono_repo_and_single_repo_coexist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A mono-repo and a single-plugin repo both register under the same group."""
        ep_mono = _make_entry_point("default", "checkmaite-plugins", _fake_mono_plugin_exports)
        ep_single = _make_entry_point("debiaser", "checkmaite-plugin-debiaser", _fake_single_plugin_exports)
        monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep_mono, ep_single])

        module_globals: dict[str, object] = {}
        exported_symbols: list[str] = []
        _plugins.inject_plugin_exports(module_globals, exported_symbols, group="checkmaite.plugins.object_detection")

        # All symbols from both plugins are available
        assert "AlphaCapability" in module_globals
        assert "BetaCapability" in module_globals
        assert "GammaCapability" in module_globals
        assert "FakeCapability" in module_globals
        assert "FakeConfig" in module_globals
        assert len(exported_symbols) == 5

        # Registry shows both plugins
        records = list_loaded_plugins(group="checkmaite.plugins.object_detection")
        assert len(records) == 2
        assert records[0].package_name == "checkmaite-plugins"
        assert records[0].symbols == ["AlphaCapability", "BetaCapability", "GammaCapability"]
        assert records[1].package_name == "checkmaite-plugin-debiaser"
        assert records[1].symbols == ["FakeCapability", "FakeConfig"]

    def test_multiple_mono_repos_coexist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two different mono-repos both register under the same group."""
        ep_team_a = _make_entry_point("default", "checkmaite-plugins-team-a", _fake_mono_plugin_exports)
        ep_team_b = _make_entry_point("default", "checkmaite-plugins-team-b", _fake_single_plugin_exports)
        monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep_team_a, ep_team_b])

        module_globals: dict[str, object] = {}
        exported_symbols: list[str] = []
        _plugins.inject_plugin_exports(module_globals, exported_symbols, group="checkmaite.plugins.object_detection")

        # All symbols from both repos
        assert "AlphaCapability" in module_globals
        assert "FakeCapability" in module_globals
        assert len(exported_symbols) == 5

        records = list_loaded_plugins(group="checkmaite.plugins.object_detection")
        assert len(records) == 2
        assert records[0].package_name == "checkmaite-plugins-team-a"
        assert records[1].package_name == "checkmaite-plugins-team-b"


# ---------------------------------------------------------------------------
# Core symbols and real entry-point discovery (always run)
# ---------------------------------------------------------------------------


def test_od_core_symbols_still_present() -> None:
    """Plugin injection doesn't break the existing core capabilities."""
    import checkmaite.core.object_detection as od

    assert hasattr(od, "DataevalBias")
    assert hasattr(od, "NrtkRobustness")
    assert hasattr(od, "XaitkExplainable")


def test_ic_core_symbols_still_present() -> None:
    """Plugin injection doesn't break the existing core IC capabilities."""
    import checkmaite.core.image_classification as ic

    assert hasattr(ic, "DataevalBias")
    assert hasattr(ic, "NrtkRobustness")
    assert hasattr(ic, "XaitkExplainable")


class TestRealEntryPoints:
    """A real (throwaway) installed distribution injects symbols at first import."""

    def test_dist_info_entry_points_inject_symbols(self, tmp_path: Path) -> None:
        (tmp_path / "fake_ckm_plugin.py").write_text(
            textwrap.dedent("""
                class DummyOdCapability: ...
                class DummyIcCapability: ...

                def od_exports():
                    return {"__plugin_api_version__": "1.0.0", "DummyOdCapability": DummyOdCapability}

                def ic_exports():
                    return {"__plugin_api_version__": "1.0.0", "DummyIcCapability": DummyIcCapability}
            """)
        )
        dist_info = tmp_path / "fake_ckm_plugin-0.0.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-ckm-plugin\nVersion: 0.0.0\n")
        (dist_info / "entry_points.txt").write_text(
            "[checkmaite.plugins.object_detection]\nfake = fake_ckm_plugin:od_exports\n\n"
            "[checkmaite.plugins.image_classification]\nfake = fake_ckm_plugin:ic_exports\n"
        )
        code = textwrap.dedent("""
            import checkmaite.core.image_classification as ic
            import checkmaite.core.object_detection as od
            from checkmaite.core._plugins import list_loaded_plugins

            assert od.DummyOdCapability.__module__ == "fake_ckm_plugin" and "DummyOdCapability" in od.__all__
            assert ic.DummyIcCapability.__module__ == "fake_ckm_plugin" and "DummyIcCapability" in ic.__all__
            fake = [r for r in list_loaded_plugins() if r.package_name == "fake-ckm-plugin"]
            assert sorted(r.status for r in fake) == ["loaded", "loaded"], fake
        """)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(tmp_path), os.environ.get("PYTHONPATH")]))}
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300
        )
        assert result.returncode == 0, result.stderr
