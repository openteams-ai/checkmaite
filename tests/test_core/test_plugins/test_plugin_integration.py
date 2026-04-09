"""Integration tests for plugin discovery across multiple packages.

TestRealPlugins: requires checkmaite-plugins installed, tests skip otherwise.
TestFakePlugins: always runs, uses mocked entry points to prove multi-repo works.

Note: The autouse fixture in conftest.py clears the plugin registry before each test.
Real plugin records are created at module import time (when __init__.py first runs) and
cannot be re-populated without re-importing. So real plugin tests check the namespace
directly, while fake plugin tests exercise the registry via explicit inject_plugin_exports calls.
"""

from __future__ import annotations

from collections.abc import Mapping
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
# Real plugin tests (skip if checkmaite-plugins not installed)
# ---------------------------------------------------------------------------


try:
    import checkmaite_plugins  # noqa: F401

    _has_plugins = True
except ImportError:
    _has_plugins = False


@pytest.mark.skipif(not _has_plugins, reason="checkmaite-plugins not installed")
class TestRealPlugins:
    """Tests that verify the real checkmaite-plugins package injects symbols into checkmaite."""

    def test_od_heart_symbols_in_namespace(self) -> None:
        import checkmaite.core.object_detection as od

        assert hasattr(od, "HeartAdversarial")
        assert hasattr(od, "HeartAdversarialConfig")
        assert hasattr(od, "HeartAdversarialOutputs")
        assert hasattr(od, "HeartAdversarialAttackConfig")

    def test_od_heart_symbols_in_all(self) -> None:
        import checkmaite.core.object_detection as od

        assert "HeartAdversarial" in od.__all__
        assert "HeartAdversarialConfig" in od.__all__
        assert "HeartAdversarialOutputs" in od.__all__
        assert "HeartAdversarialAttackConfig" in od.__all__

    def test_od_core_symbols_still_present(self) -> None:
        """Plugins don't break the existing core capabilities."""
        import checkmaite.core.object_detection as od

        assert hasattr(od, "DataevalBias")
        assert hasattr(od, "NrtkRobustness")
        assert hasattr(od, "XaitkExplainable")

    def test_ic_core_symbols_still_present(self) -> None:
        """Plugins don't break the existing core IC capabilities."""
        import checkmaite.core.image_classification as ic

        assert hasattr(ic, "DataevalBias")
        assert hasattr(ic, "NrtkRobustness")
        assert hasattr(ic, "XaitkExplainable")
