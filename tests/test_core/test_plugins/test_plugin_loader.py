import logging
import uuid
from unittest.mock import MagicMock

import pytest

from checkmaite.core import _plugins
from checkmaite.core._plugins import PLUGIN_API_VERSION, PluginRecord, _clear_registry, _registry, list_loaded_plugins


def _make_entry_point(name: str, package: str, load_return: object) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = load_return
    ep.dist = MagicMock()
    ep.dist.name = package
    return ep


def test_inject_records_successful_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_callable() -> dict[str, object]:
        return {
            "__plugin_api_version__": "1.0.0",
            "MyCapability": type("MyCapability", (), {}),
            "MyConfig": type("MyConfig", (), {}),
        }

    ep = _make_entry_point("my-plugin", "my-package", fake_callable)
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.group")

    assert "MyCapability" in module_globals
    assert "MyConfig" in module_globals
    assert "MyCapability" in exported_symbols
    assert "MyConfig" in exported_symbols

    records = list_loaded_plugins()
    assert len(records) == 1
    assert records[0].status == "loaded"
    assert records[0].group == "test.group"
    assert records[0].entry_point_name == "my-plugin"
    assert records[0].package_name == "my-package"
    assert sorted(records[0].symbols) == ["MyCapability", "MyConfig"]
    assert records[0].error is None


def test_inject_plugin_exports_fake_group_preserves_existing_values() -> None:
    existing_symbol = object()
    module_globals: dict[str, object] = {"Existing": existing_symbol}
    exported_symbols: list[str] = ["Existing"]

    _plugins.inject_plugin_exports(
        module_globals,
        exported_symbols,
        group=f"checkmaite.plugins.fake.{uuid.uuid4().hex}",
    )

    assert module_globals == {"Existing": existing_symbol}
    assert exported_symbols == ["Existing"]


def test_plugin_record_loaded() -> None:
    record = PluginRecord(
        group="checkmaite.plugins.object_detection",
        entry_point_name="survivor",
        package_name="checkmaite-plugins",
        symbols=["Survivor", "SurvivorConfig"],
        status="loaded",
        error=None,
    )
    assert record.status == "loaded"
    assert record.symbols == ["Survivor", "SurvivorConfig"]
    assert record.error is None


def test_plugin_record_failed() -> None:
    record = PluginRecord(
        group="checkmaite.plugins.object_detection",
        entry_point_name="broken",
        package_name="broken-plugin",
        symbols=[],
        status="failed",
        error="ImportError: no module named 'broken'",
    )
    assert record.status == "failed"
    assert record.symbols == []
    assert record.error is not None


def test_list_loaded_plugins_empty() -> None:
    assert list_loaded_plugins() == []


def test_list_loaded_plugins_returns_all() -> None:
    record_od = PluginRecord(
        group="checkmaite.plugins.object_detection",
        entry_point_name="survivor",
        package_name="checkmaite-plugins",
        symbols=["Survivor"],
        status="loaded",
        error=None,
    )
    record_ic = PluginRecord(
        group="checkmaite.plugins.image_classification",
        entry_point_name="survivor",
        package_name="checkmaite-plugins",
        symbols=["Survivor"],
        status="loaded",
        error=None,
    )
    _registry.extend([record_od, record_ic])

    result = list_loaded_plugins()
    assert len(result) == 2
    assert result[0] is not record_od  # returns a copy


def test_list_loaded_plugins_filters_by_group() -> None:
    record_od = PluginRecord(
        group="checkmaite.plugins.object_detection",
        entry_point_name="survivor",
        package_name="checkmaite-plugins",
        symbols=["Survivor"],
        status="loaded",
        error=None,
    )
    record_ic = PluginRecord(
        group="checkmaite.plugins.image_classification",
        entry_point_name="survivor",
        package_name="checkmaite-plugins",
        symbols=["Survivor"],
        status="loaded",
        error=None,
    )
    _registry.extend([record_od, record_ic])

    result = list_loaded_plugins(group="checkmaite.plugins.object_detection")
    assert len(result) == 1
    assert result[0].group == "checkmaite.plugins.object_detection"


def test_clear_registry() -> None:
    _registry.append(
        PluginRecord(
            group="test",
            entry_point_name="test",
            package_name="test",
            symbols=[],
            status="loaded",
            error=None,
        )
    )
    assert len(_registry) > 0
    _clear_registry()
    assert len(_registry) == 0


def test_inject_records_not_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = _make_entry_point("bad-plugin", "bad-package", "not a callable")
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.group")

    assert module_globals == {}
    assert exported_symbols == []

    records = list_loaded_plugins()
    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].entry_point_name == "bad-plugin"
    assert "not callable" in (records[0].error or "")


def test_inject_records_non_mapping_return(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = _make_entry_point("bad-plugin", "bad-package", lambda: ["not", "a", "mapping"])
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.group")

    assert module_globals == {}
    assert exported_symbols == []

    records = list_loaded_plugins()
    assert len(records) == 1
    assert records[0].status == "failed"
    assert "list" in (records[0].error or "")


def test_inject_records_exception_during_load(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = MagicMock()
    ep.name = "crash-plugin"
    ep.dist = MagicMock()
    ep.dist.name = "crash-package"
    ep.load.side_effect = RuntimeError("kaboom")
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.group")

    assert module_globals == {}
    assert exported_symbols == []

    records = list_loaded_plugins()
    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].entry_point_name == "crash-plugin"
    assert "kaboom" in (records[0].error or "")


def test_inject_records_duplicate_symbol_override(monkeypatch: pytest.MonkeyPatch) -> None:
    first_cls = type("Shared", (), {"_source": "first"})
    second_cls = type("Shared", (), {"_source": "second"})

    ep1 = _make_entry_point("plugin-a", "package-a", lambda: {"__plugin_api_version__": "1.0.0", "Shared": first_cls})
    ep2 = _make_entry_point("plugin-b", "package-b", lambda: {"__plugin_api_version__": "1.0.0", "Shared": second_cls})
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep1, ep2])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.group")

    # Last entry point wins
    assert module_globals["Shared"] is second_cls
    assert exported_symbols.count("Shared") == 1

    records = list_loaded_plugins()
    assert len(records) == 2
    assert all(r.status == "loaded" for r in records)
    assert records[0].symbols == ["Shared"]
    assert records[1].symbols == ["Shared"]


def test_inject_idempotent_second_call_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling inject_plugin_exports twice for the same group only loads once."""
    call_count = 0

    def counting_callable() -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return {"__plugin_api_version__": "1.0.0", "Cap": type("Cap", (), {})}

    ep = _make_entry_point("my-plugin", "my-package", counting_callable)
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.idempotent")
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.idempotent")

    assert call_count == 1
    records = list_loaded_plugins(group="test.idempotent")
    assert len(records) == 1


def test_import_error_logs_debug_not_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """ImportError during entry point load should log at DEBUG, not WARNING."""
    ep = MagicMock()
    ep.name = "missing-dep-plugin"
    ep.dist = MagicMock()
    ep.dist.name = "missing-package"
    ep.load.side_effect = ImportError("No module named 'heavy_dep'")
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []

    with caplog.at_level(logging.DEBUG, logger="checkmaite.core._plugins"):
        _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.loglevel")

    plugin_records = [r for r in caplog.records if r.name == "checkmaite.core._plugins"]
    assert any(r.levelno == logging.DEBUG for r in plugin_records)
    assert not any(r.levelno == logging.WARNING for r in plugin_records)

    records = list_loaded_plugins(group="test.loglevel")
    assert len(records) == 1
    assert records[0].status == "failed"
    assert "heavy_dep" in (records[0].error or "")


def test_other_exception_logs_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Non-ImportError during entry point load should log at WARNING."""
    ep = MagicMock()
    ep.name = "buggy-plugin"
    ep.dist = MagicMock()
    ep.dist.name = "buggy-package"
    ep.load.side_effect = RuntimeError("plugin has a bug")
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []

    with caplog.at_level(logging.DEBUG, logger="checkmaite.core._plugins"):
        _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.loglevel2")

    plugin_records = [r for r in caplog.records if r.name == "checkmaite.core._plugins"]
    assert any(r.levelno == logging.WARNING for r in plugin_records)

    records = list_loaded_plugins(group="test.loglevel2")
    assert len(records) == 1
    assert records[0].status == "failed"


def test_version_match_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plugin with matching major version loads normally."""

    def versioned_callable() -> dict[str, object]:
        return {"__plugin_api_version__": PLUGIN_API_VERSION, "MyCap": type("MyCap", (), {})}

    ep = _make_entry_point("good-plugin", "good-package", versioned_callable)
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.version.match")

    assert "MyCap" in module_globals
    assert "__plugin_api_version__" not in module_globals

    records = list_loaded_plugins(group="test.version.match")
    assert len(records) == 1
    assert records[0].status == "loaded"


def test_version_minor_mismatch_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plugin with different minor version but same major loads normally."""

    def versioned_callable() -> dict[str, object]:
        return {"__plugin_api_version__": "1.99.0", "MyCap": type("MyCap", (), {})}

    ep = _make_entry_point("good-plugin", "good-package", versioned_callable)
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.version.minor")

    assert "MyCap" in module_globals
    records = list_loaded_plugins(group="test.version.minor")
    assert records[0].status == "loaded"


def test_version_major_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plugin with different major version is rejected."""

    def versioned_callable() -> dict[str, object]:
        return {"__plugin_api_version__": "99.0.0", "MyCap": type("MyCap", (), {})}

    ep = _make_entry_point("old-plugin", "old-package", versioned_callable)
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.version.major")

    assert "MyCap" not in module_globals

    records = list_loaded_plugins(group="test.version.major")
    assert len(records) == 1
    assert records[0].status == "failed"
    assert "API version" in (records[0].error or "") or "major version" in (records[0].error or "").lower()


def test_version_missing_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plugin without __plugin_api_version__ is rejected."""

    def no_version_callable() -> dict[str, object]:
        return {"MyCap": type("MyCap", (), {})}

    ep = _make_entry_point("no-version", "no-version-pkg", no_version_callable)
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.version.missing")

    assert "MyCap" not in module_globals

    records = list_loaded_plugins(group="test.version.missing")
    assert len(records) == 1
    assert records[0].status == "failed"
    assert "__plugin_api_version__" in (records[0].error or "")


def test_version_not_string_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plugin with non-string version is rejected."""

    def bad_version_callable() -> dict[str, object]:
        return {"__plugin_api_version__": 1, "MyCap": type("MyCap", (), {})}

    ep = _make_entry_point("bad-version", "bad-version-pkg", bad_version_callable)
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.version.notstr")

    assert "MyCap" not in module_globals

    records = list_loaded_plugins(group="test.version.notstr")
    assert records[0].status == "failed"


def test_version_malformed_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plugin with unparseable version string is rejected."""

    def bad_version_callable() -> dict[str, object]:
        return {"__plugin_api_version__": "not.a.version", "MyCap": type("MyCap", (), {})}

    ep = _make_entry_point("malformed", "malformed-pkg", bad_version_callable)
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.version.malformed")

    assert "MyCap" not in module_globals

    records = list_loaded_plugins(group="test.version.malformed")
    assert records[0].status == "failed"


def test_non_class_export_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plugin exporting a non-class value gets a warning and the value is skipped."""

    def bad_exports() -> dict[str, object]:
        return {"__plugin_api_version__": "1.0.0", "NotAClass": 42}

    ep = _make_entry_point("bad-export", "bad-export-pkg", bad_exports)
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.typecheck")

    assert "NotAClass" not in module_globals
    assert exported_symbols == []

    records = list_loaded_plugins(group="test.typecheck")
    assert len(records) == 1
    assert records[0].status == "loaded"
    assert records[0].symbols == []


def test_mixed_valid_invalid_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid class exports load, non-class exports are skipped."""
    good_class = type("GoodClass", (), {})

    def mixed_exports() -> dict[str, object]:
        return {"__plugin_api_version__": "1.0.0", "GoodClass": good_class, "BadValue": "not a class"}

    ep = _make_entry_point("mixed", "mixed-pkg", mixed_exports)
    monkeypatch.setattr(_plugins.importlib_metadata, "entry_points", lambda group: [ep])

    module_globals: dict[str, object] = {}
    exported_symbols: list[str] = []
    _plugins.inject_plugin_exports(module_globals, exported_symbols, group="test.typecheck2")

    assert "GoodClass" in module_globals
    assert module_globals["GoodClass"] is good_class
    assert "BadValue" not in module_globals
    assert exported_symbols == ["GoodClass"]

    records = list_loaded_plugins(group="test.typecheck2")
    assert records[0].symbols == ["GoodClass"]
