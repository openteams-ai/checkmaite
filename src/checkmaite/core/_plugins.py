import importlib.metadata as importlib_metadata
import logging
import traceback
from collections.abc import Mapping, MutableMapping
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

PLUGIN_API_VERSION = "1.0.0"


class PluginRecord(BaseModel):
    """Record of a single plugin entry point load attempt."""

    group: str
    entry_point_name: str
    package_name: str | None
    symbols: list[str]
    status: Literal["loaded", "failed"]
    error: str | None = None


_registry: list[PluginRecord] = []
_loaded_groups: set[str] = set()


def list_loaded_plugins(group: str | None = None) -> list[PluginRecord]:
    """Return plugin records, optionally filtered by group.

    Parameters
    ----------
    group : str or None
        If provided, only return records matching this entry-point group.
        If None, return all records.

    Returns
    -------
    list[PluginRecord]
        Copy of matching plugin records.
    """
    if group is None:
        return [r.model_copy() for r in _registry]
    return [r.model_copy() for r in _registry if r.group == group]


def _clear_registry() -> None:
    """Reset the plugin registry. For testing only."""
    _registry.clear()
    _loaded_groups.clear()


def _check_api_version(declared_version: object, entry_point_name: str, group: str) -> str | None:
    """Validate the plugin API version. Returns an error message, or None if compatible."""
    if declared_version is None:
        msg = "Plugin must declare '__plugin_api_version__' in its exports mapping"
        logger.warning(
            "Plugin entry point '%s' in group '%s' missing required '__plugin_api_version__' key.",
            entry_point_name,
            group,
        )
        return msg

    if not isinstance(declared_version, str):
        msg = f"'__plugin_api_version__' must be a string, got {type(declared_version).__name__}"
        logger.warning("Plugin entry point '%s' in group '%s': %s.", entry_point_name, group, msg)
        return msg

    try:
        plugin_major = int(declared_version.split(".")[0])
        core_major = int(PLUGIN_API_VERSION.split(".")[0])
    except (ValueError, IndexError):
        msg = f"Cannot parse plugin API version '{declared_version}'"
        logger.warning("Plugin entry point '%s' in group '%s': %s.", entry_point_name, group, msg)
        return msg

    if plugin_major != core_major:
        msg = (
            f"Plugin declares API version {declared_version} but checkmaite requires "
            f"major version {core_major} (current: {PLUGIN_API_VERSION}). Update the plugin."
        )
        logger.warning("Plugin entry point '%s' in group '%s': %s", entry_point_name, group, msg)
        return msg

    return None


def _load_entry_point(entry_point: Any, group: str, exports: dict[str, Any]) -> None:
    """Load a single entry point, validate it, and merge symbols into exports."""
    package_name = entry_point.dist.name if entry_point.dist else None

    try:
        loaded_obj = entry_point.load()
        if not callable(loaded_obj):
            logger.warning(
                "Plugin entry point '%s' in group '%s' did not resolve to a callable.",
                entry_point.name,
                group,
            )
            _registry.append(
                PluginRecord(
                    group=group,
                    entry_point_name=entry_point.name,
                    package_name=package_name,
                    symbols=[],
                    status="failed",
                    error=f"Entry point '{entry_point.name}' is not callable",
                )
            )
            return

        plugin_exports = loaded_obj()
        if not isinstance(plugin_exports, Mapping):
            logger.warning(
                "Plugin entry point '%s' in group '%s' callable returned %s; expected a mapping.",
                entry_point.name,
                group,
                type(plugin_exports).__name__,
            )
            _registry.append(
                PluginRecord(
                    group=group,
                    entry_point_name=entry_point.name,
                    package_name=package_name,
                    symbols=[],
                    status="failed",
                    error=f"Callable returned {type(plugin_exports).__name__}, expected a mapping",
                )
            )
            return

        # --- Version check ---
        mutable_exports = dict(plugin_exports)
        declared_version = mutable_exports.pop("__plugin_api_version__", None)
        version_error = _check_api_version(declared_version, entry_point.name, group)
        if version_error is not None:
            _registry.append(
                PluginRecord(
                    group=group,
                    entry_point_name=entry_point.name,
                    package_name=package_name,
                    symbols=[],
                    status="failed",
                    error=version_error,
                )
            )
            return

        for symbol_name, symbol in mutable_exports.items():
            if not isinstance(symbol, type):
                logger.warning(
                    "Plugin entry point '%s' in group '%s' export '%s' is not a class (got %s), skipping.",
                    entry_point.name,
                    group,
                    symbol_name,
                    type(symbol).__name__,
                )
                continue
            if symbol_name in exports:
                logger.warning(
                    "Plugin entry point '%s' in group '%s' overrides export '%s'.",
                    entry_point.name,
                    group,
                    symbol_name,
                )
            exports[symbol_name] = symbol

        accepted_symbols = [s for s in mutable_exports if s in exports and isinstance(mutable_exports[s], type)]
        _registry.append(
            PluginRecord(
                group=group,
                entry_point_name=entry_point.name,
                package_name=package_name,
                symbols=accepted_symbols,
                status="loaded",
                error=None,
            )
        )
    except ImportError:
        logger.debug(
            "Plugin entry point '%s' from group '%s' not loadable (missing dependency).",
            entry_point.name,
            group,
            exc_info=True,
        )
        _registry.append(
            PluginRecord(
                group=group,
                entry_point_name=entry_point.name,
                package_name=package_name,
                symbols=[],
                status="failed",
                error=traceback.format_exc(),
            )
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "Failed to load plugin entry point '%s' from group '%s'.",
            entry_point.name,
            group,
            exc_info=True,
        )
        _registry.append(
            PluginRecord(
                group=group,
                entry_point_name=entry_point.name,
                package_name=package_name,
                symbols=[],
                status="failed",
                error=traceback.format_exc(),
            )
        )


def inject_plugin_exports(module_globals: MutableMapping[str, Any], exported_symbols: list[str], *, group: str) -> None:
    if group in _loaded_groups:
        return
    _loaded_groups.add(group)

    exports: dict[str, Any] = {}

    for entry_point in list(importlib_metadata.entry_points(group=group)):
        _load_entry_point(entry_point, group, exports)

    for symbol_name, symbol in exports.items():
        module_globals[symbol_name] = symbol
        if symbol_name not in exported_symbols:
            exported_symbols.append(symbol_name)

    logger.debug("Loaded %d symbol(s) from plugin group '%s'.", len(exports), group)
