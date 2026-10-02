"""Load classes referenced by a batch-container run plan."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from checkmaite_container._plan import ObjectSpec, RunPlan, RunPlanConfigurationError, TaskSpec

_NAMESPACES = ("image_classification", "object_detection")


def _load_file_symbol(reference: str, plan_root: Path) -> Any:
    path_reference, separator, symbol_name = reference.removeprefix("file:").rpartition(":")
    if not separator or not path_reference or not symbol_name:
        raise RunPlanConfigurationError("file references must use 'file:path/to/plugin.py:Symbol'")

    module_path = Path(path_reference)
    if not module_path.is_absolute():
        module_path = plan_root / module_path
    module_path = module_path.resolve()
    if not module_path.is_file():
        raise RunPlanConfigurationError(f"plugin file does not exist: {module_path}")

    digest = hashlib.sha256(str(module_path).encode()).hexdigest()[:16]
    module_name = f"checkmaite_file_plugin_{digest}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RunPlanConfigurationError(f"cannot load plugin module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            sys.modules.pop(module_name, None)
            raise RunPlanConfigurationError(f"cannot import plugin {module_path}: {exc}") from exc
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    try:
        return getattr(module, symbol_name)
    except AttributeError:
        raise RunPlanConfigurationError(f"plugin file {module_path} does not export {symbol_name!r}") from None


def load_symbol(reference: str, plan_root: Path) -> Any:
    """Load a symbol from a Python module or mounted plugin file."""
    if reference.startswith("file:"):
        return _load_file_symbol(reference, plan_root)

    module_name, separator, symbol_name = reference.rpartition(".")
    if not separator:
        raise RunPlanConfigurationError(f"class reference must be a dotted path or file reference: {reference!r}")
    module = _import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError:
        raise RunPlanConfigurationError(f"module {module_name!r} does not export {symbol_name!r}") from None


def _import_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RunPlanConfigurationError(f"cannot import {module_name!r}: {exc}") from exc


def _namespace_from_plan(task: TaskSpec, plan: RunPlan) -> str | None:
    # Datasets, models, and metrics are separate mappings, so one name may appear in more than one.
    lookups = ((task.dataset, plan.datasets), (task.model, plan.models), (task.metrics, plan.metrics))
    class_paths = [
        specs[name].class_path for reference, specs in lookups for name in _as_names(reference) if name in specs
    ]
    namespaces = {
        namespace for class_path in class_paths for namespace in _NAMESPACES if f".core.{namespace}." in class_path
    }
    if len(namespaces) > 1:
        raise RunPlanConfigurationError(f"task {task.name!r} mixes objects from different CheckMAITE problem types")
    return next(iter(namespaces), None)


def load_capability(reference: str, task: TaskSpec, plan: RunPlan, plan_root: Path) -> Any:
    """Load a capability class, including short built-in references."""
    if not reference.startswith("builtin:"):
        return load_symbol(reference, plan_root)

    builtin_reference = reference.removeprefix("builtin:")
    first_component, separator, remainder = builtin_reference.partition(".")
    if separator and first_component in _NAMESPACES:
        namespace = first_component
        symbol_name = remainder
    else:
        namespace = _namespace_from_plan(task, plan)
        symbol_name = builtin_reference

    if namespace is None:
        raise RunPlanConfigurationError(
            f"cannot infer the problem type for {reference!r}; use "
            "'builtin:object_detection.Name' or 'builtin:image_classification.Name'"
        )
    module = _import_module(f"checkmaite.core.{namespace}")
    try:
        return getattr(module, symbol_name)
    except AttributeError:
        raise RunPlanConfigurationError(
            f"CheckMAITE {namespace!r} does not export capability {symbol_name!r}"
        ) from None


def instantiate_object(
    spec: ObjectSpec,
    *,
    plan_root: Path,
    inject_device: bool,
    device: str | None = None,
) -> Any:
    """Instantiate one dataset, model, or metric specification."""
    constructor = load_symbol(spec.class_path, plan_root)
    if not callable(constructor):
        raise RunPlanConfigurationError(f"configured object is not callable: {spec.class_path}")

    kwargs = _instantiate_nested(spec.args, plan_root)
    if inject_device and device is not None and "device" not in kwargs and _accepts_keyword(constructor, "device"):
        kwargs["device"] = device
    check_arguments(constructor, kwargs, spec.class_path)
    return constructor(**kwargs)


def check_arguments(constructor: Any, kwargs: Mapping[str, Any], reference: str) -> None:
    """Reject keyword arguments the constructor cannot accept, before calling it."""
    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError):
        return
    try:
        signature.bind(**kwargs)
    except TypeError as exc:
        raise RunPlanConfigurationError(f"invalid arguments for {reference}: {exc}") from None


def _instantiate_nested(value: Any, plan_root: Path) -> Any:
    # Nested objects use ``_class`` so ordinary arguments may contain a ``class`` key.
    if isinstance(value, list):
        return [_instantiate_nested(item, plan_root) for item in value]
    if isinstance(value, dict):
        if set(value).issubset({"_class", "args"}) and "_class" in value:
            try:
                nested_spec = ObjectSpec.model_validate({"class": value["_class"], "args": value.get("args", {})})
            except ValidationError as exc:
                raise RunPlanConfigurationError(f"invalid nested object: {exc}") from exc
            return instantiate_object(nested_spec, plan_root=plan_root, inject_device=False)
        return {key: _instantiate_nested(item, plan_root) for key, item in value.items()}
    return value


def _accepts_keyword(callable_object: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(callable_object).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.name == name or parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)


def _as_names(reference: str | list[str] | None) -> list[str]:
    if reference is None:
        return []
    return [reference] if isinstance(reference, str) else reference
