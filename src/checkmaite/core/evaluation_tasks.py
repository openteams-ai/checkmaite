"""Registry of the evaluation tasks CheckMAITE can run.

An evaluation task is a kind of AI problem, such as image classification or
object detection. Each task is described by an :class:`EvaluationTask`, which
tells the task-independent core what it needs to know: the task's name, where
its capabilities live, and how to store its predictions and targets in the
cache.

The built-in tasks register themselves when their namespaces are imported.
Other packages add tasks without changing CheckMAITE by declaring an entry
point in the ``checkmaite.tasks`` group that resolves to an
:class:`EvaluationTask`::

    [project.entry-points."checkmaite.tasks"]
    semantic_segmentation = "my_package.segmentation:TASK"

Entry points are loaded the first time a task is looked up. A task loaded
this way must set ``plugin_api_version``; one whose major version differs
from CheckMAITE's ``PLUGIN_API_VERSION`` is not registered.
"""

import importlib.metadata as importlib_metadata
import logging
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter, ValidationError

from checkmaite.core._plugins import PluginRecord, _check_api_version, _registry

logger = logging.getLogger(__name__)

TASK_ENTRY_POINT_GROUP = "checkmaite.tasks"

__all__ = [
    "TASK_ENTRY_POINT_GROUP",
    "EvaluationTask",
    "get_task",
    "list_tasks",
    "register_task",
]


T_TaskTarget = TypeVar("T_TaskTarget")


@dataclass(frozen=True, kw_only=True)
class EvaluationTask(Generic[T_TaskTarget]):
    """Description of an evaluation task that the core needs in order to run it.

    Parameters
    ----------
    name
        Stable identifier for the task. Use the MAITE protocol module name when
        one exists (``maite.protocols.<name>``), for example ``"object_detection"``.
    namespace
        Import path of the module that exports the task's capabilities, for
        example ``"checkmaite.core.object_detection"``.
    target_adapter
        Pydantic ``TypeAdapter`` that validates one model prediction or one
        dataset target before it is cached. On a cache hit, predictions and
        targets come back as the adapter's type. Any type pydantic can validate
        works, including unions such as image classification's ``ArrayLike``.
        Annotate the task with the MAITE target protocol, for example
        ``EvaluationTask[SegmentationTarget]``, so the type checker verifies
        that the adapter's type satisfies it. Values are validated with
        ``from_attributes=True``, so a ``CapabilityOutputsBase`` subclass whose
        array fields are MAITE ``ArrayLike`` accepts any object with matching
        attributes and stores the arrays as binary sidecars.
        Like MAITE, a task has one type for both predictions and targets.
    aliases
        Other names accepted by :func:`get_task`, such as DataMAITE's short task
        codes (``"od"``).
    materialize_inputs
        Whether model inputs can be one-shot iterables, such as streams of video
        frames, that must be turned into lists before they are returned to the
        caller.
    plugin_api_version
        The plugin API version the task was written and tested against, as a
        string such as ``"1.0.0"``. Required for tasks loaded from a
        ``checkmaite.tasks`` entry point; its major version must match
        CheckMAITE's ``PLUGIN_API_VERSION``. Do not set it by importing
        ``PLUGIN_API_VERSION``, which always matches the installed CheckMAITE.
    """

    name: str
    namespace: str
    target_adapter: TypeAdapter[T_TaskTarget]
    aliases: tuple[str, ...] = ()
    materialize_inputs: bool = False
    plugin_api_version: str | None = None

    def validate_target_batches(self, batches: Sequence[Sequence[object]]) -> list[list[T_TaskTarget]]:
        """Validate batches of predictions or targets for caching.

        Parameters
        ----------
        batches
            Batches of model predictions or dataset targets for this task.

        Returns
        -------
        list[list[T_TaskTarget]]
            The batches, converted by ``target_adapter``.

        Raises
        ------
        pydantic.ValidationError
            If a prediction or target does not match ``target_adapter``.
        """
        validate = self.target_adapter.validate_python
        return [[validate(value, from_attributes=True) for value in batch] for batch in batches]

    def _matches(self, values: Sequence[object]) -> bool:
        try:
            self.validate_target_batches([values])
        except ValidationError:
            return False
        except Exception:  # noqa: BLE001 - e.g. a target class whose constructor rejects the copied fields
            logger.debug("Evaluation task %r could not validate these values.", self.name, exc_info=True)
            return False
        return True

    def _require_batch(self, predictions: Sequence[object], targets: Sequence[object]) -> None:
        """Raise if a batch of predictions or targets does not belong to this task."""
        for name, values in (("predictions", predictions), ("targets", targets)):
            if len(values) > 0 and not self._matches(values):
                example = type(values[0]).__name__
                raise ValueError(
                    f"The {name} do not match evaluation task {self.name!r} (got {example}). "
                    f"Pass the task of the model and dataset as task=; registered tasks: "
                    f"{', '.join(task.name for task in list_tasks())}."
                )


_tasks: dict[str, EvaluationTask] = {}
_aliases: dict[str, str] = {}
_entry_points_loaded = False


def register_task(task: EvaluationTask) -> EvaluationTask:
    """Register an evaluation task so the core can look it up by name.

    Registering the same task, or an equal one, again has no effect and returns
    the task already registered.

    Parameters
    ----------
    task
        The task to register.

    Returns
    -------
    EvaluationTask
        The registered task, which is the earlier object when an equal task
        was already registered.

    Raises
    ------
    TypeError
        If ``task`` is not an :class:`EvaluationTask` or its ``target_adapter``
        is not a complete pydantic ``TypeAdapter``.
    ValueError
        If a different task already uses the task's name or one of its aliases.
    """
    if not isinstance(task, EvaluationTask):
        raise TypeError(f"Expected an EvaluationTask, got {type(task).__name__}.")
    registered = _tasks.get(task.name)
    if registered is not None and registered == task:
        return registered
    for name in (task.name, *task.aliases):
        owner = _tasks.get(name) or _tasks.get(_aliases.get(name, ""))
        if owner is not None:
            raise ValueError(f"Evaluation task name {name!r} is already used by task {owner.name!r}.")
    _check_target_adapter(task)
    _tasks[task.name] = task
    _aliases.update(dict.fromkeys(task.aliases, task.name))
    return task


def _check_target_adapter(task: EvaluationTask) -> None:
    # A malformed adapter would otherwise fail on every call, so it is rejected
    # before the task is registered.
    adapter = task.target_adapter
    if not isinstance(adapter, TypeAdapter):
        kind = type(adapter).__name__
        raise TypeError(f"Evaluation task {task.name!r} target_adapter must be a pydantic TypeAdapter, got {kind}.")
    try:
        # An adapter with unresolved forward references is incomplete until rebuilt.
        adapter.rebuild(raise_errors=True)
    except Exception as error:
        raise TypeError(f"Evaluation task {task.name!r} target_adapter cannot validate: {error}") from error


def _plugin_api_version_error(task: EvaluationTask, entry_point_name: str) -> str | None:
    if task.plugin_api_version is None:
        logger.warning(
            "Evaluation task entry point '%s' in group '%s' does not set 'plugin_api_version'.",
            entry_point_name,
            TASK_ENTRY_POINT_GROUP,
        )
        return "Evaluation task must set 'plugin_api_version'"
    # Logs its own warning, like for capability plugins.
    return _check_api_version(task.plugin_api_version, entry_point_name, TASK_ENTRY_POINT_GROUP)


def _load_entry_point(entry_point: Any) -> None:
    package_name = entry_point.dist.name if entry_point.dist else None
    task: Any = None
    try:
        task = entry_point.load()
        if not isinstance(task, EvaluationTask):
            raise TypeError(f"Entry point resolved to {type(task).__name__}, expected an EvaluationTask.")
        error = _plugin_api_version_error(task, entry_point.name)
        if error is None:
            register_task(task)
    except Exception:  # noqa: BLE001 - one broken plugin must not hide the other tasks
        logger.warning(
            "Failed to load evaluation task entry point '%s' from group '%s'.",
            entry_point.name,
            TASK_ENTRY_POINT_GROUP,
            exc_info=True,
        )
        error = traceback.format_exc()
    _registry.append(
        PluginRecord(
            group=TASK_ENTRY_POINT_GROUP,
            entry_point_name=entry_point.name,
            package_name=package_name,
            symbols=[task.name] if error is None else [],
            status="loaded" if error is None else "failed",
            error=error,
        )
    )


def _load_entry_points() -> None:
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True

    for entry_point in importlib_metadata.entry_points(group=TASK_ENTRY_POINT_GROUP):
        _load_entry_point(entry_point)


def get_task(name: str) -> EvaluationTask:
    """Return the registered evaluation task with this name or alias.

    Raises
    ------
    KeyError
        If no registered task uses the name.
    """
    _load_entry_points()
    task = _tasks.get(name) or _tasks.get(_aliases.get(name, ""))
    if task is None:
        known = ", ".join(sorted(_tasks)) or "none"
        raise KeyError(f"Unknown evaluation task {name!r}. Registered tasks: {known}.")
    return task


def list_tasks() -> list[EvaluationTask]:
    """Return every registered evaluation task, in registration order."""
    _load_entry_points()
    return list(_tasks.values())
