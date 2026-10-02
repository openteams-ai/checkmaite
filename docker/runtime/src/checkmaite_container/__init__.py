"""Finite batch-container execution for CheckMAITE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from checkmaite_container._plan import ObjectSpec, ResourceSpec, RunPlan, TaskSpec, load_plan

if TYPE_CHECKING:
    from checkmaite_container._runner import RunResult, TaskResult, run_plan


__all__ = [
    "ObjectSpec",
    "ResourceSpec",
    "RunPlan",
    "RunResult",
    "TaskResult",
    "TaskSpec",
    "load_plan",
    "run_plan",
]


def __getattr__(name: str) -> Any:
    """Load the torch-dependent runner only when execution is requested."""
    if name in {"RunResult", "TaskResult", "run_plan"}:
        from checkmaite_container import _runner

        return getattr(_runner, name)
    raise AttributeError(name)
