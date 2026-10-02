"""Validated run-plan models for the CheckMAITE batch container."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, PositiveInt, model_validator

_DEVICE_PATTERN = r"auto|cpu|cuda(?::\d+)?"


def _validate_device(value: str) -> str:
    if re.fullmatch(_DEVICE_PATTERN, value) is None:
        raise ValueError("device must be 'auto', 'cpu', or a CUDA device such as 'cuda:0'")
    return value


ThreadCount = Literal["auto"] | PositiveInt
DeviceRequest = Annotated[
    str,
    AfterValidator(_validate_device),
    Field(json_schema_extra={"pattern": f"^(?:{_DEVICE_PATTERN})$"}),
]
ObjectReferences = str | list[str] | None


class RunPlanConfigurationError(ValueError):
    """A run plan could not be loaded or prepared for execution."""


class ObjectSpec(BaseModel):
    """Import reference and constructor arguments for a runtime object."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_path: str = Field(alias="class", min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


class ResourceSpec(BaseModel):
    """Resources requested by a run plan."""

    model_config = ConfigDict(extra="forbid")

    threads: ThreadCount = "auto"
    device: DeviceRequest = "auto"


class TaskSpec(BaseModel):
    """One finite capability invocation in a run plan."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    capability: str = Field(min_length=1)
    capability_args: dict[str, Any] = Field(default_factory=dict)
    dataset: ObjectReferences = None
    model: ObjectReferences = None
    metrics: ObjectReferences = None
    config: dict[str, Any] = Field(default_factory=dict)
    use_cache: bool = True
    report_threshold: float = 0.5


class RunPlan(BaseModel):
    """A complete finite CheckMAITE container run."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    datasets: dict[str, ObjectSpec] = Field(default_factory=dict)
    models: dict[str, ObjectSpec] = Field(default_factory=dict)
    metrics: dict[str, ObjectSpec] = Field(default_factory=dict)
    tasks: list[TaskSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_task_names(self) -> RunPlan:
        names = [task.name for task in self.tasks]
        if len(names) != len(set(names)):
            raise ValueError("task names must be unique")
        return self


def load_plan(path: str | Path) -> RunPlan:
    """Read and validate a YAML run plan."""
    plan_path = Path(path)
    with plan_path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("run plan must contain a YAML mapping")
    return RunPlan.model_validate(payload)
