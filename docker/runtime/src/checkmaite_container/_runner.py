"""Execute a finite CheckMAITE batch-container run plan."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, ValidationError
from threadpoolctl import threadpool_limits
from yaml import YAMLError

from checkmaite import cache_path
from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend
from checkmaite.core.capability_core import Capability, CapabilityConfigBase, CapabilityRunBase, _check_cardinality
from checkmaite.core.report import ArtifactReport, InlineTextReport
from checkmaite_container._loading import _as_names, check_arguments, instantiate_object, load_capability
from checkmaite_container._plan import (
    ObjectReferences,
    ObjectSpec,
    ResourceSpec,
    RunPlan,
    RunPlanConfigurationError,
    TaskSpec,
    ThreadCount,
    load_plan,
)

logger = logging.getLogger(__name__)


class TaskResult(BaseModel):
    """Durable locations and identifiers produced by one task."""

    model_config = ConfigDict(frozen=True)

    name: str
    run_uid: str
    capability_id: str
    run_file: str
    report_file: str | None = None
    report_uri: str | None = None


class RunResult(BaseModel):
    """Summary of a completed run plan."""

    model_config = ConfigDict(frozen=True)

    version: Literal[1] = 1
    tasks: list[TaskResult]
    analytics_directory: str


@dataclass(frozen=True)
class HostResources:
    """CPU and CUDA capacity visible to the current process."""

    available_cpus: int
    cuda_device_count: int


@dataclass(frozen=True)
class RuntimeResources:
    """Resolved resource policy used for one process."""

    available_cpus: int
    threads: int
    device: str


@dataclass(frozen=True)
class _PreparedTask:
    """Validated objects needed to execute one task."""

    spec: TaskSpec
    capability: Any
    config: CapabilityConfigBase
    datasets: list[Any]
    models: list[Any]
    metrics: list[Any]


def inspect_host_resources() -> HostResources:
    """Inspect the CPU and CUDA capacity visible to the current process."""
    cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return HostResources(
        available_cpus=effective_cpu_count(),
        cuda_device_count=cuda_device_count,
    )


def effective_cpu_count() -> int:
    """Return the CPU capacity visible through affinity and cgroup limits."""
    get_affinity = getattr(os, "sched_getaffinity", None)
    affinity_count = len(cast(set[int], get_affinity(0))) if callable(get_affinity) else (os.cpu_count() or 1)
    quota_count = _cgroup_cpu_count()
    if quota_count is None:
        return max(1, affinity_count)
    return max(1, min(affinity_count, quota_count))


def _cgroup_cpu_count() -> int | None:
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        quota, period = cpu_max.read_text(encoding="utf-8").strip().split()
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))

    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.is_file() and period_path.is_file():
        quota = int(quota_path.read_text(encoding="utf-8"))
        if quota > 0:
            period = int(period_path.read_text(encoding="utf-8"))
            return max(1, math.ceil(quota / period))
    return None


def resolve_resources(
    plan: RunPlan,
    host: HostResources,
    *,
    threads: ThreadCount | None = None,
    device: str | None = None,
) -> RuntimeResources:
    """Resolve CLI and plan resource requests against inspected hardware."""
    requested = ResourceSpec(
        threads=plan.resources.threads if threads is None else threads,
        device=plan.resources.device if device is None else device,
    )
    resolved_threads = (
        host.available_cpus if requested.threads == "auto" else min(requested.threads, host.available_cpus)
    )

    if requested.device == "auto":
        resolved_device = "cuda:0" if host.cuda_device_count > 0 else "cpu"
    elif requested.device.startswith("cuda"):
        if host.cuda_device_count == 0:
            raise RunPlanConfigurationError(
                f"CUDA device {requested.device!r} was requested, but CUDA is not available"
            )
        device_index = int(requested.device.partition(":")[2] or "0")
        if device_index >= host.cuda_device_count:
            raise RunPlanConfigurationError(
                f"CUDA device {requested.device!r} was requested, but only "
                f"{host.cuda_device_count} device(s) are available"
            )
        resolved_device = f"cuda:{device_index}"
    else:
        resolved_device = requested.device

    return RuntimeResources(
        available_cpus=host.available_cpus,
        threads=resolved_threads,
        device=resolved_device,
    )


def configure_process(resources: RuntimeResources) -> None:
    """Apply a resolved resource policy to the current Python process."""
    # The variables cover pools loaded later; OpenBLAS and OpenMP read them only
    # at load, so pools NumPy and Torch already loaded are limited directly.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(resources.threads)
    os.environ["CHECKMAITE_THREADS"] = str(resources.threads)
    os.environ["CHECKMAITE_DEVICE"] = resources.device
    threadpool_limits(limits=resources.threads)
    torch.set_num_threads(resources.threads)


def run_plan(
    plan_path: str | Path,
    *,
    output_directory: str | Path,
    cache_directory: str | Path,
    threads: ThreadCount | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> RunResult:
    """Load and execute a finite run plan, then persist its outputs."""
    resolved_plan_path = Path(plan_path).resolve()
    output_path = Path(output_directory).resolve()
    # A summary from an earlier run must not survive a run that fails.
    (output_path / "run-results.json").unlink(missing_ok=True)
    try:
        plan = load_plan(resolved_plan_path)
    except RunPlanConfigurationError:
        raise
    except (FileNotFoundError, ValueError, YAMLError) as exc:
        raise RunPlanConfigurationError(str(exc)) from exc
    # Host inspection is outside the handler above: a host fault is not a bad plan.
    resources = resolve_resources(plan, inspect_host_resources(), threads=threads, device=device)
    configure_process(resources)
    cache_path_value = Path(cache_directory).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path(cache_path_value)

    logger.info(
        "Starting plan with %d CPU(s), %d thread(s), and device %s",
        resources.available_cpus,
        resources.threads,
        resources.device,
    )

    with _working_directory(resolved_plan_path.parent):
        prepared_tasks = _prepare_tasks(
            plan,
            plan_root=resolved_plan_path.parent,
            device=resources.device,
            batch_size=batch_size,
        )

        runs: list[CapabilityRunBase[Any, Any]] = []
        task_results: list[TaskResult] = []
        for prepared in prepared_tasks:
            task = prepared.spec
            logger.info("Running task %s", task.name)
            run = prepared.capability.run(
                datasets=prepared.datasets,
                models=prepared.models,
                metrics=prepared.metrics,
                config=prepared.config,
                use_cache=task.use_cache,
            )
            if not isinstance(run, CapabilityRunBase):
                raise TypeError(f"task {task.name!r} returned {type(run).__name__}, expected CapabilityRunBase")
            runs.append(run)
            task_results.append(_write_task_outputs(task.name, run, task.report_threshold, output_path))
            logger.info("Completed task %s with run UID %s", task.name, run.run_uid)

    analytics_directory = output_path / "analytics"
    analytics_directory.mkdir(parents=True, exist_ok=True)
    AnalyticsStore(ParquetBackend(str(analytics_directory))).write(runs)
    result = RunResult(
        tasks=task_results,
        analytics_directory=str(analytics_directory),
    )
    _write_text_atomic(output_path / "run-results.json", result.model_dump_json(indent=2) + "\n")
    logger.info("Completed %d task(s); results written to %s", len(task_results), output_path)
    return result


def _prepare_tasks(
    plan: RunPlan,
    *,
    plan_root: Path,
    device: str,
    batch_size: int | None,
) -> list[_PreparedTask]:
    """Load and validate every referenced object before execution starts."""
    # Check everything the plan can get wrong before any dataset, model, or metric is built.
    configured: list[tuple[TaskSpec, Any, CapabilityConfigBase]] = []
    for task in plan.tasks:
        _check_known_names(task.dataset, plan.datasets, "dataset")
        _check_known_names(task.model, plan.models, "model")
        _check_known_names(task.metrics, plan.metrics, "metric")
        capability_type = load_capability(task.capability, task, plan, plan_root)
        if not callable(capability_type):
            raise RunPlanConfigurationError(f"configured capability is not callable: {task.capability}")
        check_arguments(capability_type, task.capability_args, task.capability)
        capability: Any = capability_type(**task.capability_args)
        _check_counts(task, capability)
        configured.append((task, capability, _build_capability_config(capability, task.config, batch_size)))

    datasets = _instantiate_referenced(
        plan.datasets, [task.dataset for task in plan.tasks], "dataset", plan_root, device, inject_device=False
    )
    models = _instantiate_referenced(
        plan.models, [task.model for task in plan.tasks], "model", plan_root, device, inject_device=True
    )
    metrics = _instantiate_referenced(
        plan.metrics, [task.metrics for task in plan.tasks], "metric", plan_root, device, inject_device=False
    )
    return [
        _PreparedTask(
            spec=task,
            capability=capability,
            config=config,
            datasets=_select_objects(task.dataset, datasets),
            models=_select_objects(task.model, models),
            metrics=_select_objects(task.metrics, metrics),
        )
        for task, capability, config in configured
    ]


def _check_known_names(reference: ObjectReferences, specs: dict[str, ObjectSpec], kind: str) -> None:
    for name in _as_names(reference):
        if name not in specs:
            raise RunPlanConfigurationError(f"run plan references unknown {kind} {name!r}")


def _check_counts(task: TaskSpec, capability: Any) -> None:
    """Reject a task whose object counts the capability cannot accept, before anything runs."""
    if not isinstance(capability, Capability):
        return
    counts = (
        ("dataset", capability.supports_datasets, task.dataset),
        ("model", capability.supports_models, task.model),
        ("metric", capability.supports_metrics, task.metrics),
    )
    try:
        for label, required, reference in counts:
            _check_cardinality(owner_id=capability.id, label=label, required=required, n=len(_as_names(reference)))
    except TypeError as exc:
        raise RunPlanConfigurationError(f"task {task.name!r}: {exc}") from None


def _build_capability_config(
    capability: Any,
    config_values: dict[str, Any],
    batch_size: int | None,
) -> CapabilityConfigBase:
    create_config = getattr(capability, "_create_config", None)
    if not callable(create_config):
        raise RunPlanConfigurationError(
            f"capability {type(capability).__name__} does not provide a CheckMAITE configuration"
        )
    default_config = create_config()
    if not isinstance(default_config, CapabilityConfigBase):
        raise RunPlanConfigurationError(
            f"capability {type(capability).__name__} returned an invalid configuration object"
        )

    values = dict(config_values)
    if batch_size is not None and "batch_size" in type(default_config).model_fields:
        values["batch_size"] = batch_size
    try:
        return type(default_config).model_validate(values)
    except ValidationError as exc:
        raise RunPlanConfigurationError(f"invalid config for {type(capability).__name__}: {exc}") from exc


def _instantiate_referenced(
    specs: dict[str, ObjectSpec],
    references: list[ObjectReferences],
    kind: str,
    plan_root: Path,
    device: str,
    *,
    inject_device: bool,
) -> dict[str, Any]:
    """Instantiate, once each, only the objects that some task references."""
    names = list(dict.fromkeys(name for reference in references for name in _as_names(reference)))
    unused = sorted(set(specs) - set(names))
    if unused:
        logger.warning("Skipping unused %s(s): %s", kind, ", ".join(unused))
    return {
        name: instantiate_object(specs[name], plan_root=plan_root, device=device, inject_device=inject_device)
        for name in names
    }


def _select_objects(reference: ObjectReferences, objects: dict[str, Any]) -> list[Any]:
    return [objects[name] for name in _as_names(reference)]


def _write_task_outputs(
    task_name: str,
    run: CapabilityRunBase[Any, Any],
    report_threshold: float,
    output_path: Path,
) -> TaskResult:
    task_directory = output_path / "tasks" / task_name
    task_directory.mkdir(parents=True, exist_ok=True)
    run_file = task_directory / "run.json"
    _write_text_atomic(run_file, run.model_dump_json(indent=2) + "\n")

    report_file: Path | None = None
    report_uri: str | None = None
    try:
        report = run.collect_md_report(threshold=report_threshold)
    except NotImplementedError:
        report = None
    if isinstance(report, InlineTextReport):
        report_file = task_directory / report.filename
        _write_text_atomic(report_file, report.content)
    elif isinstance(report, ArtifactReport):
        report_uri = report.uri

    return TaskResult(
        name=task_name,
        run_uid=run.run_uid,
        capability_id=run.capability_id,
        run_file=str(run_file),
        report_file=str(report_file) if report_file is not None else None,
        report_uri=report_uri,
    )


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
