"""Run a behavioral smoke plan inside a hardened product container."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import subprocess  # nosec B404 - invokes the fixed container entry point
from pathlib import Path

import torch
import yaml

_PLUGIN = """
from checkmaite.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
)
from checkmaite.core.report import InlineTextReport


class SmokeDataset:
    metadata = {"id": "smoke-dataset"}


class SmokeModel:
    metadata = {"id": "smoke-model"}

    def __init__(self, device="missing"):
        self.device = device


class SmokeMetric:
    metadata = {"id": "smoke-metric"}


class SmokeConfig(CapabilityConfigBase):
    batch_size: int = 1


class SmokeOutputs(CapabilityOutputsBase):
    device: str
    batch_size: int


class SmokeRun(CapabilityRunBase[SmokeConfig, SmokeOutputs]):
    def collect_md_report(self, threshold):
        return InlineTextReport(
            media_type="text/markdown",
            filename="report.md",
            content=f"# Container smoke test\\n\\nThreshold: {threshold}",
        )


class SmokeCapability(Capability):
    _RUN_TYPE = SmokeRun

    @classmethod
    def _create_config(cls):
        return SmokeConfig()

    @property
    def supports_datasets(self):
        return Number.ONE

    @property
    def supports_models(self):
        return Number.ONE

    @property
    def supports_metrics(self):
        return Number.ONE

    def _run(self, models, datasets, metrics, config, use_prediction_and_evaluation_cache):
        assert datasets and metrics
        return SmokeOutputs(device=models[0].device, batch_size=config.batch_size)
"""


def _installed_versions() -> dict[str, str]:
    return {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }


def _require_packages(versions: dict[str, str], variant: str, packages: tuple[str, ...]) -> None:
    for package in packages:
        if package not in versions:
            raise RuntimeError(f"{variant} container is missing {package}")


def _verify_cpu(versions: dict[str, str]) -> None:
    _require_packages(versions, "CPU", ("onnxruntime",))
    if not versions["torch"].endswith("+cpu") or not versions["torchvision"].endswith("+cpu"):
        raise RuntimeError("CPU container does not contain CPU-only PyTorch packages")
    if torch.version.cuda is not None:
        raise RuntimeError("CPU PyTorch unexpectedly reports CUDA support")
    prohibited = sorted(package for package in versions if package == "triton" or package.startswith("nvidia-"))
    if prohibited:
        raise RuntimeError(f"CPU container contains CUDA packages: {prohibited}")


def _verify_cuda(versions: dict[str, str]) -> None:
    _require_packages(versions, "CUDA", ("onnxruntime-gpu", "triton"))
    if not versions["torch"].endswith("+cu130") or not versions["torchvision"].endswith("+cu130"):
        raise RuntimeError("CUDA container does not contain CUDA 13.0 PyTorch packages")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"CUDA PyTorch reports CUDA {torch.version.cuda!r}, expected '13.0'")
    if not any(package.startswith("nvidia-") for package in versions):
        raise RuntimeError("CUDA container is missing NVIDIA Python packages")
    if "onnxruntime" in versions:
        raise RuntimeError("CUDA container contains the CPU ONNX Runtime package")


def _verify_variant(variant: str) -> None:
    versions = _installed_versions()
    _require_packages(versions, variant, ("datamaite", "modelmaite", "torch", "torchvision"))
    if variant == "cpu":
        _verify_cpu(versions)
    elif variant == "cuda":
        _verify_cuda(versions)
    else:
        raise ValueError(f"unsupported container variant: {variant}")


def _verify_runtime_libraries() -> None:
    """Exercise the installed CheckMAITE, MAITE, PyTorch, and ONNX Runtime stack."""
    for namespace in ("image_classification", "object_detection"):
        importlib.import_module(f"checkmaite.core.{namespace}")

    loaded = 0
    for entry_point in importlib.metadata.distribution("checkmaite").entry_points:
        if entry_point.group.startswith("maite."):
            entry_point.load()
            loaded += 1
    if loaded == 0:
        raise RuntimeError("container does not advertise CheckMAITE MAITE entry points")

    if torch.matmul(torch.ones(2, 2), torch.ones(2, 2)).sum().item() != 8:
        raise RuntimeError("PyTorch returned an unexpected matrix product")

    import onnxruntime

    if not onnxruntime.get_available_providers():
        raise RuntimeError("ONNX Runtime reports no execution providers")


def _verify_ray() -> None:
    """Run a local Ray task that needs the runtime-env agent.

    The image removes Ray's vendored aiohttp, so the agent must start on the
    locked aiohttp instead. A task with a ``runtime_env`` only runs once the
    agent is up, which ``import ray`` alone does not check.
    """
    import ray

    # The root filesystem is read-only and /dev/shm is small, so keep Ray's
    # session files and object store under the writable cache.
    ray_directory = Path("/cache/ray")
    plasma_directory = Path("/cache/ray-plasma")
    plasma_directory.mkdir(parents=True, exist_ok=True)
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    ray.init(
        num_cpus=1,
        include_dashboard=False,
        object_store_memory=100 * 1024**2,
        _temp_dir=str(ray_directory),
        _plasma_directory=str(plasma_directory),
    )
    try:

        @ray.remote
        def read_environment() -> str | None:
            return os.environ.get("CHECKMAITE_RAY_SMOKE")

        task = read_environment.options(
            runtime_env={"env_vars": {"CHECKMAITE_RAY_SMOKE": "ok"}},
        ).remote()
        value = ray.get(task, timeout=180)
    finally:
        ray.shutdown()
    if value != "ok":
        raise RuntimeError(f"Ray runtime_env task returned {value!r}, expected 'ok'")


def main() -> int:
    """Create and execute a real temporary run plan without network access."""
    variant = os.environ["CHECKMAITE_CONTAINER_VARIANT"]
    # The root filesystem is read-only. Like the entry point, keep temporary
    # files under the writable cache before importing CheckMAITE here.
    temporary_directory = Path("/cache/tmp")
    temporary_directory.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary_directory)
    _verify_variant(variant)
    _verify_runtime_libraries()
    _verify_ray()

    fixture_directory = Path("/cache/container-smoke")
    fixture_directory.mkdir(parents=True, exist_ok=True)
    plugin_path = fixture_directory / "plugin.py"
    plan_path = fixture_directory / "run.yaml"
    plugin_path.write_text(_PLUGIN, encoding="utf-8")

    reference = "file:plugin.py"
    plan = {
        "version": 1,
        "resources": {"threads": 1, "device": "cpu"},
        "datasets": {
            "data": {"class": f"{reference}:SmokeDataset"},
        },
        "models": {
            "model": {"class": f"{reference}:SmokeModel"},
        },
        "metrics": {
            "metric": {"class": f"{reference}:SmokeMetric"},
        },
        "tasks": [
            {
                "name": "evaluation",
                "capability": f"{reference}:SmokeCapability",
                "dataset": "data",
                "model": "model",
                "metrics": "metric",
                "config": {"batch_size": 2},
                "use_cache": False,
            },
        ],
    }
    plan_path.write_text(yaml.safe_dump(plan), encoding="utf-8")

    completed = subprocess.run(  # noqa: S603  # nosec B603 - fixed executable
        [
            "/opt/venv/bin/checkmaite-container",
            "run",
            "--config",
            str(plan_path),
            "--output",
            "/output/results",
            "--cache",
            "/cache",
        ],
        check=False,
    )
    if completed.returncode != 0:
        return completed.returncode

    result = json.loads(
        Path("/output/results/run-results.json").read_text(encoding="utf-8"),
    )
    if len(result["tasks"]) != 1 or result["tasks"][0]["name"] != "evaluation":
        raise RuntimeError("container smoke result did not contain the expected task")
    if not Path("/output/results/tasks/evaluation/report.md").is_file():
        raise RuntimeError("container smoke run did not write its report")
    if not Path(result["analytics_directory"]).is_dir():
        raise RuntimeError("container smoke run did not create its analytics directory")
    run = json.loads(
        Path("/output/results/tasks/evaluation/run.json").read_text(encoding="utf-8"),
    )
    if run["outputs"] != {"device": "cpu", "batch_size": 2}:
        raise RuntimeError(f"container smoke run produced unexpected outputs: {run['outputs']}")
    print(f"{variant} product container smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
