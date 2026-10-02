"""Command-line interface for finite CheckMAITE container runs."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import PositiveInt, TypeAdapter, ValidationError

from checkmaite_container._plan import DeviceRequest, RunPlanConfigurationError, ThreadCount

_DESCRIPTION = "Run one finite CheckMAITE plan on the CPU and GPU resources visible to this container."
_OPERATIONAL_HELP = """command syntax:
  checkmaite-container run [--config PLAN] [--output DIR] [--cache DIR]
                           [--secrets DIR]
                           [--threads auto|N]
                           [--device auto|cpu|cuda|cuda:N]
                           [--batch-size N] [--log-level LEVEL]

container defaults:
  With no arguments, the container runs: run
  The default plan is /checkmaite/run.yaml.

required environment variables:
  None.

optional environment variables:
  CHECKMAITE_CONFIG       Default for --config (default: /checkmaite/run.yaml).
  CHECKMAITE_OUTPUT_DIR   Default for --output (default: /output/results).
  CHECKMAITE_CACHE_DIR    Default for --cache (default: /cache).
  CHECKMAITE_SECRETS_DIR  Default for --secrets (default: /run/secrets).
  CHECKMAITE_THREADS      Default for --threads (default: plan, then auto).
  CHECKMAITE_DEVICE       Default for --device (default: plan, then auto).
  CHECKMAITE_BATCH_SIZE   Default for --batch-size (default: task/capability).
  CHECKMAITE_LOG_LEVEL    Default for --log-level (default: INFO).

volume mounts:
  /checkmaite   Read-only run plan, data, models, and trusted plugins.
  /output       Writable durable reports, analytics, and result manifests.
  /cache        Writable reusable caches and temporary files.
  /run/secrets  Optional read-only secret files consumed by trusted plugins.

The built-in runtime requires no secrets. Plugins that require secrets must read
files from the directory in CHECKMAITE_SECRETS_DIR, which the runtime sets from
--secrets. Do not put secrets in environment variables, run plans, or the
container. /output and /cache must be writable by UID:GID
10001:10001.
"""
_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def _validated(annotation: Any, message: str) -> Callable[[str], Any]:
    """Return an argparse type that validates with the run-plan rules."""
    adapter: TypeAdapter[Any] = TypeAdapter(annotation)

    def parse(value: str) -> Any:
        try:
            return adapter.validate_python(value)
        except ValidationError:
            raise argparse.ArgumentTypeError(message) from None

    return parse


_thread_count = _validated(ThreadCount, "threads must be 'auto' or a positive integer")
_device = _validated(DeviceRequest, "device must be auto, cpu, cuda, or cuda:N")
_positive_batch_size = _validated(PositiveInt, "batch size must be a positive integer")


def _log_level(value: str) -> str:
    normalized = value.upper()
    if normalized not in _LOG_LEVELS:
        raise argparse.ArgumentTypeError(
            "log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL",
        )
    return normalized


def build_parser(environment: Mapping[str, str] | None = None) -> argparse.ArgumentParser:
    """Build the batch-container argument parser."""
    runtime_environment = os.environ if environment is None else environment
    parser = argparse.ArgumentParser(
        prog="checkmaite-container",
        description=_DESCRIPTION,
        epilog=_OPERATIONAL_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('checkmaite')}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="run one finite CheckMAITE plan",
        description=_DESCRIPTION,
        epilog=_OPERATIONAL_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--config",
        default=runtime_environment.get("CHECKMAITE_CONFIG", "/checkmaite/run.yaml"),
        help="path to the YAML run plan (default: CHECKMAITE_CONFIG, then /checkmaite/run.yaml)",
    )
    run_parser.add_argument(
        "--output",
        default=runtime_environment.get("CHECKMAITE_OUTPUT_DIR", "/output/results"),
        help="directory for reports and analytics (default: CHECKMAITE_OUTPUT_DIR, then /output/results)",
    )
    run_parser.add_argument(
        "--cache",
        default=runtime_environment.get("CHECKMAITE_CACHE_DIR", "/cache"),
        help="directory for reusable cache data (default: CHECKMAITE_CACHE_DIR, then /cache)",
    )
    run_parser.add_argument(
        "--secrets",
        default=runtime_environment.get("CHECKMAITE_SECRETS_DIR", "/run/secrets"),
        help="directory of secret files for plugins (default: CHECKMAITE_SECRETS_DIR, then /run/secrets)",
    )
    run_parser.add_argument(
        "--threads",
        type=_thread_count,
        default=runtime_environment.get("CHECKMAITE_THREADS"),
        help="process thread budget: auto or a positive integer (default: environment, plan, then auto)",
    )
    run_parser.add_argument(
        "--device",
        type=_device,
        default=runtime_environment.get("CHECKMAITE_DEVICE"),
        help="device override: auto, cpu, cuda, or cuda:N (default: environment, plan, then auto)",
    )
    run_parser.add_argument(
        "--batch-size",
        type=_positive_batch_size,
        default=runtime_environment.get("CHECKMAITE_BATCH_SIZE"),
        help="positive override for compatible tasks (default: environment, then task/capability)",
    )
    run_parser.add_argument(
        "--log-level",
        type=_log_level,
        default=runtime_environment.get("CHECKMAITE_LOG_LEVEL", "INFO"),
        help="logging level (default: CHECKMAITE_LOG_LEVEL, then INFO)",
    )
    return parser


def _container_arguments(argv: Sequence[str]) -> list[str]:
    """Apply the container entrypoint's default-command behavior."""
    arguments = list(argv)
    if not arguments:
        return ["run"]
    if arguments[0] != "--version" and arguments[0].startswith("-"):
        return ["run", *arguments]
    return arguments


def _cache_environment(cache_directory: str | Path) -> dict[str, Path]:
    """Return the library cache and temporary-file locations under a cache directory."""
    cache = Path(cache_directory).resolve()
    return {
        "HOME": cache,
        "TMPDIR": cache / "tmp",
        "XDG_CACHE_HOME": cache / ".cache",
        "MPLCONFIGDIR": cache / "matplotlib",
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
    }


def _configure_cache_environment(cache_directory: str | Path) -> None:
    """Point common library caches and temporary files at the selected cache."""
    for name, location in _cache_environment(cache_directory).items():
        if name != "HOME" and name in os.environ:
            continue
        location.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(location)


def container_main(argv: Sequence[str] | None = None) -> int:
    """Run the shell-free product-container entrypoint."""
    return main(_container_arguments(sys.argv[1:] if argv is None else argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CheckMAITE batch-container command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=_LOG_LEVELS[args.log_level],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    managed_names = [*_cache_environment(args.cache), "CHECKMAITE_SECRETS_DIR"]
    previous_environment = {name: os.environ.get(name) for name in managed_names}
    try:
        # The directory is optional, so it is not required to exist.
        os.environ["CHECKMAITE_SECRETS_DIR"] = str(Path(args.secrets).resolve())
        try:
            _configure_cache_environment(args.cache)
        except OSError as exc:
            logging.getLogger(__name__).error("Invalid cache directory: %s", exc)
            return 2

        try:
            # Delay the heavy runtime import until cache locations are resolved.
            from checkmaite_container._runner import run_plan

            run_plan(
                args.config,
                output_directory=args.output,
                cache_directory=args.cache,
                threads=args.threads,
                device=args.device,
                batch_size=args.batch_size,
            )
        except RunPlanConfigurationError as exc:
            logging.getLogger(__name__).error("Invalid run configuration: %s", exc)
            return 2
        except Exception:
            logging.getLogger(__name__).exception("CheckMAITE run failed")
            return 1
        return 0
    finally:
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
