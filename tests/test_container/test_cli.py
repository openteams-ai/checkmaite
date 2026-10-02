import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from checkmaite import __version__
from checkmaite_container import cli


def _write_plan(tmp_path: Path, capability: str = "FakeCapability") -> Path:
    def reference(name: str) -> str:
        return f"file:plugin.py:{name}"

    payload = {
        "version": 1,
        "resources": {"threads": 1, "device": "cpu"},
        "datasets": {"data": {"class": reference("FakeDataset")}},
        "models": {"model": {"class": reference("FakeModel")}},
        "metrics": {"metric": {"class": reference("FakeMetric")}},
        "tasks": [
            {
                "name": "evaluation",
                "capability": reference(capability),
                "dataset": "data",
                "model": "model",
                "metrics": "metric",
                "use_cache": False,
            }
        ],
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], ["run"]),
        (["--threads", "2"], ["run", "--threads", "2"]),
        (["--version"], ["--version"]),
        (["run", "--config", "plan.yaml"], ["run", "--config", "plan.yaml"]),
    ],
)
def test_container_arguments(arguments: list[str], expected: list[str]) -> None:
    assert cli._container_arguments(arguments) == expected


@pytest.mark.parametrize(("value", "expected"), [("auto", "auto"), ("3", 3)])
def test_parser_accepts_thread_count(value: str, expected: str | int) -> None:
    args = cli.build_parser({}).parse_args(["run", "--threads", value])

    assert args.threads == expected


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_parser_rejects_invalid_thread_count(value: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.build_parser({}).parse_args(["run", "--threads", value])

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("option", "value"),
    [("--device", "gpu"), ("--device", "cuda:x"), ("--batch-size", "0"), ("--batch-size", "many")],
)
def test_parser_rejects_invalid_device_and_batch_size(option: str, value: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.build_parser({}).parse_args(["run", option, value])

    assert error.value.code == 2


def test_main_exports_secrets_directory_while_running(
    tmp_path: Path,
    plugin_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert plugin_file.is_file()
    seen: list[str | None] = []

    def fake_run_plan(*args: object, **kwargs: object) -> None:
        seen.append(os.environ.get("CHECKMAITE_SECRETS_DIR"))

    monkeypatch.setattr("checkmaite_container._runner.run_plan", fake_run_plan)
    # The secrets directory is optional and need not exist.
    secrets = tmp_path / "missing-secrets"

    assert cli.main(["run", "--cache", str(tmp_path / "cache"), "--secrets", str(secrets)]) == 0
    assert seen == [str(secrets)]


def test_environment_defaults_and_cli_precedence() -> None:
    environment = {
        "CHECKMAITE_CONFIG": "/environment/run.yaml",
        "CHECKMAITE_OUTPUT_DIR": "/environment/output",
        "CHECKMAITE_CACHE_DIR": "/environment/cache",
        "CHECKMAITE_SECRETS_DIR": "/environment/secrets",
        "CHECKMAITE_THREADS": "3",
        "CHECKMAITE_DEVICE": "cuda:1",
        "CHECKMAITE_BATCH_SIZE": "8",
        "CHECKMAITE_LOG_LEVEL": "warning",
    }
    parser = cli.build_parser(environment)

    defaults = parser.parse_args(["run"])
    assert defaults.config == "/environment/run.yaml"
    assert defaults.output == "/environment/output"
    assert defaults.cache == "/environment/cache"
    assert defaults.secrets == "/environment/secrets"
    assert defaults.threads == 3
    assert defaults.device == "cuda:1"
    assert defaults.batch_size == 8
    assert defaults.log_level == "WARNING"

    overrides = parser.parse_args(
        [
            "run",
            "--config",
            "/cli/run.yaml",
            "--output",
            "/cli/output",
            "--cache",
            "/cli/cache",
            "--secrets",
            "/cli/secrets",
            "--threads",
            "2",
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--log-level",
            "DEBUG",
        ]
    )
    assert overrides.config == "/cli/run.yaml"
    assert overrides.output == "/cli/output"
    assert overrides.cache == "/cli/cache"
    assert overrides.secrets == "/cli/secrets"
    assert overrides.threads == 2
    assert overrides.device == "cpu"
    assert overrides.batch_size == 4
    assert overrides.log_level == "DEBUG"


def test_parser_rejects_invalid_environment_value() -> None:
    with pytest.raises(SystemExit) as error:
        cli.build_parser({"CHECKMAITE_LOG_LEVEL": "verbose"}).parse_args(["run"])

    assert error.value.code == 2


def test_cache_environment_follows_selected_directory(tmp_path: Path) -> None:
    names = ("HOME", "TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "HF_HOME", "TORCH_HOME")
    previous = {name: os.environ.pop(name, None) for name in names}
    try:
        os.environ["HOME"] = str(tmp_path / "existing-home")
        cli._configure_cache_environment(tmp_path)
        assert os.environ["HOME"] == str(tmp_path)
        assert os.environ["TMPDIR"] == str(tmp_path / "tmp")
        assert os.environ["XDG_CACHE_HOME"] == str(tmp_path / ".cache")
        assert os.environ["MPLCONFIGDIR"] == str(tmp_path / "matplotlib")
        assert os.environ["HF_HOME"] == str(tmp_path / "huggingface")
        assert os.environ["TORCH_HOME"] == str(tmp_path / "torch")
        assert all(Path(os.environ[name]).is_dir() for name in names)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_cache_environment_respects_user_library_locations(tmp_path: Path) -> None:
    names = ("HOME", "TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "HF_HOME", "TORCH_HOME")
    library_names = names[1:]
    previous = {name: os.environ.get(name) for name in names}
    configured = {name: str(tmp_path / name.lower()) for name in library_names}
    try:
        os.environ["HOME"] = str(tmp_path / "existing-home")
        os.environ.update(configured)
        defaults = tmp_path / "defaults"
        cli._configure_cache_environment(defaults)
        assert os.environ["HOME"] == str(defaults)
        assert {name: os.environ[name] for name in library_names} == configured
        assert defaults.is_dir()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_main_runs_plan(tmp_path: Path, plugin_file: Path) -> None:
    assert plugin_file.is_file()
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    managed_names = [*cli._cache_environment(cache), "CHECKMAITE_SECRETS_DIR"]
    environment_before = {name: os.environ.get(name) for name in managed_names}

    status = cli.main(
        [
            "run",
            "--config",
            str(_write_plan(tmp_path)),
            "--output",
            str(output),
            "--cache",
            str(cache),
            "--threads",
            "1",
            "--device",
            "cpu",
            "--batch-size",
            "4",
        ]
    )

    assert status == 0
    run_payload = json.loads((output / "tasks/evaluation/run.json").read_text(encoding="utf-8"))
    assert run_payload["config"]["batch_size"] == 4
    assert run_payload["outputs"]["device"] == "cpu"
    assert (output / "run-results.json").is_file()
    assert cache.is_dir()
    assert {name: os.environ.get(name) for name in managed_names} == environment_before


def test_main_limits_already_loaded_thread_pools(tmp_path: Path, plugin_file: Path) -> None:
    assert plugin_file.is_file()
    # NumPy's BLAS pool is loaded before the runner sets OPENBLAS_NUM_THREADS.
    threadpool_limits(limits=2)

    status = cli.main(
        [
            "run",
            "--config",
            str(_write_plan(tmp_path)),
            "--output",
            str(tmp_path / "output"),
            "--cache",
            str(tmp_path / "cache"),
            "--threads",
            "1",
        ],
    )

    assert status == 0
    assert torch.get_num_threads() == 1
    assert all(pool["num_threads"] == 1 for pool in threadpool_info())


def test_main_returns_plan_error_status(tmp_path: Path) -> None:
    invalid_plan = tmp_path / "invalid.yaml"
    invalid_plan.write_text("version: [", encoding="utf-8")

    assert cli.main(["run", "--config", str(invalid_plan), "--cache", str(tmp_path / "cache")]) == 2


def test_main_stops_at_first_execution_failure(tmp_path: Path, plugin_file: Path) -> None:
    assert plugin_file.is_file()
    plan_path = _write_plan(tmp_path)
    payload = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    completed_task = payload["tasks"][0]
    payload["tasks"] += [
        {**completed_task, "name": "failure", "capability": "file:plugin.py:FailingCapability"},
        {**completed_task, "name": "skipped"},
    ]
    plan_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    output = tmp_path / "output"

    status = cli.main(
        ["run", "--config", str(plan_path), "--output", str(output), "--cache", str(tmp_path / "cache")],
    )

    assert status == 1
    # Completed task outputs remain; the run summary is written only after every task succeeds.
    assert (output / "tasks/evaluation/run.json").is_file()
    assert not (output / "tasks/failure").exists()
    assert not (output / "tasks/skipped").exists()
    assert not (output / "run-results.json").exists()


def test_main_removes_stale_summary_when_rerun_fails(tmp_path: Path, plugin_file: Path) -> None:
    assert plugin_file.is_file()
    output = tmp_path / "output"
    arguments = ["run", "--output", str(output), "--cache", str(tmp_path / "cache"), "--config"]

    assert cli.main([*arguments, str(_write_plan(tmp_path))]) == 0
    assert (output / "run-results.json").is_file()

    assert cli.main([*arguments, str(_write_plan(tmp_path, capability="FailingCapability"))]) == 1
    assert not (output / "run-results.json").exists()
    # Earlier task outputs are left in place.
    assert (output / "tasks/evaluation/run.json").is_file()


@pytest.mark.parametrize(
    ("change", "status"),
    [
        # Bad input data is an execution failure, not a configuration error.
        ({"datasets": {"data": {"class": "file:plugin.py:CorruptDataset"}}}, 1),
        ({"datasets": {"data": {"class": "file:plugin.py:FakeDataset", "args": {"unknown": 1}}}}, 2),
        ({"datasets": {"data": {"class": "file:plugin.py:Missing"}}}, 2),
        ({"capability_args": {"unknown": 1}}, 2),
        ({"config": {"batch_size": "many"}}, 2),
    ],
)
def test_main_separates_configuration_and_execution_errors(
    tmp_path: Path,
    plugin_file: Path,
    change: dict[str, Any],
    status: int,
) -> None:
    assert plugin_file.is_file()
    plan_path = _write_plan(tmp_path)
    payload = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if "datasets" in change:
        payload["datasets"] = change["datasets"]
    else:
        payload["tasks"][0].update(change)
    plan_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    assert (
        cli.main(
            [
                "run",
                "--config",
                str(plan_path),
                "--cache",
                str(tmp_path / "cache"),
                "--output",
                str(tmp_path / "output"),
            ]
        )
        == status
    )


def test_main_does_not_misclassify_execution_value_error(
    tmp_path: Path,
    plugin_file: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert plugin_file.is_file()

    status = cli.main(
        [
            "run",
            "--config",
            str(_write_plan(tmp_path, capability="ValueFailingCapability")),
            "--output",
            str(tmp_path / "output"),
            "--cache",
            str(tmp_path / "cache"),
        ]
    )

    assert status == 1
    failure = next(record for record in caplog.records if record.message == "CheckMAITE run failed")
    assert failure.exc_info is not None
    assert failure.exc_info[0] is ValueError


@pytest.mark.parametrize("arguments", [["--help"], ["run", "--help"]])
def test_main_prints_operational_help(
    arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)

    assert error.value.code == 0
    output = capsys.readouterr().out
    for expected in (
        "Run one finite CheckMAITE plan",
        "CHECKMAITE_CONFIG",
        "CHECKMAITE_OUTPUT_DIR",
        "CHECKMAITE_CACHE_DIR",
        "CHECKMAITE_SECRETS_DIR",
        "CHECKMAITE_THREADS",
        "CHECKMAITE_DEVICE",
        "CHECKMAITE_BATCH_SIZE",
        "CHECKMAITE_LOG_LEVEL",
        "auto|cpu|cuda|cuda:N",
        "/checkmaite",
        "/output",
        "/cache",
        "/run/secrets",
        "The built-in runtime requires no secrets",
    ):
        assert expected in output


def test_main_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--version"])

    assert error.value.code == 0
    assert capsys.readouterr().out.strip() == f"checkmaite-container {__version__}"
