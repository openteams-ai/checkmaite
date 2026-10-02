import json
import subprocess
import sys
from pathlib import Path

from checkmaite_container import load_plan

_REPOSITORY_ROOT = Path(__file__).parents[2]
_SCHEMA_PATH = _REPOSITORY_ROOT / "docker/runtime/schema/run-plan-v1.schema.json"
_GENERATOR_PATH = _REPOSITORY_ROOT / "docker/runtime/generate_schema.py"


def test_committed_run_plan_schema_is_current() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(_GENERATOR_PATH), "--check"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_schema_defines_yaml_field_names() -> None:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert set(schema["properties"]) == {"version", "resources", "datasets", "models", "metrics", "tasks"}
    assert set(schema["$defs"]["ResourceSpec"]["properties"]) == {"threads", "device"}
    assert set(schema["$defs"]["ObjectSpec"]["properties"]) == {"class", "args"}
    assert set(schema["$defs"]["TaskSpec"]["properties"]) == {
        "name",
        "capability",
        "capability_args",
        "dataset",
        "model",
        "metrics",
        "config",
        "use_cache",
        "report_threshold",
    }


def test_example_run_plan_is_valid() -> None:
    plan = load_plan(_REPOSITORY_ROOT / "docker/example-run.yaml")

    assert plan.version == 1
    assert [task.name for task in plan.tasks] == ["baseline"]
