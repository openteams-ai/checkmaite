"""Generate the versioned JSON Schema for container run-plan YAML."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import cast

from pydantic import BaseModel

_RUNTIME_ROOT = Path(__file__).resolve().parent
_SCHEMA_PATH = _RUNTIME_ROOT / "schema" / "run-plan-v1.schema.json"
_PLAN_PATH = _RUNTIME_ROOT / "src" / "checkmaite_container" / "_plan.py"


def render_schema() -> str:
    """Return the canonical serialized run-plan schema."""
    plan_model = cast(type[BaseModel], runpy.run_path(str(_PLAN_PATH))["RunPlan"])
    schema = plan_model.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "CheckMAITE container run plan v1"
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Write the schema, or verify that the committed schema is current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the committed schema is out of date",
    )
    args = parser.parse_args()
    rendered = render_schema()

    if args.check:
        if not _SCHEMA_PATH.is_file() or _SCHEMA_PATH.read_text(encoding="utf-8") != rendered:
            print(
                f"{_SCHEMA_PATH} is out of date; run {Path(__file__).relative_to(_RUNTIME_ROOT.parents[1])}",
                file=sys.stderr,
            )
            return 1
        return 0

    _SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SCHEMA_PATH.write_text(rendered, encoding="utf-8")
    print(_SCHEMA_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
