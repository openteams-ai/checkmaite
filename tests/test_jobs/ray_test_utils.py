from __future__ import annotations

import os
from typing import Any

import ray

_DEFAULT_TEST_RAY_CPUS = 2
_DEFAULT_TEST_OBJECT_STORE_BYTES = 256 * 1024 * 1024


def init_local_ray(**overrides: Any):
    """Start a small local Ray runtime for tests.

    CI runners can expose many CPUs. Ray may then keep many worker processes
    around, which is expensive for the lightweight job-backend tests. Keep test
    runtimes small unless a test explicitly overrides the defaults.
    """
    os.environ.setdefault("RAY_memory_usage_threshold", "0.99")

    kwargs: dict[str, Any] = {
        "address": "local",
        "include_dashboard": False,
        "log_to_driver": True,
        "num_cpus": int(os.environ.get("CHECKMAITE_TEST_RAY_NUM_CPUS", _DEFAULT_TEST_RAY_CPUS)),
        "object_store_memory": int(
            os.environ.get("CHECKMAITE_TEST_RAY_OBJECT_STORE_MEMORY", _DEFAULT_TEST_OBJECT_STORE_BYTES)
        ),
    }
    kwargs.update(overrides)
    return ray.init(**kwargs)
