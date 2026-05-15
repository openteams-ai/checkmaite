from __future__ import annotations

import pytest
import ray

from checkmaite.jobs import shutdown_job_backend


@pytest.fixture
def _jobs_smoke_ray_runtime():
    """Function-scoped local Ray runtime for unmarked jobs smoke tests.

    These tests intentionally exercise live Ray paths in the default test run for
    coverage. Each test owns Ray initialization/shutdown so marked integration
    tests cannot invalidate a long-lived shared runtime by calling ray.shutdown().
    """
    shutdown_job_backend(wait=False)
    ray.shutdown()
    ray.init(address="local")

    try:
        yield
    finally:
        shutdown_job_backend(wait=False)
        ray.shutdown()
