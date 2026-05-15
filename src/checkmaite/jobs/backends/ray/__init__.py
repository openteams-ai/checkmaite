"""Ray task-based jobs backend."""

from checkmaite.jobs.backends.ray.job_backend import RayJob, RayJobBackend

__all__ = ["RayJobBackend", "RayJob"]
