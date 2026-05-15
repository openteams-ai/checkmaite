"""Simple process-local Ray job backend."""

from .job_backend import RaySimpleJob, RaySimpleJobBackend

__all__ = ["RaySimpleJob", "RaySimpleJobBackend"]
