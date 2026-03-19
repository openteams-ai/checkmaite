"""Ray Serve deployment classes that wrap existing checkmaite models for remote inference."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from ray import serve


# Performance tuning note:
# - ``executor_workers`` (configured via ``.bind(...)``) controls how many blocking
#   model calls can run concurrently inside a replica.
# - ``@serve.deployment`` / ``.options(...)`` queue settings (notably
#   ``max_ongoing_requests`` and ``max_queued_requests``) control request admission
#   and queueing at the replica boundary.
# Tune these together when balancing latency vs throughput.
@serve.deployment
class RayObjectDetectionDeployment:
    """Ray Serve deployment wrapping a checkmaite object detection model.
    Parameters
    ----------
    model_cls
        Model class to instantiate.
    model_kwargs
        Keyword args passed to ``model_cls``.
    executor_workers
        One-time thread-pool worker count for running blocking model calls.
        Configure this at deployment construction time via ``.bind(...)``.
    """

    def __init__(
        self,
        model_cls: type,
        model_kwargs: dict[str, Any],
        executor_workers: int = 1,
    ) -> None:
        self.model = model_cls(**model_kwargs)
        self._executor = ThreadPoolExecutor(max_workers=executor_workers)

    async def predict(self, input_batch: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        """Run inference and return results as numpy dicts."""
        # Run blocking model inference in a thread to avoid blocking the
        # Ray Serve asyncio event loop, which would prevent concurrent
        # requests and health checks during inference.
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(self._executor, self.model, input_batch)

        return [
            {
                "boxes": np.asarray(pred.boxes),
                "labels": np.asarray(pred.labels),
                "scores": np.asarray(pred.scores),
            }
            for pred in results
        ]

    async def get_metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        return {**self.model.metadata, "model_cls": type(self.model).__name__}


@serve.deployment
class RayImageClassificationDeployment:
    """Ray Serve deployment wrapping a checkmaite image classification model.

    Parameters
    ----------
    model_cls
        Model class to instantiate.
    model_kwargs
        Keyword args passed to ``model_cls``.
    executor_workers
        One-time thread-pool worker count for running blocking model calls.
        Configure this at deployment construction time via ``.bind(...)``.
    """

    def __init__(
        self,
        model_cls: type,
        model_kwargs: dict[str, Any],
        executor_workers: int = 1,
    ) -> None:
        self.model = model_cls(**model_kwargs)
        self._executor = ThreadPoolExecutor(max_workers=executor_workers)

    async def predict(self, input_batch: list[np.ndarray]) -> list[np.ndarray]:
        """Run inference and return results as numpy arrays."""
        # Run blocking model inference in a thread to avoid blocking the
        # Ray Serve asyncio event loop, which would prevent concurrent
        # requests and health checks during inference.
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(self._executor, self.model, input_batch)

        return [np.asarray(pred) for pred in results]

    async def get_metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        return {**self.model.metadata, "model_cls": type(self.model).__name__}
