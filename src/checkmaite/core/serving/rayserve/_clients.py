"""MAITE-compliant client classes that perform inference via Ray Serve deployments."""

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from maite.protocols import image_classification as ic
from maite.protocols import object_detection as od
from ray.serve.handle import DeploymentHandle

from checkmaite.core.object_detection.dataset_loaders import DetectionTarget

_executor = ThreadPoolExecutor(max_workers=1)


def _get_result(response: Any) -> Any:
    """Get result from a DeploymentResponse, handling both sync and async contexts.

    Ray Serve's ``DeploymentResponse.result()`` raises a ``RuntimeError`` when
    called inside a running asyncio event loop (e.g., Jupyter notebooks):

        RuntimeError: Sync methods should not be called from within an
        ``asyncio`` event loop. Use ``await response`` instead.

    Since MAITE model protocols require a synchronous ``__call__``, we cannot
    simply ``await`` the response. Instead, when an event loop is detected, we
    offload the blocking ``.result()`` call to a background thread which is
    allowed to block without interfering with the event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop — safe to call .result() directly
        return response.result()

    # Inside an event loop (e.g., Jupyter) — run .result() in a thread
    return _executor.submit(response.result).result()


class RayObjectDetectionClient:
    """MAITE-compliant object detection client that delegates inference to a Ray Serve deployment."""

    def __init__(self, handle: DeploymentHandle) -> None:
        self._handle = handle
        self.metadata: dict[str, Any] = _get_result(handle.get_metadata.remote())

    def __call__(self, input_batch: Sequence[od.InputType]) -> Sequence[od.TargetType]:
        """Run object detection on a batch of images via the remote deployment.

        Parameters
        ----------
        input_batch : Sequence[od.InputType]
            A batch of images, each with shape (C, H, W).

        Returns
        -------
        Sequence[od.TargetType]
            Detection results with boxes, labels, and scores for each image.

        """
        np_batch = [np.asarray(img) for img in input_batch]
        result = _get_result(self._handle.predict.remote(np_batch))
        return [
            DetectionTarget(
                boxes=pred["boxes"],
                labels=pred["labels"],
                scores=pred["scores"],
            )
            for pred in result
        ]

    @property
    def name(self) -> str:
        """Human-readable name for the remote object detection model."""
        return f"remote-od-{self.metadata['id']}"


class RayImageClassificationClient:
    """MAITE-compliant image classification client that delegates inference to a Ray Serve deployment."""

    def __init__(self, handle: DeploymentHandle) -> None:
        self._handle = handle
        self.metadata: dict[str, Any] = _get_result(handle.get_metadata.remote())

    def __call__(self, input_batch: Sequence[ic.InputType]) -> Sequence[ic.TargetType]:
        """Run image classification on a batch of images via the remote deployment.

        Parameters
        ----------
        input_batch : Sequence[ic.InputType]
            A batch of images, each with shape (C, H, W).

        Returns
        -------
        Sequence[ic.TargetType]
            Classification probabilities for each image.

        """
        np_batch = [np.asarray(img) for img in input_batch]
        return _get_result(self._handle.predict.remote(np_batch))

    @property
    def name(self) -> str:
        """Human-readable name for the remote image classification model."""
        return f"remote-ic-{self.metadata['id']}"
