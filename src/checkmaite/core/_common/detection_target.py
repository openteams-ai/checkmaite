"""The object-detection target CheckMAITE builds internally."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionTarget:
    """Detections for one image, conforming to ``maite.protocols.object_detection.ObjectDetectionTarget``.

    MAITE types the fields as ``ArrayLike``; CheckMAITE converts at the
    boundary so code that builds or reads these targets works on arrays.

    Parameters
    ----------
    boxes
        ``(N, 4)`` boxes in ``xyxy`` format.
    labels
        ``(N,)`` integer label ids.
    scores
        ``(N,)`` scores, or ``(N, n_classes)`` per-class scores.
    """

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
