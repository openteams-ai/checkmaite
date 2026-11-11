"""Cached MAITE tasks for object detection.

``evaluate`` and ``predict`` take the same arguments as
:mod:`checkmaite.core.cached_tasks` without ``task``, which is always
``'object_detection'``, so they can replace ``maite.tasks.evaluate`` and
``maite.tasks.predict`` for object detection models.
"""

from checkmaite.core.cached_tasks import evaluate_from_predictions, for_task
from checkmaite.core.object_detection._task import TASK

__all__ = ["evaluate", "evaluate_from_predictions", "predict"]

_cached = for_task(TASK)
evaluate = _cached.evaluate
predict = _cached.predict
