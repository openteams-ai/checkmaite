"""Cached MAITE tasks for image classification.

``evaluate`` and ``predict`` take the same arguments as
:mod:`checkmaite.core.cached_tasks` without ``task``, which is always
``'image_classification'``, so they can replace ``maite.tasks.evaluate`` and
``maite.tasks.predict`` for image classification models.
"""

from checkmaite.core.cached_tasks import evaluate_from_predictions, for_task
from checkmaite.core.image_classification._task import TASK

__all__ = ["evaluate", "evaluate_from_predictions", "predict"]

_cached = for_task(TASK)
evaluate = _cached.evaluate
predict = _cached.predict
