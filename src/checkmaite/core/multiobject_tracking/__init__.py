from checkmaite.core._common.maite_evaluation_capability import (
    MaiteEvaluationConfig,
    MaiteEvaluationOutputs,
    MaiteMetricResult,
)
from checkmaite.core._common.metric_fanout import MaiteEvaluationMetricError
from checkmaite.core._plugins import inject_plugin_exports
from checkmaite.core.evaluation_tasks import register_task
from checkmaite.core.multiobject_tracking._task import TASK
from checkmaite.core.multiobject_tracking.dataset_loaders import load_mot_dataset
from checkmaite.core.multiobject_tracking.maite_evaluation_capability import MaiteEvaluation

__all__ = [
    "MaiteEvaluation",
    "MaiteEvaluationConfig",
    "MaiteEvaluationOutputs",
    "MaiteEvaluationMetricError",
    "MaiteMetricResult",
    "load_mot_dataset",
]

register_task(TASK)
inject_plugin_exports(globals(), __all__, group="checkmaite.plugins.multiobject_tracking")
