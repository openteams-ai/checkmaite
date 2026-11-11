from maite.protocols.image_classification import TargetType
from pydantic import ConfigDict, TypeAdapter

from checkmaite.core.evaluation_tasks import EvaluationTask

TASK: EvaluationTask[TargetType] = EvaluationTask(
    name="image_classification",
    namespace="checkmaite.core.image_classification",
    # TargetType is MAITE's ArrayLike, a union that includes NumPy's buffer
    # protocols, which pydantic only validates as arbitrary types.
    target_adapter=TypeAdapter(TargetType, config=ConfigDict(arbitrary_types_allowed=True)),
    aliases=("ic",),
)
