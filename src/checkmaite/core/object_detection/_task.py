from maite.protocols import ArrayLike
from maite.protocols.object_detection import ObjectDetectionTarget
from pydantic import TypeAdapter

from checkmaite.core.capability_core import CapabilityOutputsBase
from checkmaite.core.evaluation_tasks import EvaluationTask


# Pydantic needs a model to de/serialize object-detection targets, which MAITE defines as a protocol.
class PydanticCompatObjectDetectionTarget(CapabilityOutputsBase):
    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike


TASK: EvaluationTask[ObjectDetectionTarget] = EvaluationTask(
    name="object_detection",
    namespace="checkmaite.core.object_detection",
    target_adapter=TypeAdapter(PydanticCompatObjectDetectionTarget),
    aliases=("od",),
)
