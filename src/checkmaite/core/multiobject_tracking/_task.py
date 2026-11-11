from collections.abc import Sequence
from typing import Any

from maite.protocols import ArrayLike
from maite.protocols.multiobject_tracking import MultiobjectTrackingTarget
from pydantic import TypeAdapter, field_validator

from checkmaite.core.capability_core import CapabilityOutputsBase
from checkmaite.core.evaluation_tasks import EvaluationTask


# Pydantic needs models to de/serialize tracking targets, which MAITE defines as protocols.
class PydanticCompatSingleFrameObjectTrackingTarget(CapabilityOutputsBase):
    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike
    track_ids: ArrayLike


class PydanticCompatMultiobjectTrackingTarget(CapabilityOutputsBase):
    frame_tracks: Sequence[PydanticCompatSingleFrameObjectTrackingTarget]

    @field_validator("frame_tracks", mode="before")
    @classmethod
    def _materialize_frame_tracks(cls, value: Any) -> Any:
        # Pydantic rebuilds a Sequence that is not a list or tuple by calling its
        # type with the validated items. Lazy sequences such as DataMAITE's
        # _FrameTracks have a different constructor, so that fails and the cache
        # is skipped. Copy them into a list of frame targets instead.
        if isinstance(value, Sequence) and not isinstance(value, list | tuple | str | bytes):
            return list(value)
        return value


TASK: EvaluationTask[MultiobjectTrackingTarget] = EvaluationTask(
    name="multiobject_tracking",
    namespace="checkmaite.core.multiobject_tracking",
    target_adapter=TypeAdapter(PydanticCompatMultiobjectTrackingTarget),
    aliases=("mot",),
    # Each datum is a video whose frames may arrive as a one-shot stream.
    materialize_inputs=True,
)
