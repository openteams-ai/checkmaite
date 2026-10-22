"""Load native datamaite multi-object-tracking datasets for checkmaite.

The factory here returns :class:`datamaite.BoxTrackDataset` objects directly.
Datamaite owns format parsing, frame decoding and MAITE target construction.

What stays CheckMAITE-side is evaluation policy:

- Every frame of every video is evaluated. Scoring only the annotated frames
  would never count false positives on the others, and would show the tracker
  gaps in time. A sequence whose exact frame count is unknown is refused rather
  than silently reduced to its annotated frames.
- Only formats where that is correct are accepted. Treating a frame without
  boxes as empty is only right when every frame is labeled.
- Datamaite gives every dataset the same default ``metadata["id"]``, and
  CheckMAITE's prediction, evaluation and run caches are keyed on that id, so
  it is derived here from everything that changes the loaded targets.
"""

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any

from datamaite import BoxTrackDataset, load_mot
from upath import UPath

from checkmaite.core._common.dataset_utils import datamaite_root
from checkmaite.core._utils import id_hash

StorageOptions = Mapping[str, Any] | None

SUPPORTED_DATASET_FORMATS = frozenset({"motchallenge", "flat_mp4"})

_UNSUPPORTED_REASONS = {
    "tao": "TAO labels only some frames (about one per second)",
    "hmie": "HMIE usually labels only some frames (its annotation frame rate is below the video frame rate)",
    "visdrone_video": "datamaite currently drops VisDrone ground-truth tracks whose id is 0",
}


def load_mot_dataset(
    root: str | UPath,
    dataset_format: str,
    dataset_id: str | None = None,
    *,
    storage_options: StorageOptions = None,
    **options: Any,
) -> BoxTrackDataset:
    """Load a multi-object-tracking dataset for evaluation over every frame.

    ``dataset_format`` is ``"motchallenge"``, or ``"flat_mp4"`` for unannotated
    videos to run a tracker over. ``**options`` are passed to
    :func:`datamaite.load_mot` (for example ``classes={1}``). Every frame of every
    video is emitted; a sequence whose exact frame count is unknown raises
    ``ValueError``.

    Unless ``dataset_id`` is given, the id is derived from the root, format and
    loader options, so datasets that load different targets never share cached
    predictions or evaluations. ``storage_options`` are excluded from the id
    because they configure access, not content.
    """
    format_name = str(getattr(dataset_format, "value", dataset_format)).lower()
    if format_name not in SUPPORTED_DATASET_FORMATS:
        reason = _UNSUPPORTED_REASONS.get(format_name, "it is not a supported multi-object-tracking format")
        supported = ", ".join(sorted(SUPPORTED_DATASET_FORMATS))
        raise ValueError(
            f"Cannot evaluate {format_name!r} datasets over every frame: {reason}. Supported formats: {supported}."
        )

    dataset = load_mot(
        datamaite_root(root),
        dataset_format=format_name,
        storage_options=storage_options,
        **options,
    )
    inexact = [str(seq.video_id) for seq in dataset.sequences if not (seq.num_frames and seq.num_frames_exact)]
    if inexact:
        raise ValueError(
            "Cannot evaluate every frame: the exact frame count is unknown for sequence(s) "
            f"{', '.join(inexact)}, so datamaite would emit only their annotated frames."
        )

    if dataset_id is None:
        # Sort set-valued options such as ``classes`` so the id does not depend on set order.
        hashed_options = {
            key: sorted(value) if isinstance(value, AbstractSet) else value for key, value in options.items()
        }
        dataset_id = f"mot_{format_name}_" + id_hash(root=str(root), dataset_format=format_name, **hashed_options)
    return dataset.with_mot_options(empty_frame_policy="all", dataset_id=dataset_id)
