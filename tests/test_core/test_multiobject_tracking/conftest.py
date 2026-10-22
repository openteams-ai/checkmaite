"""Fakes and fixtures for multi-object-tracking tests.

The fake model's canned predictions are constructed against the fake dataset's
ground truth so every CLEAR-MOT/identity figure has a known, hand-computed
value. Two videos:

**video_0** (3 frames, ground-truth tracks 1 and 2 present in every frame):

- track 1 is matched exactly (IoU 1.0) in all three frames by prediction id 101
- track 2 is matched in frame 0 by prediction id 102 and in frame 1 by
  prediction id 103 — an identity switch — and missed entirely in frame 2
- frame 2 also contains a false-positive prediction (id 104) overlapping
  no ground truth
- frame 0 additionally contains an unconfirmed detection (``track_id == -1``)
  that must be excluded from evaluation rather than counted as a false
  positive

**video_1** (2 frames, ground-truth track 7): matched exactly in both frames
by prediction id 201.

Totals: 8 ground-truth boxes, 1 miss, 1 false positive, 1 identity switch,
and every match is exact, giving

- MOTA = 1 - (1 + 1 + 1) / 8 = 0.625
- MOTP = 0.0 (mean IoU distance over matches; motmetrics convention)
- IDF1 = 2*6 / (2*6 + 2 + 2) = 0.75 (IDTP=6: 3 for track 1, 1 for track 2's
  best identity pairing, 2 for video_1; IDFN=2; IDFP=2)
- num_switches = 1
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from maite.protocols import DatasetMetadata, ModelMetadata

N_VIDEOS = 2
FRAME_SHAPE = (3, 8, 12)  # (C, H, W)

_INDEX2LABEL = {0: "vehicle"}

DEFAULT_MOT_DATASET_METADATA = DatasetMetadata(id="fake_mot_dataset", index2label=_INDEX2LABEL)
DEFAULT_MOT_MODEL_METADATA = ModelMetadata(id="fake_mot_model", index2label=_INDEX2LABEL)

EXPECTED_MOT_MOTA = 0.625
EXPECTED_MOT_MOTP = 0.0
EXPECTED_MOT_IDF1 = 0.75
EXPECTED_MOT_NUM_SWITCHES = 1


@dataclass
class FakeVideoFrame:
    """Satisfies maite.protocols.multiobject_tracking.VideoFrame."""

    frame_index: int
    pixels: Any = field(default_factory=lambda: np.zeros(FRAME_SHAPE, dtype=np.uint8))
    time_s: float = 0.0
    pts: int = 0

    def __post_init__(self) -> None:
        self.time_s = self.frame_index / 10.0
        self.pts = self.frame_index


class FakeClip:
    """A re-iterable VideoStream tagged with its video index.

    Deliberately not a list: datamaite's MOT input reopens its source file on
    every iteration, so a real video stream is an ``Iterable[VideoFrame]`` and
    not a sequence. Anything that treats streams differently from arrays has to
    be exercised against that shape.
    """

    def __init__(self, video_index: int, n_frames: int) -> None:
        self.video_index = video_index
        self._frames = tuple(FakeVideoFrame(frame_index=i) for i in range(n_frames))

    def __iter__(self) -> Any:
        return iter(self._frames)


class OneShotClip:
    """A one-shot VideoStream: iterating it consumes it, like a raw decoder generator.

    MAITE types MOT inputs as ``Iterable[VideoFrame]``, which does not promise
    re-iterability. Anything that inspects an input must therefore decide from
    its type, never by pulling a frame off it.
    """

    def __init__(self, video_index: int, n_frames: int) -> None:
        self.video_index = video_index
        self._frames = iter([FakeVideoFrame(frame_index=i) for i in range(n_frames)])

    def __iter__(self) -> Any:
        return self._frames


@dataclass
class FakeFrameTracks:
    """Satisfies maite.protocols.multiobject_tracking.SingleFrameObjectTrackingTarget."""

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    track_ids: np.ndarray


@dataclass
class FakeMOTTarget:
    """Satisfies maite.protocols.multiobject_tracking.MultiobjectTrackingTarget."""

    frame_tracks: list[FakeFrameTracks]


def frame_tracks(boxes: list[list[float]], track_ids: list[int]) -> FakeFrameTracks:
    n = len(track_ids)
    return FakeFrameTracks(
        boxes=np.asarray(boxes, dtype=np.float32).reshape(n, 4),
        labels=np.zeros(n, dtype=np.int64),
        scores=np.ones(n, dtype=np.float32),
        track_ids=np.asarray(track_ids, dtype=np.int64),
    )


# Ground-truth geometry reused by predictions so matches are exact (IoU 1.0).
_GT1 = [0.0, 0.0, 10.0, 10.0]
_GT2 = [20.0, 20.0, 30.0, 30.0]
_GT7 = [5.0, 5.0, 15.0, 15.0]
_FP_BOX = [50.0, 50.0, 60.0, 60.0]
_UNCONFIRMED_BOX = [70.0, 70.0, 80.0, 80.0]

DEFAULT_MOT_DATASET_TARGETS: list[FakeMOTTarget] = [
    FakeMOTTarget(
        frame_tracks=[
            frame_tracks([_GT1, _GT2], [1, 2]),
            frame_tracks([_GT1, _GT2], [1, 2]),
            frame_tracks([_GT1, _GT2], [1, 2]),
        ]
    ),
    FakeMOTTarget(
        frame_tracks=[
            frame_tracks([_GT7], [7]),
            frame_tracks([_GT7], [7]),
        ]
    ),
]

DEFAULT_MOT_MODEL_PREDICTIONS: list[FakeMOTTarget] = [
    FakeMOTTarget(
        frame_tracks=[
            # Frame 0: both tracks matched; extra unconfirmed detection (-1).
            frame_tracks([_GT1, _GT2, _UNCONFIRMED_BOX], [101, 102, -1]),
            # Frame 1: track 2's match changes identity (102 -> 103): 1 switch.
            frame_tracks([_GT1, _GT2], [101, 103]),
            # Frame 2: track 2 missed; one false positive.
            frame_tracks([_GT1, _FP_BOX], [101, 104]),
        ]
    ),
    FakeMOTTarget(
        frame_tracks=[
            frame_tracks([_GT7], [201]),
            frame_tracks([_GT7], [201]),
        ]
    ),
]

N_FRAMES = [len(t.frame_tracks) for t in DEFAULT_MOT_DATASET_TARGETS]

DEFAULT_MOT_DATUM_METADATAS: list[dict[str, Any]] = [{"id": f"video_{i}"} for i in range(N_VIDEOS)]


class FakeMOTDataset:
    """Protocol-compatible MOT dataset over the canned videos."""

    def __init__(
        self,
        targets: list[FakeMOTTarget] | None = None,
        metadata: DatasetMetadata | None = None,
        *,
        clip_factory: Any = FakeClip,
        clip_frames: int | None = None,
    ) -> None:
        self._targets = targets if targets is not None else DEFAULT_MOT_DATASET_TARGETS
        self.metadata = metadata if metadata is not None else DEFAULT_MOT_DATASET_METADATA
        # Video streams vary in kind (re-iterable or one-shot) and length
        # (including empty), independently of the canned targets.
        self._clip_factory = clip_factory
        self._clip_frames = clip_frames

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> tuple[Any, FakeMOTTarget, dict[str, Any]]:
        n_frames = self._clip_frames if self._clip_frames is not None else len(self._targets[index].frame_tracks)
        return (
            self._clip_factory(index, n_frames),
            self._targets[index],
            DEFAULT_MOT_DATUM_METADATAS[index],
        )


class FakeMOTModel:
    """Protocol-compatible MOT model returning canned per-video predictions."""

    def __init__(self, predictions: list[FakeMOTTarget] | None = None, metadata: ModelMetadata | None = None) -> None:
        self._predictions = predictions if predictions is not None else DEFAULT_MOT_MODEL_PREDICTIONS
        self.metadata = metadata if metadata is not None else DEFAULT_MOT_MODEL_METADATA
        self.n_calls = 0

    def __call__(self, batch: list[FakeClip]) -> list[FakeMOTTarget]:
        self.n_calls += 1
        return [self._predictions[clip.video_index] for clip in batch]


@pytest.fixture
def fake_mot_dataset_default() -> FakeMOTDataset:
    return FakeMOTDataset()


@pytest.fixture
def fake_mot_model_default() -> FakeMOTModel:
    return FakeMOTModel()
