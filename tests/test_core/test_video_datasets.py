"""IR-3.3-S-1 (#794): checkmaite accepts a flat folder of H.264 / MPEG-2 ``.mp4`` videos.

Video loading is datamaite's job; these tests pin the consume path checkmaite
documents: on a default install, load the folder with ``datamaite.load_mot`` and get
a MAITE multi-object-tracking dataset whose frames decode.
"""

from dataclasses import dataclass
from pathlib import Path

import av
import maite.protocols.multiobject_tracking as mot
import numpy as np
import pytest
from datamaite import load_mot

from checkmaite import cached_tasks

_FRAMES = 4
_HEIGHT, _WIDTH = 24, 32


def _write_mp4(path: Path, codec: str) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=5)
        stream.width, stream.height, stream.pix_fmt = _WIDTH, _HEIGHT, "yuv420p"
        for i in range(_FRAMES):
            pixels = np.full((_HEIGHT, _WIDTH, 3), i * 40, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def flat_mp4_folder(tmp_path: Path) -> Path:
    _write_mp4(tmp_path / "a_h264.mp4", "libx264")
    _write_mp4(tmp_path / "b_mpeg2.mp4", "mpeg2video")
    return tmp_path


def test_flat_mp4_folder_loads_as_maite_mot_dataset(flat_mp4_folder: Path) -> None:
    dataset = load_mot(flat_mp4_folder, dataset_format="flat_mp4").with_mot_options(empty_frame_policy="all")

    assert isinstance(dataset, mot.Dataset)
    assert len(dataset) == 2  # one datum per video, both codecs accepted
    for index in range(len(dataset)):
        stream, _target, metadata = dataset[index]
        frames = list(stream)
        assert len(frames) == _FRAMES
        assert frames[0].pixels.shape == (3, _HEIGHT, _WIDTH)
        assert (metadata["height"], metadata["width"]) == (_HEIGHT, _WIDTH)


@dataclass
class _FrameTracks:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    track_ids: np.ndarray


@dataclass
class _TrackingTarget:
    frame_tracks: list[_FrameTracks]


class _FrameCountingTracker:
    """Emits one empty set of tracks per decoded frame."""

    metadata = {"id": "frame-counting-tracker", "index2label": {}}

    def __call__(self, streams):
        empty = _FrameTracks(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), np.zeros(0, int))
        return [_TrackingTarget(frame_tracks=[empty for _frame in stream]) for stream in streams]


def test_flat_mp4_dataset_runs_through_cached_predict(flat_mp4_folder: Path) -> None:
    dataset = load_mot(flat_mp4_folder, dataset_format="flat_mp4").with_mot_options(empty_frame_policy="all")

    predictions, _ = cached_tasks.predict(model=_FrameCountingTracker(), dataset=dataset)

    targets = [target for batch in predictions for target in batch]
    assert len(targets) == 2
    assert all(len(target.frame_tracks) == _FRAMES for target in targets)
