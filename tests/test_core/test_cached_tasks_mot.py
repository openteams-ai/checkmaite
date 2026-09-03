from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from checkmaite import cached_tasks


@dataclass
class VideoFrame:
    pixels: np.ndarray
    frame_index: int
    pts: int
    time_s: float


@dataclass
class FrameTracks:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    track_ids: np.ndarray


@dataclass
class TrackingTarget:
    frame_tracks: list[FrameTracks]


def _target(track_id: int) -> TrackingTarget:
    return TrackingTarget(
        frame_tracks=[
            FrameTracks(
                boxes=np.asarray([[0.0, 0.0, 1.0, 1.0]]),
                labels=np.asarray([1]),
                scores=np.asarray([0.9]),
                track_ids=np.asarray([track_id]),
            )
        ]
    )


class DelegatingOneShotVideoStream:
    def __init__(self, frames: list[VideoFrame]) -> None:
        self._iterator = iter(frames)

    def __iter__(self):
        return self._iterator


class VideoDataset:
    metadata = {"id": "mot-video-dataset", "index2label": {}}

    def __init__(self) -> None:
        self.videos = [
            [VideoFrame(np.zeros((3, 2, 2)), 0, 0, 0.0)],
            [VideoFrame(np.ones((3, 2, 2)), 0, 0, 0.0)],
        ]

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> tuple[Any, TrackingTarget, dict[str, Any]]:
        return DelegatingOneShotVideoStream(self.videos[index]), _target(index), {"id": index}


class TrackingModel:
    metadata = {"id": "mot-tracking-model", "index2label": {}}

    def __init__(self) -> None:
        self.calls = 0
        self.received_materialized_streams: list[bool] = []

    def __call__(self, streams):
        self.calls += 1
        self.received_materialized_streams.extend(isinstance(stream, list) for stream in streams)
        materialized = [list(stream) for stream in streams]
        return [_target(int(frames[0].pixels[0, 0, 0])) for frames in materialized]


def test_mot_predictions_targets_and_track_ids_are_cached():
    model = TrackingModel()
    dataset = VideoDataset()

    first, _ = cached_tasks.predict(model=model, dataset=dataset)
    calls_after_first = model.calls
    second, _ = cached_tasks.predict(model=model, dataset=dataset)

    assert model.calls == calls_after_first
    for first_target, second_target in zip(first[0] + first[1], second[0] + second[1], strict=True):
        np.testing.assert_array_equal(
            first_target.frame_tracks[0].track_ids,
            second_target.frame_tracks[0].track_ids,
        )


def test_materializing_augmentation_preserves_original_metadata():
    import maite.protocols.generic as gen

    from checkmaite.core import cached_tasks as cached_tasks_module

    class IdentityAugmentation:
        metadata = {"id": "identity-augmentation"}

        def __call__(self, batch):
            return batch

    augmentation = IdentityAugmentation()
    wrapper = cached_tasks_module._materializing_augmentation(augmentation)

    assert isinstance(wrapper, gen.Augmentation)
    assert wrapper.metadata is augmentation.metadata


def test_full_mot_augmented_data_materializes_delegating_streams_and_stays_fresh():
    model = TrackingModel()
    dataset = VideoDataset()
    stream, _, _ = dataset[0]
    assert iter(stream) is not stream

    cached_tasks.predict(model=model, dataset=dataset)
    calls_after_ordinary = model.calls
    assert not any(model.received_materialized_streams)
    for _ in range(2):
        _, augmented_data = cached_tasks.predict(
            model=model,
            dataset=dataset,
            return_augmented_data=True,
        )
        assert isinstance(augmented_data[0][0][0], list)
        assert augmented_data[0][0][0][0].frame_index == 0
    calls_after_full_data = model.calls
    cached_tasks.predict(model=model, dataset=dataset)

    assert calls_after_full_data > calls_after_ordinary
    assert model.calls == calls_after_full_data
    assert all(model.received_materialized_streams[-len(dataset) :])


def test_strict_mot_targets_skip_prediction_cache_publication():
    model = TrackingModel()
    dataset = VideoDataset()

    for _ in range(2):
        with pytest.warns(UserWarning, match="Cache publication is disabled"):
            cached_tasks.predict(
                model=model,
                dataset=dataset,
                strict_cache_serialization=True,
            )

    assert model.calls == 2 * len(dataset)
