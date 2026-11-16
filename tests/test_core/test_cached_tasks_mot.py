from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
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

    first, _ = cached_tasks.predict(task="multiobject_tracking", model=model, dataset=dataset)
    calls_after_first = model.calls
    second, _ = cached_tasks.predict(task="multiobject_tracking", model=model, dataset=dataset)

    assert model.calls == calls_after_first
    for first_target, second_target in zip(first[0] + first[1], second[0] + second[1], strict=True):
        np.testing.assert_array_equal(
            first_target.frame_tracks[0].track_ids,
            second_target.frame_tracks[0].track_ids,
        )


def test_materializing_augmentation_preserves_original_metadata():
    import maite.protocols.generic as gen

    from checkmaite.core import cached_tasks as cached_tasks_module
    from checkmaite.core.evaluation_tasks import get_task

    class IdentityAugmentation:
        metadata = {"id": "identity-augmentation"}

        def __call__(self, batch):
            return batch

    augmentation = IdentityAugmentation()
    wrapper = cached_tasks_module._materializing_augmentation(augmentation, get_task("multiobject_tracking"))

    assert isinstance(wrapper, gen.Augmentation)
    assert wrapper.metadata is augmentation.metadata


def test_full_mot_augmented_data_materializes_delegating_streams_and_stays_fresh():
    model = TrackingModel()
    dataset = VideoDataset()
    stream, _, _ = dataset[0]
    assert iter(stream) is not stream

    cached_tasks.predict(task="multiobject_tracking", model=model, dataset=dataset)
    calls_after_ordinary = model.calls
    assert not any(model.received_materialized_streams)
    for _ in range(2):
        _, augmented_data = cached_tasks.predict(
            task="multiobject_tracking",
            model=model,
            dataset=dataset,
            return_augmented_data=True,
        )
        assert isinstance(augmented_data[0][0][0], list)
        assert augmented_data[0][0][0][0].frame_index == 0
    calls_after_full_data = model.calls
    cached_tasks.predict(task="multiobject_tracking", model=model, dataset=dataset)

    assert calls_after_full_data > calls_after_ordinary
    assert model.calls == calls_after_full_data
    assert all(model.received_materialized_streams[-len(dataset) :])


def test_strict_mot_targets_skip_prediction_cache_publication():
    model = TrackingModel()
    dataset = VideoDataset()

    for _ in range(2):
        with pytest.warns(UserWarning, match="Cache publication is disabled"):
            cached_tasks.predict(
                task="multiobject_tracking",
                model=model,
                dataset=dataset,
                strict_cache_serialization=True,
            )

    assert model.calls == 2 * len(dataset)


def test_augmented_data_keeps_streams_for_lazy_datamaite_targets():
    # Lazy frame tracks like DataMAITE's pass the tracking adapter, so the streams are materialized.
    # DataMAITE 0.7 made its target classes private, so LazyFrameTracks mirrors them.
    class LazyTargetVideoDataset(VideoDataset):
        def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
            stream, _, metadata = super().__getitem__(index)
            return stream, TrackingTarget(frame_tracks=LazyFrameTracks([0], {})), metadata  # type: ignore[arg-type]

    _, augmented_data = cached_tasks.predict(
        task="multiobject_tracking",
        model=TrackingModel(),
        dataset=LazyTargetVideoDataset(),
        return_augmented_data=True,
        use_cache=False,
    )

    returned_stream = augmented_data[0][0][0]
    assert isinstance(returned_stream, list)
    assert len(returned_stream) == 1


class LazyFrameTracks(Sequence[FrameTracks]):
    """Mirrors DataMAITE's lazy ``_FrameTracks``: its constructor does not take a list of frames."""

    def __init__(self, frame_order: Sequence[int], by_frame: dict[int, int]) -> None:
        self._frame_order = frame_order
        self._by_frame = by_frame

    def __len__(self) -> int:
        return len(self._frame_order)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return LazyFrameTracks(self._frame_order[index], self._by_frame)
        track_id = self._by_frame.get(self._frame_order[index])
        if track_id is None:
            return FrameTracks(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), np.zeros(0, int))
        return _target(track_id).frame_tracks[0]


class LazyTrackingDataset:
    metadata = {"id": "lazy-mot-video-dataset", "index2label": {}}

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[Any, TrackingTarget, dict[str, Any]]:
        frames = [VideoFrame(np.full((3, 2, 2), index), frame, frame, frame / 10) for frame in range(3)]
        target = TrackingTarget(frame_tracks=LazyFrameTracks(range(3), {0: index, 2: index + 10}))  # type: ignore[arg-type]
        # DataMAITE video datasets report time_base as a Fraction.
        return DelegatingOneShotVideoStream(frames), target, {"id": index, "time_base": Fraction(1, 10)}


class TrackIdMetric:
    def __init__(self, metric_id: str) -> None:
        self.metadata = {"id": metric_id}
        self.reset()

    def reset(self) -> None:
        self.seen: list[tuple[str, list[list[int]]]] = []

    def update(self, predictions, targets, metadata) -> None:
        for name, batch in (("pred", predictions), ("target", targets)):
            for value in batch:
                self.seen.append((name, [frame.track_ids.tolist() for frame in value.frame_tracks]))
        for datum_metadata in metadata:
            assert datum_metadata["time_base"] == Fraction(1, 10)

    def compute(self) -> dict[str, Any]:
        return {"seen": self.seen}


def _frames(target: Any) -> list[tuple[list[list[float]], list[int], list[float], list[int]]]:
    return [
        (frame.boxes.tolist(), frame.labels.tolist(), frame.scores.tolist(), frame.track_ids.tolist())
        for frame in target.frame_tracks
    ]


def test_lazy_frame_tracks_are_cached(recwarn: pytest.WarningsRecorder):
    model = TrackingModel()
    dataset = LazyTrackingDataset()

    fresh_results, fresh, _ = cached_tasks.evaluate(
        task="multiobject_tracking", model=model, dataset=dataset, metric=TrackIdMetric("first"), return_preds=True
    )
    calls_after_first = model.calls
    cached, _ = cached_tasks.predict(task="multiobject_tracking", model=model, dataset=dataset)
    # A new metric is scored from the cached predictions, targets and metadata.
    cached_results, _, _ = cached_tasks.evaluate(
        task="multiobject_tracking", model=model, dataset=dataset, metric=TrackIdMetric("second")
    )

    assert model.calls == calls_after_first
    assert cached_results == fresh_results
    assert ("target", [[0], [], [10]]) in cached_results["seen"]
    assert [_frames(p) for b in cached for p in b] == [_frames(p) for b in fresh for p in b]
    assert not [w for w in recwarn if "Cache publication is disabled" in str(w.message)]


def test_lazy_frame_tracks_reach_models_and_metrics_unchanged():
    dataset = LazyTrackingDataset()
    received: list[type] = []

    class RecordingMetric(TrackIdMetric):
        def update(self, predictions, targets, metadata) -> None:
            received.extend(type(target.frame_tracks) for target in targets)
            super().update(predictions, targets, metadata)

    cached_tasks.evaluate(
        task="multiobject_tracking", model=TrackingModel(), dataset=dataset, metric=RecordingMetric("recording")
    )

    assert received == [LazyFrameTracks, LazyFrameTracks]
