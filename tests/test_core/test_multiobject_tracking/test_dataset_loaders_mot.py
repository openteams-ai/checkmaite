import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest
from datamaite import load_mot
from PIL import Image

from checkmaite import cached_tasks
from checkmaite.core.multiobject_tracking import MaiteEvaluation, load_mot_dataset
from checkmaite.core.multiobject_tracking.metrics import clear_mot_metrics_factory
from tests.test_core.test_multiobject_tracking.conftest import FakeMOTTarget, frame_tracks

_FRAMES = 2
_BOX = [0.0, 0.0, 10.0, 10.0]


def _write_motchallenge(root: Path, box_xywh: tuple[int, int, int, int], labeled_frames: int = _FRAMES) -> Path:
    """Write a one-sequence MOTChallenge root whose pedestrian sits at ``box_xywh`` in the first ``labeled_frames``."""
    sequence = root / "train" / "SEQ-01"
    (sequence / "img1").mkdir(parents=True)
    (sequence / "gt").mkdir()
    for frame in range(1, _FRAMES + 1):
        Image.new("RGB", (64, 48)).save(sequence / "img1" / f"{frame:06d}.jpg")
    (sequence / "seqinfo.ini").write_text(
        f"[Sequence]\nname=SEQ-01\nimDir=img1\nframeRate=10\nseqLength={_FRAMES}\nimWidth=64\nimHeight=48\nimExt=.jpg\n"
    )
    x, y, w, h = box_xywh
    rows = [f"{frame},1,{x},{y},{w},{h},1,1,1.0" for frame in range(1, labeled_frames + 1)]
    (sequence / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")
    return root


@pytest.fixture
def mot_root_a(tmp_path: Path) -> Path:
    return _write_motchallenge(tmp_path / "a", (0, 0, 10, 10))


@pytest.fixture
def mot_root_b(tmp_path: Path) -> Path:
    return _write_motchallenge(tmp_path / "b", (40, 30, 10, 10))


class FixedBoxTracker:
    """Tracks one object at ``_BOX`` in every frame of every video."""

    metadata = {"id": "fixed-box-tracker"}

    def __call__(self, streams):
        return [FakeMOTTarget([frame_tracks([_BOX], [1]) for _ in stream]) for stream in streams]


def test_dataset_id_is_derived_and_stable(mot_root_a: Path) -> None:
    first = load_mot_dataset(mot_root_a, "motchallenge")
    second = load_mot_dataset(mot_root_a, "motchallenge")

    assert first.metadata["id"].startswith("mot_motchallenge_")
    assert first.metadata["id"] == second.metadata["id"]


def test_dataset_id_changes_with_everything_that_changes_targets(mot_root_a: Path, mot_root_b: Path) -> None:
    ids = {
        load_mot_dataset(mot_root_a, "motchallenge").metadata["id"],
        load_mot_dataset(mot_root_b, "motchallenge").metadata["id"],
        load_mot_dataset(mot_root_a, "motchallenge", classes={1}).metadata["id"],
    }
    assert len(ids) == 3


def test_explicit_dataset_id_is_applied(mot_root_a: Path) -> None:
    assert load_mot_dataset(mot_root_a, "motchallenge", "my-dataset").metadata["id"] == "my-dataset"


def test_every_frame_is_emitted_including_unlabeled_ones(tmp_path: Path) -> None:
    """datamaite's default would emit only the labeled first frame; evaluation needs both."""
    root = _write_motchallenge(tmp_path / "partly_labeled", (0, 0, 10, 10), labeled_frames=1)

    stream, target, _ = load_mot_dataset(root, "motchallenge")[0]

    assert len(list(stream)) == len(target.frame_tracks) == _FRAMES
    assert len(target.frame_tracks[1].track_ids) == 0


@pytest.mark.parametrize(
    ("dataset_format", "reason"),
    [
        ("tao", "labels only some frames"),
        ("hmie", "labels only some frames"),
        ("visdrone_video", "track"),
        ("coco", "not a supported multi-object-tracking format"),
    ],
)
def test_formats_that_cannot_be_evaluated_over_every_frame_are_refused(dataset_format: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        load_mot_dataset("does/not/need/to/exist", dataset_format)


def test_sequences_without_an_exact_frame_count_are_refused(mot_root_a: Path, monkeypatch) -> None:
    from checkmaite.core.multiobject_tracking import dataset_loaders

    def load_with_estimated_count(*args, **kwargs):
        dataset = load_mot(*args, **kwargs)
        sequences = tuple(dataclasses.replace(seq, num_frames_exact=False) for seq in dataset.sequences)
        return dataclasses.replace(dataset, sequences=sequences)

    monkeypatch.setattr(dataset_loaders, "load_mot", load_with_estimated_count)

    with pytest.raises(ValueError, match="exact frame count is unknown"):
        load_mot_dataset(mot_root_a, "motchallenge")


def test_different_datasets_do_not_share_cached_evaluations(mot_root_a: Path, mot_root_b: Path) -> None:
    """The same tracker on two datasets must be scored on each, not served the first one's cached result."""
    tracker = FixedBoxTracker()
    results = []
    for root in (mot_root_a, mot_root_b):
        run = MaiteEvaluation().run(
            models=[tracker],
            datasets=[load_mot_dataset(root, "motchallenge")],
            metrics=[clear_mot_metrics_factory()],
        )
        (metric_result,) = run.outputs.metrics.values()
        results.append(metric_result.scalar_values["mota"])

    # The tracker's box matches dataset a exactly and misses dataset b entirely (1 miss + 1 FP per frame).
    assert results == [pytest.approx(1.0), pytest.approx(-1.0)]
    assert np.isfinite(results).all()


class CountingTracker(FixedBoxTracker):
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, streams):
        self.calls += 1
        return super().__call__(streams)


def _assert_predictions_and_evaluations_are_cached(dataset, recwarn: pytest.WarningsRecorder) -> None:
    tracker = CountingTracker()
    metric = clear_mot_metrics_factory()

    fresh_results, fresh_preds, _ = cached_tasks.evaluate(
        task="multiobject_tracking", model=tracker, dataset=dataset, return_preds=True
    )
    calls_after_first = tracker.calls
    cached_preds, _ = cached_tasks.predict(task="multiobject_tracking", model=tracker, dataset=dataset)
    # The metric is scored from cached predictions and targets the first time, then served from its own cache.
    first_scores, _, _ = cached_tasks.evaluate(
        task="multiobject_tracking", model=tracker, dataset=dataset, metric=metric
    )
    second_scores, _, _ = cached_tasks.evaluate(
        task="multiobject_tracking", model=tracker, dataset=dataset, metric=metric
    )

    assert calls_after_first > 0
    assert tracker.calls == calls_after_first
    assert fresh_results == {}
    assert first_scores == second_scores
    for fresh, cached in zip(
        [p for batch in fresh_preds for p in batch], [p for batch in cached_preds for p in batch], strict=True
    ):
        for fresh_frame, cached_frame in zip(fresh.frame_tracks, cached.frame_tracks, strict=True):
            np.testing.assert_array_equal(fresh_frame.track_ids, cached_frame.track_ids)
            np.testing.assert_array_equal(fresh_frame.boxes, cached_frame.boxes)
    assert not [w for w in recwarn if "Cache publication is disabled" in str(w.message)]


def test_datamaite_mot_predictions_and_evaluations_are_cached(
    mot_root_a: Path, recwarn: pytest.WarningsRecorder
) -> None:
    """datamaite's lazy frame_tracks and Fraction time_base used to stop every MOT run being cached."""
    _assert_predictions_and_evaluations_are_cached(load_mot_dataset(mot_root_a, "motchallenge"), recwarn)


_EXAMPLE_DATASETS = os.environ.get("DATAMAITE_EXAMPLE_DATASETS")


@pytest.mark.skipif(
    not _EXAMPLE_DATASETS or not Path(_EXAMPLE_DATASETS, "hmie", "real-meva").is_dir(),
    reason="set DATAMAITE_EXAMPLE_DATASETS to the datamaite example datasets directory",
)
def test_datamaite_example_video_predictions_are_cached(recwarn: pytest.WarningsRecorder) -> None:
    dataset = load_mot(Path(str(_EXAMPLE_DATASETS), "hmie", "real-meva"), dataset_format="hmie")
    dataset.metadata["id"] = "datamaite-example-hmie-real-meva"
    _assert_predictions_and_evaluations_are_cached(dataset, recwarn)
