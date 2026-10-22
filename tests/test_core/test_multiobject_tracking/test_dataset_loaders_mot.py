import dataclasses
from pathlib import Path

import numpy as np
import pytest
from datamaite import load_mot
from PIL import Image

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
        results.append(metric_result.overall_metric_value)

    # The tracker's box matches dataset a exactly and misses dataset b entirely (1 miss + 1 FP per frame).
    assert results == [pytest.approx(1.0), pytest.approx(-1.0)]
    assert np.isfinite(results).all()
