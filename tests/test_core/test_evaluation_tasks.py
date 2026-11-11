import dataclasses
import json
import math
import os
import subprocess
import sys
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest
import yaml
from maite.protocols import ArrayLike
from pydantic import AfterValidator, ConfigDict, TypeAdapter

from checkmaite.core import _plugins, cached_tasks, evaluation_tasks
from checkmaite.core._common.maite_evaluation_capability import MaiteEvaluationBase, MaiteEvaluationConfig
from checkmaite.core._plugins import PLUGIN_API_VERSION, list_loaded_plugins
from checkmaite.core.capability_core import CapabilityOutputsBase
from checkmaite.core.evaluation_tasks import EvaluationTask, get_task, list_tasks, register_task
from checkmaite.core.object_detection._task import PydanticCompatObjectDetectionTarget


@pytest.fixture(autouse=True)
def _restore_task_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(evaluation_tasks, "_tasks", dict(evaluation_tasks._tasks))
    monkeypatch.setattr(evaluation_tasks, "_aliases", dict(evaluation_tasks._aliases))
    # evaluation_tasks appends to this list object, so restore its contents in place.
    plugin_records = list(_plugins._registry)
    yield
    _plugins._registry[:] = plugin_records


_ARRAY_ADAPTER = TypeAdapter(ArrayLike, config=ConfigDict(arbitrary_types_allowed=True))
_UNRESOLVED = TypeAdapter(list["NotDefinedAnywhere"])  # noqa: F821


class SegmentationTarget(CapabilityOutputsBase):
    mask: ArrayLike


SEGMENTATION = EvaluationTask(
    name="semantic_segmentation",
    namespace=__name__,
    target_adapter=TypeAdapter(SegmentationTarget),
    aliases=("seg",),
    plugin_api_version=PLUGIN_API_VERSION,
)


class Mask:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask


class SegmentationDataset:
    metadata = {"id": "segmentation-dataset"}

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> tuple[np.ndarray, Mask, dict[str, Any]]:
        return np.zeros((3, 4, 4)), Mask(np.full((4, 4), index % 2)), {"id": index}


class SegmentationModel:
    metadata = {"id": "segmentation-model"}

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, inputs: Any) -> list[Mask]:
        self.calls += 1
        return [Mask(np.zeros((4, 4))) for _ in inputs]


class PixelAccuracy:
    def __init__(self, metric_id: str = "pixel-accuracy") -> None:
        self.metadata = {"id": metric_id}
        self.return_key = "pixel_accuracy"
        self.reset()

    def reset(self) -> None:
        self._scores: list[float] = []

    def update(self, predictions: Any, targets: Any, metadata: Any) -> None:
        self._scores.extend(
            float((np.asarray(p.mask) == np.asarray(t.mask)).mean()) for p, t in zip(predictions, targets, strict=True)
        )

    def compute(self) -> dict[str, float]:
        return {"pixel_accuracy": float(np.mean(self._scores))}


class SegmentationEvaluation(MaiteEvaluationBase[Any, Any, Any, MaiteEvaluationConfig]):
    task = "semantic_segmentation"

    @classmethod
    def _create_config(cls) -> MaiteEvaluationConfig:
        return MaiteEvaluationConfig()


def test_builtin_tasks_are_registered_with_aliases() -> None:
    names = [task.name for task in list_tasks()]
    assert {"image_classification", "object_detection", "multiobject_tracking"} <= set(names)
    for name, alias in [("image_classification", "ic"), ("object_detection", "od"), ("multiobject_tracking", "mot")]:
        assert get_task(alias) is get_task(name)
        assert get_task(name).namespace == f"checkmaite.core.{name}"


def test_task_is_required() -> None:
    with pytest.raises(TypeError, match="task"):
        cached_tasks.predict(model=SegmentationModel(), dataset=SegmentationDataset())  # type: ignore[call-arg]


@pytest.mark.parametrize("task", ["image_classification", "object_detection", "multiobject_tracking"])
def test_a_wrong_task_raises_before_anything_is_cached(task: str, tmp_cache_path: Path) -> None:
    register_task(SEGMENTATION)
    with pytest.raises(ValueError, match=f"do not match evaluation task {task!r}"):
        cached_tasks.predict(model=SegmentationModel(), dataset=SegmentationDataset(), task=task)
    assert not (tmp_cache_path / "cached-tasks" / "predict").exists() or not _cache_entries(tmp_cache_path, "predict")


def test_targets_that_do_not_match_the_task_raise() -> None:
    class WrongTargetsDataset(SegmentationDataset):
        def __getitem__(self, index: int) -> tuple[np.ndarray, Any, dict[str, Any]]:
            return np.zeros((3, 4, 4)), {"label": index}, {"id": index}

    register_task(SEGMENTATION)
    with pytest.raises(ValueError, match="The targets do not match evaluation task 'semantic_segmentation'"):
        cached_tasks.predict(model=SegmentationModel(), dataset=WrongTargetsDataset(), task="seg", use_cache=False)


def test_an_unknown_task_name_lists_the_registered_tasks() -> None:
    with pytest.raises(KeyError, match="Registered tasks: .*object_detection"):
        cached_tasks.predict(model=SegmentationModel(), dataset=SegmentationDataset(), task="semantic_segmentation")


def test_a_capability_without_a_task_cannot_run() -> None:
    class Taskless(SegmentationEvaluation):
        task = None

    with pytest.raises(TypeError, match="Taskless must set the class attribute 'task'"):
        Taskless().run(models=[SegmentationModel()], datasets=[SegmentationDataset()], metrics=[PixelAccuracy()])


def test_get_task_reports_registered_names() -> None:
    with pytest.raises(KeyError, match="object_detection"):
        get_task("semantic_segmentation")


def test_register_task_rejects_name_clashes_but_allows_reregistration() -> None:
    assert register_task(SEGMENTATION) is SEGMENTATION
    assert register_task(SEGMENTATION) is SEGMENTATION
    assert get_task("seg") is SEGMENTATION

    with pytest.raises(ValueError, match="already used"):
        register_task(
            EvaluationTask(name="segmentation", namespace="other", target_adapter=TypeAdapter(Any), aliases=("seg",))
        )
    with pytest.raises(ValueError, match="already used"):
        register_task(EvaluationTask(name="od", namespace="other", target_adapter=TypeAdapter(Any)))


def _entry_point(name: str, **load: Any) -> MagicMock:
    entry_point = MagicMock()
    entry_point.name = name
    entry_point.dist.name = f"fake-{name}"
    if "error" in load:
        entry_point.load.side_effect = load["error"]
    else:
        entry_point.load.return_value = load["value"]
    return entry_point


def test_task_entry_points_are_loaded_on_first_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    def segmentation(**changes: Any) -> EvaluationTask:
        return dataclasses.replace(SEGMENTATION, aliases=(), **changes)

    entry_points = [
        _entry_point("broken_import", error=ImportError("No module named 'missing_dependency'")),
        _entry_point("not_a_task", value="not a task"),
        _entry_point("malformed_target", value=segmentation(name="malformed", target_adapter=object())),
        _entry_point("unresolvable_target", value=segmentation(name="unresolvable", target_adapter=_UNRESOLVED)),
        _entry_point("no_api_version", value=segmentation(name="unversioned", plugin_api_version=None)),
        _entry_point("old_api_version", value=segmentation(name="old", plugin_api_version="0.9.0")),
        _entry_point("future_api_version", value=segmentation(name="future", plugin_api_version="2.0.0")),
        _entry_point("segmentation", value=SEGMENTATION),
    ]
    monkeypatch.setattr(evaluation_tasks.importlib_metadata, "entry_points", lambda group: entry_points)
    monkeypatch.setattr(evaluation_tasks, "_entry_points_loaded", False)

    assert get_task("semantic_segmentation") is SEGMENTATION
    assert {task.name for task in list_tasks()} == {
        "image_classification",
        "multiobject_tracking",
        "object_detection",
        "semantic_segmentation",
    }
    records = {r.entry_point_name: r for r in list_loaded_plugins(evaluation_tasks.TASK_ENTRY_POINT_GROUP)}
    assert records["segmentation"].status == "loaded"
    assert records["segmentation"].symbols == ["semantic_segmentation"]
    failed = {name: record.error for name, record in records.items() if record.status == "failed"}
    assert set(failed) == {entry_point.name for entry_point in entry_points} - {"segmentation"}
    assert "must be a pydantic TypeAdapter" in failed["malformed_target"]
    assert "cannot validate" in failed["unresolvable_target"]
    assert "must set 'plugin_api_version'" in failed["no_api_version"]
    assert "requires major version 1" in failed["old_api_version"]
    assert "requires major version 1" in failed["future_api_version"]


def test_register_task_rejects_malformed_target_adapters() -> None:
    with pytest.raises(TypeError, match="must be a pydantic TypeAdapter"):
        register_task(EvaluationTask(name="numbers", namespace="other", target_adapter=42))
    with pytest.raises(TypeError, match="cannot validate"):
        register_task(EvaluationTask(name="forward", namespace="other", target_adapter=_UNRESOLVED))
    assert "numbers" not in {task.name for task in list_tasks()}


def test_new_task_predictions_are_cached_and_restored() -> None:
    register_task(SEGMENTATION)
    model = SegmentationModel()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first, _, _ = cached_tasks.evaluate(
            model=model, metric=PixelAccuracy(), dataset=SegmentationDataset(), task="seg"
        )
        second, predictions, _ = cached_tasks.evaluate(
            task="seg",
            model=model,
            metric=PixelAccuracy("other-metric"),
            dataset=SegmentationDataset(),
            return_preds=True,
        )

    assert model.calls == 3
    assert first == second == {"pixel_accuracy": pytest.approx(2 / 3)}
    assert all(isinstance(prediction, SegmentationTarget) for batch in predictions for prediction in batch)


def test_maite_evaluation_runs_on_a_new_task() -> None:
    register_task(SEGMENTATION)
    model = SegmentationModel()
    capability = SegmentationEvaluation()

    run = capability.run(models=[model], datasets=[SegmentationDataset()], metrics=[PixelAccuracy()])
    capability.run(models=[model], datasets=[SegmentationDataset()], metrics=[PixelAccuracy("other-metric")])

    assert run.outputs.metrics["pixel-accuracy"].overall_metric_value == pytest.approx(2 / 3)
    assert model.calls == 3  # the second run reuses the cached predictions


def test_prediction_cache_entries_require_a_task() -> None:
    target = {"boxes": np.zeros((1, 4)), "labels": np.zeros(1), "scores": np.ones(1)}
    with pytest.raises(ValueError, match="must name its evaluation task"):
        cached_tasks._PredictCall.model_validate(
            {
                "config": {
                    "model_id": "m",
                    "dataset_id": "d",
                    "augmentation_id": None,
                    "task": "object_detection",
                    "batch_size": 1,
                },
                "predictions": [[target]],
                "augmented_data": [([], [target], [{"id": 0}])],
            }
        )


_SEGMENTATION_PACKAGE = """
import numpy as np
from maite.protocols import ArrayLike

from checkmaite.core._common.maite_evaluation_capability import MaiteEvaluationBase, MaiteEvaluationConfig
from checkmaite.core._plugins import PLUGIN_API_VERSION
from checkmaite.core.capability_core import CapabilityOutputsBase
from checkmaite.core.evaluation_tasks import EvaluationTask
from pydantic import TypeAdapter

MODEL_CALLS = []


class SegmentationTarget(CapabilityOutputsBase):
    mask: ArrayLike


class Mask:
    def __init__(self, mask):
        self.mask = mask


class SegmentationDataset:
    def __init__(self, **kwargs):
        self.metadata = {"id": "segmentation-dataset", "index2label": {0: "background", 1: "road"}}

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return np.zeros((3, 4, 4)), Mask(np.full((4, 4), index % 2)), {"id": index}


class SegmentationModel:
    def __init__(self, **kwargs):
        self.metadata = {"id": "segmentation-model", "index2label": {0: "background", 1: "road"}}

    def __call__(self, inputs):
        MODEL_CALLS.append(len(inputs))
        return [Mask(np.zeros((4, 4))) for _ in inputs]


class PixelAccuracy:
    def __init__(self, metric_id, **kwargs):
        self.metadata = {"id": metric_id}
        self.return_key = "pixel_accuracy"
        self.reset()

    def reset(self):
        self._scores = []

    def update(self, predictions, targets, metadata):
        self._scores.extend(float((np.asarray(p.mask) == np.asarray(t.mask)).mean()) for p, t in zip(predictions, targets))

    def compute(self):
        return {"pixel_accuracy": float(np.mean(self._scores))}


class MaiteEvaluation(MaiteEvaluationBase):
    task = "semantic_segmentation"

    @classmethod
    def _create_config(cls):
        return MaiteEvaluationConfig()


TASK = EvaluationTask(
    name="semantic_segmentation",
    namespace="fake_segmentation_plugin",
    target_adapter=TypeAdapter(SegmentationTarget),
    plugin_api_version=PLUGIN_API_VERSION,
)
"""

_WORKFLOW = """
import json
import sys

import fake_segmentation_plugin
from checkmaite.core._plugins import list_loaded_plugins
from checkmaite.core.evaluation_tasks import get_task
from checkmaite_container import cli

status = cli.main(["run", "--config", sys.argv[1], "--output", sys.argv[2], "--cache", sys.argv[3]])
records = list_loaded_plugins("checkmaite.tasks")
print(json.dumps({
    "status": status,
    "model_calls": fake_segmentation_plugin.MODEL_CALLS,
    "task_namespace": get_task("semantic_segmentation").namespace,
    "plugins": [[r.entry_point_name, r.status, r.symbols] for r in records],
}))
"""


def _install_segmentation_package(site: Path) -> None:
    (site / "fake_segmentation_plugin.py").write_text(_SEGMENTATION_PACKAGE)
    dist_info = site / "fake_segmentation_plugin-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-segmentation-plugin\nVersion: 0.0.0\n")
    (dist_info / "entry_points.txt").write_text(
        "[checkmaite.tasks]\nsemantic_segmentation = fake_segmentation_plugin:TASK\n"
    )


def test_installed_package_adds_a_task_without_core_changes(tmp_path: Path) -> None:
    """A task package is found through its entry point and evaluated by the batch container, with caching."""
    site = tmp_path / "site"
    site.mkdir()
    _install_segmentation_package(site)

    def metric(metric_id: str) -> dict[str, Any]:
        return {"class": "fake_segmentation_plugin.PixelAccuracy", "args": {"metric_id": metric_id}}

    plan = {
        "version": 1,
        "resources": {"device": "cpu"},
        "datasets": {"roads": {"class": "fake_segmentation_plugin.SegmentationDataset"}},
        "models": {"segmenter": {"class": "fake_segmentation_plugin.SegmentationModel"}},
        "metrics": {"accuracy": metric("accuracy"), "accuracy-again": metric("accuracy-again")},
        "tasks": [
            # The package's task is named in builtin: references like a built-in one.
            {
                "name": "first",
                "capability": "builtin:semantic_segmentation.MaiteEvaluation",
                "dataset": "roads",
                "model": "segmenter",
                "metrics": "accuracy",
            },
            # A second metric on the same model and dataset reuses the cached predictions.
            {
                "name": "second",
                "capability": "builtin:semantic_segmentation.MaiteEvaluation",
                "dataset": "roads",
                "model": "segmenter",
                "metrics": "accuracy-again",
            },
        ],
    }
    plan_path = tmp_path / "run.yaml"
    plan_path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    output = tmp_path / "output"
    cache = tmp_path / "cache"

    pythonpath = [str(site), str(Path(__file__).parents[2] / "docker" / "runtime" / "src")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [*pythonpath, os.environ.get("PYTHONPATH")]))}
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _WORKFLOW, str(plan_path), str(output), str(cache)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["status"] == 0, result.stderr
    assert outcome["plugins"] == [["semantic_segmentation", "loaded", ["semantic_segmentation"]]]
    assert outcome["task_namespace"] == "fake_segmentation_plugin"
    # One pass over the three images; the second task reads the cached predictions.
    assert sum(outcome["model_calls"]) == 3

    predictions = [json.loads(path.read_text()) for path in (cache / "cached-tasks" / "predict").iterdir()]
    assert [entry["dumped_fields"]["task"] for entry in predictions] == ["semantic_segmentation"]

    summary = json.loads((output / "run-results.json").read_text())
    assert [task["name"] for task in summary["tasks"]] == ["first", "second"]
    for task, metric_id in [("first", "accuracy"), ("second", "accuracy-again")]:
        run = json.loads((output / "tasks" / task / "run.json").read_text())
        assert run["capability_id"] == "fake_segmentation_plugin.MaiteEvaluation"
        assert run["outputs"]["metrics"][metric_id]["scalar_values"]["pixel_accuracy"] == pytest.approx(2 / 3)
        report = next(t for t in summary["tasks"] if t["name"] == task)["report_file"]
        assert report is not None
        report_text = Path(report).read_text()
        assert f"### {metric_id}" in report_text
        assert "| pixel_accuracy | 0.6667 |" in report_text

    runs = pl.read_parquet(output / "analytics" / "runs" / "*.parquet")
    assert set(runs["run_uid"]) == {task["run_uid"] for task in summary["tasks"]}
    metrics = pl.read_parquet(output / "analytics" / "maite_evaluation" / "*.parquet").sort("metric_id")
    assert metrics["metric_id"].to_list() == ["accuracy", "accuracy-again"]
    assert set(metrics["model_id"]) == {"segmentation-model"}
    assert set(metrics["dataset_id"]) == {"segmentation-dataset"}
    assert metrics["output_value"].to_list() == pytest.approx([2 / 3, 2 / 3])


class BoxesWithMask(CapabilityOutputsBase):
    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike
    mask: ArrayLike


INSTANCE_SEGMENTATION = EvaluationTask(
    name="instance_segmentation",
    namespace=__name__,
    target_adapter=TypeAdapter(BoxesWithMask),
)


def _boxes_with_mask() -> BoxesWithMask:
    return BoxesWithMask(boxes=np.zeros((1, 4)), labels=np.zeros(1), scores=np.ones(1), mask=np.ones((4, 4)))


class BoxesWithMaskDataset:
    """Targets that also satisfy the object-detection target type."""

    metadata = {"id": "boxes-with-mask-dataset"}

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[np.ndarray, BoxesWithMask, dict[str, Any]]:
        return np.zeros((3, 4, 4)), _boxes_with_mask(), {"id": index}


class BoxesWithMaskModel:
    metadata = {"id": "boxes-with-mask-model"}

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, inputs: Any) -> list[BoxesWithMask]:
        self.calls += 1
        return [_boxes_with_mask() for _ in inputs]


class MaskSum:
    """Fails on predictions that lost their mask."""

    def __init__(self) -> None:
        self.metadata = {"id": "mask-sum"}
        self.reset()

    def reset(self) -> None:
        self._total = 0.0

    def update(self, predictions: Any, targets: Any, metadata: Any) -> None:
        self._total += sum(float(np.asarray(prediction.mask).sum()) for prediction in predictions)

    def compute(self) -> dict[str, float]:
        return {"mask_sum": self._total}


def _cache_entries(cache: Path, kind: str) -> list[Path]:
    return list((cache / "cached-tasks" / kind).iterdir())


def test_explicit_tasks_have_separate_prediction_entries(tmp_cache_path: Path) -> None:
    register_task(INSTANCE_SEGMENTATION)
    model = BoxesWithMaskModel()
    dataset = BoxesWithMaskDataset()

    # Named first as the package task, then as object detection: two entries.
    cached_tasks.predict(model=model, dataset=dataset, task="instance_segmentation")
    cached_tasks.predict(model=model, dataset=dataset, task="object_detection")
    assert model.calls == 4

    named, _ = cached_tasks.predict(model=model, dataset=dataset, task="instance_segmentation")
    detected, _ = cached_tasks.predict(model=model, dataset=dataset, task="object_detection")
    assert model.calls == 4
    assert type(named[0][0]) is BoxesWithMask
    assert type(detected[0][0]) is PydanticCompatObjectDetectionTarget
    assert len(_cache_entries(tmp_cache_path, "predict")) == 2

    # An alias shares its task's entry.
    aliased, _ = cached_tasks.predict(model=model, dataset=dataset, task="od")
    assert type(aliased[0][0]) is PydanticCompatObjectDetectionTarget
    assert model.calls == 4


def test_explicit_task_does_not_reuse_metric_results_cached_for_another_task(tmp_cache_path: Path) -> None:
    register_task(INSTANCE_SEGMENTATION)
    model = BoxesWithMaskModel()
    dataset = BoxesWithMaskDataset()

    # Named as object detection, then as instance segmentation.
    result, _, _ = cached_tasks.evaluate(model=model, metric=MaskSum(), dataset=dataset, task="object_detection")
    assert result == {"mask_sum": 32.0}
    result, _, _ = cached_tasks.evaluate(model=model, metric=MaskSum(), dataset=dataset, task="instance_segmentation")
    assert model.calls == 4
    assert result == {"mask_sum": 32.0}
    assert len(_cache_entries(tmp_cache_path, "evaluate-from-predictions")) == 2

    for task in ("object_detection", "instance_segmentation"):
        result, _, _ = cached_tasks.evaluate(model=model, metric=MaskSum(), dataset=dataset, task=task)
        assert result == {"mask_sum": 32.0}
    assert model.calls == 4
    assert len(_cache_entries(tmp_cache_path, "evaluate-from-predictions")) == 2


class VectorDataset:
    metadata = {"id": "vector-dataset"}

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return np.zeros((3, 4, 4)), np.array([1.0, 0.0]), {"id": index}


class VectorModel:
    metadata = {"id": "vector-model"}

    def __call__(self, inputs: Any) -> list[np.ndarray]:
        assert all(isinstance(value, np.ndarray) for value in inputs), "inputs were materialized into lists"
        return [np.array([1.0, 0.0]) for _ in inputs]


def test_a_streaming_task_with_the_same_targets_does_not_change_image_classification() -> None:
    # Video classification targets are arrays too, and its inputs are streams.
    register_task(
        EvaluationTask(
            name="video_classification", namespace=__name__, target_adapter=_ARRAY_ADAPTER, materialize_inputs=True
        )
    )
    predictions, augmented = cached_tasks.predict(
        model=VectorModel(), dataset=VectorDataset(), use_cache=False, return_augmented_data=True, task="ic"
    )
    assert all(isinstance(value, np.ndarray) for inputs, _, _ in augmented for value in inputs)
    assert len(predictions) == 2


def test_augmented_data_accepts_a_batch_of_targets_held_in_one_array() -> None:
    # A dataloader may stack a batch's targets into a single array.
    batch = ([np.zeros((3, 4, 4)), np.zeros((3, 4, 4))], np.eye(2), [{"id": 0}, {"id": 1}])
    predictions, _ = cached_tasks.predict(
        model=VectorModel(), dataloader=[batch], use_cache=False, return_augmented_data=True, task="ic"
    )
    assert len(predictions[0]) == 2


class FrameList(Sequence[Any]):
    """A sequence pydantic cannot rebuild, like DataMAITE's lazily decoded frame tracks."""

    def __init__(self, frames: list[Any], by_frame: bool) -> None:
        self._frames = frames

    def __getitem__(self, index: Any) -> Any:
        return self._frames[index]

    def __len__(self) -> int:
        return len(self._frames)


class LazyTracks:
    def __init__(self) -> None:
        frame = SimpleNamespace(boxes=np.zeros((1, 4)), labels=np.zeros(1), scores=np.ones(1), track_ids=np.zeros(1))
        self.frame_tracks = FrameList([frame], by_frame=True)


def test_a_target_that_fails_to_validate_with_a_type_error_does_not_match() -> None:
    # Pydantic only turns ValueError into ValidationError, so a TypeError escapes the adapter.
    def reject(value: object) -> object:
        raise TypeError("cannot copy")

    task = EvaluationTask(
        name="rejecting", namespace=__name__, target_adapter=TypeAdapter(Annotated[object, AfterValidator(reject)])
    )
    assert not task._matches([object()])


def test_lazily_built_tracks_match_tracking() -> None:
    # Pydantic would rebuild the sequence by calling FrameList(items); the tracking adapter copies it instead.
    assert get_task("mot")._matches([LazyTracks()])
    assert not get_task("mot")._matches([object()])


class CountingDataset(BoxesWithMaskDataset):
    """Counts reads, and fails them once storage is switched off."""

    def __init__(self) -> None:
        self.reads = 0
        self.available = True

    def __getitem__(self, index: int) -> tuple[np.ndarray, BoxesWithMask, dict[str, Any]]:
        if not self.available:
            raise OSError("dataset storage is unavailable")
        self.reads += 1
        return super().__getitem__(index)


def test_cache_lookups_do_not_read_the_dataset(tmp_cache_path: Path) -> None:
    model = BoxesWithMaskModel()
    dataset = CountingDataset()

    cached_tasks.evaluate(model=model, metric=MaskSum(), dataset=dataset, task="od")
    assert dataset.reads == len(dataset)

    dataset.available = False
    result, predictions, _ = cached_tasks.evaluate(
        model=model, metric=MaskSum(), dataset=dataset, return_preds=True, task="od"
    )
    assert result == {"mask_sum": 32.0}
    assert len(predictions) == 2
    assert model.calls == 2


SCALAR_REGRESSION = EvaluationTask(name="scalar_regression", namespace=__name__, target_adapter=TypeAdapter(float))


class ScalarDataset:
    metadata = {"id": "scalar-dataset"}

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[np.ndarray, float, dict[str, Any]]:
        return np.zeros(1), 0.0, {"id": index}


class LargeIntegerModel:
    metadata = {"id": "large-integer-model"}

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, inputs: Any) -> list[np.int64]:
        self.calls += 1
        # 2**53 + 1 has no exact float.
        return [np.int64(2**53 + 1) for _ in inputs]


def test_strict_caching_rejects_a_lossy_conversion_by_the_adapter() -> None:
    register_task(SCALAR_REGRESSION)
    model = LargeIntegerModel()

    for _ in range(2):
        with pytest.warns(UserWarning, match="changed from int64 to float"):
            predictions, _ = cached_tasks.predict(
                model=model,
                dataset=ScalarDataset(),
                task="scalar_regression",
                strict_cache_serialization=True,
            )
        assert predictions[0][0] == np.int64(2**53 + 1)

    assert model.calls == 2  # nothing was cached


def test_an_unregistered_task_object_is_rejected() -> None:
    with pytest.raises(ValueError, match="not registered"):
        cached_tasks.predict(model=SegmentationModel(), dataset=SegmentationDataset(), task=SEGMENTATION)

    register_task(SEGMENTATION)
    model = SegmentationModel()
    for _ in range(2):
        cached_tasks.predict(model=model, dataset=SegmentationDataset(), task=SEGMENTATION)
    assert model.calls == len(SegmentationDataset())  # the second call is a cache hit


class NegativeZeroModel:
    metadata = {"id": "negative-zero-model"}

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, inputs: Any) -> list[float]:
        self.calls += 1
        return [-0.0 for _ in inputs]


def test_strict_caching_rejects_a_conversion_that_flips_the_sign_of_zero() -> None:
    register_task(
        EvaluationTask(
            name="absolute", namespace=__name__, target_adapter=TypeAdapter(Annotated[float, AfterValidator(abs)])
        )
    )
    model = NegativeZeroModel()

    for _ in range(2):
        with pytest.warns(UserWarning, match="changed value"):
            predictions, _ = cached_tasks.predict(
                model=model, dataset=ScalarDataset(), task="absolute", strict_cache_serialization=True
            )
        assert math.copysign(1.0, predictions[0][0]) == -1.0

    assert model.calls == 2  # nothing was cached


def test_registering_an_equal_task_returns_the_registered_one() -> None:
    task = register_task(dataclasses.replace(get_task("ic")))
    assert task is get_task("ic")

    model = VectorModel()
    predictions, _ = cached_tasks.predict(model=model, dataset=VectorDataset(), task=task, use_cache=False)
    assert len(predictions) == 2


@pytest.mark.parametrize("name", ["image_classification", "object_detection", "multiobject_tracking"])
def test_builtin_task_namespaces_export_bound_cached_tasks(name: str) -> None:
    import importlib

    module = importlib.import_module(f"checkmaite.core.{name}.cached_tasks")
    assert module.evaluate.__self__.task is get_task(name)
    assert module.predict.__self__.task is get_task(name)
    assert module.evaluate_from_predictions is cached_tasks.evaluate_from_predictions


@pytest.mark.parametrize("function", ["evaluate", "predict"])
def test_bound_functions_have_the_module_signature_without_task(function: str) -> None:
    import inspect

    import maite.tasks

    bound = inspect.signature(getattr(cached_tasks.for_task(SEGMENTATION), function))
    generic = inspect.signature(getattr(cached_tasks, function))
    assert list(bound.parameters.values()) == [p for name, p in generic.parameters.items() if name != "task"]

    # Every MAITE argument is accepted with MAITE's default, except collate_fn,
    # which CheckMAITE does not support because it has no stable cache identity.
    maite_parameters = inspect.signature(getattr(maite.tasks, function)).parameters
    for name, parameter in maite_parameters.items():
        if name == "collate_fn":
            continue
        assert bound.parameters[name].default == parameter.default, name
    # CheckMAITE's own arguments all have defaults, so MAITE calls work unchanged.
    extra = set(bound.parameters) - set(maite_parameters)
    assert all(bound.parameters[name].default is not inspect.Parameter.empty for name in extra)


def test_a_package_binds_its_task_for_its_users() -> None:
    segmentation = cached_tasks.for_task(SEGMENTATION)
    model = SegmentationModel()

    # Binding does not register the task; calling an unregistered task fails.
    with pytest.raises(ValueError, match="not registered"):
        segmentation.predict(model=model, dataset=SegmentationDataset())

    register_task(SEGMENTATION)
    first, _, _ = segmentation.evaluate(model=model, metric=PixelAccuracy(), dataset=SegmentationDataset())
    second, _, _ = segmentation.evaluate(model=model, metric=PixelAccuracy("again"), dataset=SegmentationDataset())
    assert first == second == {"pixel_accuracy": pytest.approx(2 / 3)}
    assert model.calls == 3  # the second call reuses the cached predictions


def test_an_equal_copy_of_a_registered_task_is_accepted() -> None:
    copy = dataclasses.replace(get_task("ic"))
    assert copy is not get_task("ic")
    predictions, _ = cached_tasks.predict(model=VectorModel(), dataset=VectorDataset(), task=copy, use_cache=False)
    assert len(predictions) == 2
    bound, _ = cached_tasks.for_task(copy).predict(model=VectorModel(), dataset=VectorDataset(), use_cache=False)
    assert len(bound) == 2


def test_a_wrong_task_is_caught_after_an_empty_first_batch() -> None:
    register_task(SEGMENTATION)
    batches = [([], [], []), ([np.zeros((3, 4, 4))], [Mask(np.zeros((4, 4)))], [{"id": 0}])]
    with pytest.raises(ValueError, match="do not match evaluation task 'object_detection'"):
        cached_tasks.predict(model=SegmentationModel(), dataloader=batches, task="od", use_cache=False)


def test_failed_entry_points_do_not_leak_plugin_records() -> None:
    # Runs after test_task_entry_points_are_loaded_on_first_lookup in file order.
    assert not [r for r in _plugins._registry if r.group == evaluation_tasks.TASK_ENTRY_POINT_GROUP]


def test_a_capability_for_an_unregistered_task_cannot_run() -> None:
    class UnknownTask(SegmentationEvaluation):
        task = "not_a_task"

    with pytest.raises(KeyError, match="Unknown evaluation task 'not_a_task'"):
        UnknownTask().run(models=[SegmentationModel()], datasets=[SegmentationDataset()], metrics=[PixelAccuracy()])


def _key_written_by_070(fields: dict[str, Any]) -> str:
    # CheckMAITE 0.7.0 keyed cache entries on the JSON of these config fields, in this
    # order, with cache_schema_version 1 and no task.
    import hashlib

    return hashlib.sha256(json.dumps({"cache_schema_version": 1, **fields}, separators=(",", ":")).encode()).hexdigest()


def test_entries_at_070_cache_keys_are_not_reused(tmp_cache_path: Path) -> None:
    register_task(SEGMENTATION)
    model, dataset, metric = SegmentationModel(), SegmentationDataset(), PixelAccuracy()
    cached_tasks.evaluate(model=model, metric=metric, dataset=dataset, task="seg")
    calls_after_first_run = model.calls

    # Move each entry to where 0.7.0 would have stored it. The entries are readable
    # by this version, so the model running again shows the 0.7.0 keys are not looked up.
    predict_key = _key_written_by_070(
        {
            "model_id": "segmentation-model",
            "dataset_id": "segmentation-dataset",
            "augmentation_id": None,
            "batch_size": 1,
            "strict_cache_serialization": False,
        }
    )
    evaluation_key = _key_written_by_070(
        {
            "inference_id": predict_key,
            "model_id": "segmentation-model",
            "dataset_id": "segmentation-dataset",
            "augmentation_id": None,
            "metric_id": "pixel-accuracy",
            "prediction_batch_size": 1,
            "cpu_prediction_postprocessor_id": None,
            "strict_cache_serialization": False,
        }
    )
    for kind, key in (("predict", predict_key), ("evaluate-from-predictions", evaluation_key)):
        [entry] = _cache_entries(tmp_cache_path, kind)
        entry.rename(entry.with_name(key))

    result, _, _ = cached_tasks.evaluate(model=model, metric=metric, dataset=dataset, task="seg")

    assert model.calls == 2 * calls_after_first_run
    assert result == {"pixel_accuracy": pytest.approx(2 / 3)}
