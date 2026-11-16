"""Tests for checkmaite.core.component_checks.check_components."""

import builtins
import sys

import pytest
import torch
from modelmaite.object_detection import DetectionTarget
from pydantic import TypeAdapter

from checkmaite.core.component_checks import check_components
from checkmaite.core.evaluation_tasks import EvaluationTask
from tests.conftest import FakeODDataset, FakeODMetric, FakeODModel

pytestmark = pytest.mark.optional


@pytest.fixture(autouse=True)
def _needs_beartype():
    pytest.importorskip("beartype")


def _spotcheck_error():
    from maite.spotcheck import SpotcheckError

    return SpotcheckError


def _five_boxes_four_labels():
    return DetectionTarget(
        boxes=torch.zeros((5, 4)),
        labels=torch.zeros(4, dtype=torch.int32),
        scores=torch.ones(5),
    )


class IdentityAugmentation:
    metadata = {"id": "identity"}

    def __call__(self, batch):
        return batch


class BrokenAugmentation(IdentityAugmentation):
    def __call__(self, batch):
        inputs, targets, metadata = batch
        return inputs, [_five_boxes_four_labels() for _ in targets], metadata


def test_passes_compliant_od_components(fake_od_model_default, fake_od_dataset_default, fake_od_metric_default):
    checked = check_components(
        task="object_detection",
        model=fake_od_model_default,
        dataset=fake_od_dataset_default,
        metric=fake_od_metric_default,
        augmentation=IdentityAugmentation(),
    )
    assert checked == ["dataset", "augmentation", "model", "metric"]


def test_passes_compliant_ic_components(fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default):
    checked = check_components(
        task="ic",
        model=fake_ic_model_default,
        dataset=fake_ic_dataset_default,
        metric=fake_ic_metric_default,
        batches=3,
        batch_size=1,
    )
    assert checked == ["dataset", "model", "metric"]


def test_flags_a_model_that_breaks_the_protocol(fake_od_dataset_default):
    model = FakeODModel(prediction_data=[_five_boxes_four_labels()])
    with pytest.raises(_spotcheck_error()) as caught:
        check_components(task="object_detection", model=model, dataset=fake_od_dataset_default)
    assert any("The model broke" in note for note in caught.value.__notes__)


def test_flags_a_broken_dataset_item(fake_od_dataset_default):
    dataset = FakeODDataset(targets=[_five_boxes_four_labels()] * len(fake_od_dataset_default))
    with pytest.raises(_spotcheck_error()) as caught:
        check_components(task="object_detection", dataset=dataset)
    assert any("The dataset broke" in note for note in caught.value.__notes__)


def test_flags_a_broken_metric(fake_od_model_default, fake_od_dataset_default):
    class ListMetric(FakeODMetric):
        def compute(self):
            return [0.5]

    with pytest.raises(_spotcheck_error()) as caught:
        check_components(
            task="object_detection",
            model=fake_od_model_default,
            dataset=fake_od_dataset_default,
            metric=ListMetric(),
        )
    assert any("The metric broke" in note for note in caught.value.__notes__)


def test_flags_a_broken_augmentation(fake_od_model_default, fake_od_dataset_default):
    with pytest.raises(_spotcheck_error()) as caught:
        check_components(
            task="object_detection",
            model=fake_od_model_default,
            dataset=fake_od_dataset_default,
            augmentation=BrokenAugmentation(),
        )
    assert any("The augmentation broke" in note for note in caught.value.__notes__)


def test_reports_missing_extra(monkeypatch, fake_od_dataset_default):
    real_import = builtins.__import__

    def import_without_spotcheck(name, *args, **kwargs):
        if name == "maite.spotcheck":
            raise ImportError("no beartype")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "maite.spotcheck", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_spotcheck)
    with pytest.raises(ImportError, match=r"checkmaite\[spotcheck\]"):
        check_components(task="object_detection", dataset=fake_od_dataset_default)


def test_unknown_task(fake_od_dataset_default):
    with pytest.raises(KeyError, match="Unknown evaluation task"):
        check_components(task="no_such_task", dataset=fake_od_dataset_default)


def test_task_without_maite_protocols(fake_od_dataset_default):
    task = EvaluationTask(
        name="semantic_segmentation", namespace="example.segmentation", target_adapter=TypeAdapter(object)
    )
    with pytest.raises(ValueError, match="no protocols"):
        check_components(task=task, dataset=fake_od_dataset_default)


def test_needs_something_to_check():
    with pytest.raises(ValueError, match="Nothing to check"):
        check_components(task="object_detection")


def test_needs_a_dataset(fake_od_model_default):
    with pytest.raises(ValueError, match="needs a dataset to run the model"):
        check_components(task="object_detection", model=fake_od_model_default)


def test_rejects_a_short_dataset(fake_od_dataset_default):
    with pytest.raises(ValueError, match="need 100"):
        check_components(task="object_detection", dataset=fake_od_dataset_default, batches=10, batch_size=10)


def test_writes_no_cache(tmp_cache_path, fake_od_model_default, fake_od_dataset_default, fake_od_metric_default):
    check_components(
        task="object_detection",
        model=fake_od_model_default,
        dataset=fake_od_dataset_default,
        metric=fake_od_metric_default,
    )
    assert list(tmp_cache_path.iterdir()) == []


class SeededNoiseAugmentation(IdentityAugmentation):
    """Draws from its own generator on every call, like a random augmentation."""

    def __init__(self) -> None:
        self.rng = torch.Generator().manual_seed(0)

    def __call__(self, batch):
        torch.rand(1, generator=self.rng)
        return batch


class CountingMetric(FakeODMetric):
    def __init__(self) -> None:
        super().__init__()
        self.updates = 0

    def reset(self) -> None:
        self.updates = 0

    def update(self, *args) -> None:
        self.updates += 1


def test_leaves_the_augmentation_and_metric_as_they_were(fake_od_model_default, fake_od_dataset_default):
    augmentation, metric = SeededNoiseAugmentation(), CountingMetric()
    metric.update(None, None, None)

    check_components(
        task="object_detection",
        model=fake_od_model_default,
        dataset=fake_od_dataset_default,
        metric=metric,
        augmentation=augmentation,
    )

    # The augmentation's next draw is still its first, and the metric keeps its earlier update.
    assert torch.equal(
        torch.rand(1, generator=augmentation.rng), torch.rand(1, generator=torch.Generator().manual_seed(0))
    )
    assert metric.updates == 1


def test_checks_a_metric_that_cannot_be_copied_and_resets_it(fake_od_dataset_default):
    class UncopyableMetric(CountingMetric):
        def __deepcopy__(self, memo):
            raise TypeError("holds a device handle")

    metric = UncopyableMetric()
    metric.update(None, None, None)
    with pytest.warns(UserWarning, match="could not copy the metric"):
        check_components(task="object_detection", dataset=fake_od_dataset_default, metric=metric)
    assert metric.updates == 0


class UncopyableMetric(CountingMetric):
    def __deepcopy__(self, memo):
        raise TypeError("holds a device handle")


class UncopyableMetricFailingUpdate(UncopyableMetric):
    def update(self, *args) -> None:
        super().update(*args)
        if self.updates == 2:
            raise RuntimeError("update failed")


class UncopyableMetricBrokenCompute(UncopyableMetric):
    def compute(self):
        return "not a dict"


class UncopyableMetricFailingReset(UncopyableMetricFailingUpdate):
    resets = 0

    def reset(self) -> None:
        self.resets += 1
        if self.resets > 1:  # the reset at the start of the check succeeds
            raise OSError("device gone")
        super().reset()


def test_resets_an_uncopyable_metric_when_an_update_fails(fake_od_dataset_default):
    metric = UncopyableMetricFailingUpdate()
    with pytest.warns(UserWarning, match="could not copy the metric"), pytest.raises(RuntimeError, match="update"):
        check_components(task="object_detection", dataset=fake_od_dataset_default, metric=metric, batches=3)
    assert metric.updates == 0


def test_resets_an_uncopyable_metric_when_compute_breaks_the_protocol(fake_od_dataset_default):
    metric = UncopyableMetricBrokenCompute()
    with pytest.warns(UserWarning, match="could not copy the metric"), pytest.raises(_spotcheck_error()):
        check_components(task="object_detection", dataset=fake_od_dataset_default, metric=metric, batches=2)
    assert metric.updates == 0


def test_keeps_the_check_error_when_the_reset_after_it_fails(fake_od_dataset_default):
    metric = UncopyableMetricFailingReset()
    with (
        pytest.warns(UserWarning, match="could not copy the metric"),
        pytest.raises(RuntimeError, match="update failed") as raised,
    ):
        check_components(task="object_detection", dataset=fake_od_dataset_default, metric=metric, batches=3)
    assert any("could not reset the metric" in note for note in raised.value.__notes__)
