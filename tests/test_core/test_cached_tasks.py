import dataclasses
import functools
from collections.abc import Callable

import pytest


class CountingModel:
    """Protocol-compatible model wrapper that records real inference calls."""

    def __init__(self, model, *, model_id: str):
        self._model = model
        self.metadata = {**model.metadata, "id": model_id}
        self.calls = 0
        self.batch_sizes = []

    def __call__(self, inputs):
        self.calls += 1
        self.batch_sizes.append(len(inputs))
        return self._model(inputs)


def _assert_predict_close(actual, expected, /, *, assert_target_close_fn, **kwargs):
    import torch

    actual_predictions, actual_augmented_data = actual
    expected_predictions, expected_augmented_data = expected

    for ab, eb in zip(actual_predictions, expected_predictions, strict=True):
        for a, e in zip(ab, eb, strict=True):
            assert_target_close_fn(a, e, **kwargs)

    for (aib, atb, amb), (eib, etb, emb) in zip(actual_augmented_data, expected_augmented_data, strict=True):
        torch.testing.assert_close(aib, eib, **kwargs)

        for a, e in zip(atb, etb, strict=True):
            assert_target_close_fn(a, e, **kwargs)

        for a, e in zip(amb, emb, strict=True):
            assert a == e


@pytest.fixture
def predict_domain_fixture(
    request, fake_ic_model_default, fake_ic_dataset_default, fake_od_model_default, fake_od_dataset_default
):
    import torch
    from maite._internals.protocols.generic import Dataset, Model

    @dataclasses.dataclass
    class DomainFixture:
        model: Model
        dataset: Dataset
        assert_closeness_fn: Callable

    if request.param == "IC":
        return DomainFixture(
            model=fake_ic_model_default,
            dataset=fake_ic_dataset_default,
            assert_closeness_fn=functools.partial(
                _assert_predict_close, assert_target_close_fn=torch.testing.assert_close
            ),
        )
    if request.param == "OD":

        def assert_target_close_fn(actual, expected, **kwargs):
            torch.testing.assert_close(actual.boxes, expected.boxes, **kwargs)
            torch.testing.assert_close(actual.labels, expected.labels, **kwargs)
            torch.testing.assert_close(actual.scores, expected.scores, **kwargs)

        return DomainFixture(
            model=fake_od_model_default,
            dataset=fake_od_dataset_default,
            assert_closeness_fn=functools.partial(_assert_predict_close, assert_target_close_fn=assert_target_close_fn),
        )
    raise ValueError(f"No fixture available for domain {request.param!r}")


@pytest.mark.parametrize("predict_domain_fixture", ["IC", "OD"], indirect=True)
def test_predict(predict_domain_fixture):
    from maite import tasks as tasks

    from checkmaite import cached_tasks

    model = CountingModel(
        predict_domain_fixture.model,
        model_id=f"{predict_domain_fixture.model.metadata['id']}-counting-predict",
    )
    actual = tasks.predict(model=model, dataset=predict_domain_fixture.dataset)
    expected = cached_tasks.predict(model=model, dataset=predict_domain_fixture.dataset, return_augmented_data=False)

    predict_domain_fixture.assert_closeness_fn(actual, expected)

    calls_before_cache_hit = model.calls
    cached = cached_tasks.predict(model=model, dataset=predict_domain_fixture.dataset)

    assert model.calls == calls_before_cache_hit
    predict_domain_fixture.assert_closeness_fn(cached, actual, atol=0, rtol=0)


def test_evaluate_cache_respects_different_metrics(fake_ic_model_default, fake_ic_dataset_default):
    """
    Test that confirms that cached_tasks.evaluate() returns different results for different metrics, even when model
    and dataset are the same.  A failure of this test may indicate that the "predict cache" hit (i.e. same dataset and
    model) is incorrectly leading to an "evaluate cache" hit even when the metric is different.
    """
    import torch

    from checkmaite import cached_tasks
    from tests.conftest import FakeICMetric

    # Create two different fake metrics with different return values
    metric_1 = FakeICMetric(
        calculated_metrics={"metric_1_result": torch.Tensor([0.25])},
        metric_metadata={"id": "fake_metric_1"},
        return_key="metric_1_result",
    )

    metric_2 = FakeICMetric(
        calculated_metrics={"metric_2_result": torch.Tensor([0.75])},
        metric_metadata={"id": "fake_metric_2"},
        return_key="metric_2_result",
    )

    # Run evaluate with the first metric
    results_1, _, _ = cached_tasks.evaluate(
        model=fake_ic_model_default,
        dataset=fake_ic_dataset_default,
        metric=metric_1,
        use_cache=True,
    )

    # Run evaluate with the second metric (same model and dataset)
    results_2, _, _ = cached_tasks.evaluate(
        model=fake_ic_model_default,
        dataset=fake_ic_dataset_default,
        metric=metric_2,
        use_cache=True,
    )

    # The results should be different since we're using different metrics
    # This assertion will FAIL due to the cache bug - metric_2 will incorrectly
    # return the cached results from metric_1
    assert results_1 != results_2, (
        f"Expected different results for different metrics, but got:\n"
        f"  metric_1 results: {results_1}\n"
        f"  metric_2 results: {results_2}\n"
        f"This indicates a cache hit occurred when it shouldn't have."
    )

    # Verify the actual values are what we expect
    assert "metric_1_result" in results_1
    assert results_1["metric_1_result"].item() == 0.25

    assert "metric_2_result" in results_2
    assert results_2["metric_2_result"].item() == 0.75


def test_evaluate_cache_respects_cpu_prediction_postprocessor(fake_ic_model_default, fake_ic_dataset_default):
    """Raw predictions are reused while postprocessed evaluations have distinct cache keys."""
    import torch

    from checkmaite import cached_tasks

    class SumMetric:
        metadata = {"id": "sum"}

        def reset(self):
            self.total = 0.0

        def update(self, preds, targets, metadatas):
            self.total += sum(float(pred.sum()) for pred in preds)

        def compute(self):
            return {"sum": self.total}

    def zero_predictions(predictions):
        return [[torch.zeros_like(prediction) for prediction in batch] for batch in predictions]

    model = CountingModel(fake_ic_model_default, model_id="postprocessor-cache-model")
    zero_result, _, _ = cached_tasks.evaluate(
        model=model,
        dataset=fake_ic_dataset_default,
        metric=SumMetric(),
        cpu_prediction_postprocessor=zero_predictions,
        cpu_prediction_postprocessor_id="zero",
        use_cache=True,
    )

    calls_after_first_evaluation = model.calls
    raw_result, _, _ = cached_tasks.evaluate(
        model=model,
        dataset=fake_ic_dataset_default,
        metric=SumMetric(),
        cpu_prediction_postprocessor=lambda predictions: predictions,
        cpu_prediction_postprocessor_id="identity",
        use_cache=True,
    )

    assert model.calls == calls_after_first_evaluation
    assert zero_result == {"sum": 0.0}
    assert raw_result["sum"] != 0


def test_predict_cache_respects_batch_size(fake_od_model_default, fake_od_dataset_default):
    """Changing batch size must miss the prediction cache and use the requested batching."""
    from checkmaite import cached_tasks

    model = CountingModel(fake_od_model_default, model_id="batch-size-cache-model")
    cached_tasks.predict(model=model, dataset=fake_od_dataset_default, batch_size=2, use_cache=True)

    model.batch_sizes.clear()
    cached_tasks.predict(model=model, dataset=fake_od_dataset_default, batch_size=4, use_cache=True)

    assert model.batch_sizes == [4, 2]


def test_evaluation_cache_respects_prediction_batch_size(fake_ic_dataset_default):
    """Evaluation results must follow the batch-specific prediction identity."""
    import torch

    from checkmaite import cached_tasks

    class BatchSensitiveModel:
        metadata = {"id": "batch-sensitive-evaluation-model"}

        def __init__(self):
            self.batch_sizes = []

        def __call__(self, inputs):
            batch_size = len(inputs)
            self.batch_sizes.append(batch_size)
            return [torch.tensor(float(batch_size)) for _ in inputs]

    class SumMetric:
        metadata = {"id": "batch-sensitive-sum-metric"}

        def reset(self):
            self.total = 0.0

        def update(self, preds, targets, metadatas):
            self.total += sum(float(prediction) for prediction in preds)

        def compute(self):
            return {"sum": self.total}

    model = BatchSensitiveModel()
    batch_one_result, _, _ = cached_tasks.evaluate(
        model=model,
        dataset=fake_ic_dataset_default,
        metric=SumMetric(),
        batch_size=1,
        use_cache=True,
    )
    calls_after_batch_one = len(model.batch_sizes)
    batch_four_result, _, _ = cached_tasks.evaluate(
        model=model,
        dataset=fake_ic_dataset_default,
        metric=SumMetric(),
        batch_size=4,
        use_cache=True,
    )

    assert len(model.batch_sizes) > calls_after_batch_one
    assert batch_four_result["sum"] == 4 * batch_one_result["sum"]


def test_prediction_cache_distinguishes_model_wrapper_ids(fake_ic_dataset_default):
    """Wrapper postprocessing semantics can be isolated through model metadata IDs."""
    import torch

    from checkmaite import cached_tasks

    class ConstantModel:
        def __init__(self, model_id, value):
            self.metadata = {"id": model_id}
            self.value = value
            self.calls = 0

        def __call__(self, inputs):
            self.calls += 1
            return [torch.tensor(self.value) for _ in inputs]

    first_model = ConstantModel("wrapper-postprocessor-a", 1.0)
    second_model = ConstantModel("wrapper-postprocessor-b", 2.0)
    first_predictions, _ = cached_tasks.predict(model=first_model, dataset=fake_ic_dataset_default, use_cache=True)
    second_predictions, _ = cached_tasks.predict(model=second_model, dataset=fake_ic_dataset_default, use_cache=True)

    assert first_model.calls > 0
    assert second_model.calls > 0
    assert float(first_predictions[0][0]) == 1.0
    assert float(second_predictions[0][0]) == 2.0


def test_evaluate_rejects_cpu_prediction_postprocessor_id_without_postprocessor(
    fake_ic_model_default, fake_ic_dataset_default
):
    from checkmaite import cached_tasks
    from tests.conftest import FakeICMetric

    with pytest.raises(
        ValueError, match="cpu_prediction_postprocessor_id was provided without a cpu_prediction_postprocessor"
    ):
        cached_tasks.evaluate(
            model=fake_ic_model_default,
            dataset=fake_ic_dataset_default,
            metric=FakeICMetric(),
            cpu_prediction_postprocessor_id="orphan-id",
            use_cache=True,
        )


def test_evaluate_postprocessor_without_id_warns_and_preserves_prediction_cache(
    fake_ic_model_default, fake_ic_dataset_default
):
    from checkmaite import cached_tasks

    update_calls = []

    class CountingMetric:
        metadata = {"id": "counting-postprocessor-without-id"}

        def reset(self):
            pass

        def update(self, preds, targets, metadatas):
            update_calls.append(len(preds))

        def compute(self):
            return {"updates": len(update_calls)}

    model = CountingModel(fake_ic_model_default, model_id="postprocessor-without-id-model")
    with pytest.warns(UserWarning, match="evaluation-result caching is disabled"):
        first_result, _, _ = cached_tasks.evaluate(
            model=model,
            dataset=fake_ic_dataset_default,
            metric=CountingMetric(),
            cpu_prediction_postprocessor=lambda predictions: predictions,
            use_cache=True,
        )

    calls_after_first_evaluation = model.calls
    with pytest.warns(UserWarning, match="evaluation-result caching is disabled"):
        second_result, _, _ = cached_tasks.evaluate(
            model=model,
            dataset=fake_ic_dataset_default,
            metric=CountingMetric(),
            cpu_prediction_postprocessor=lambda predictions: predictions,
            use_cache=True,
        )

    assert model.calls == calls_after_first_evaluation
    assert first_result["updates"] > 0
    assert second_result["updates"] == 2 * first_result["updates"]


def test_predict_cache_warning_no_model_id(fake_ic_dataset_default):
    """Test that a warning is emitted when predict() is called with use_cache=True but model has no ID."""
    from checkmaite import cached_tasks
    from tests.conftest import FakeICModel

    # Create a model without an ID
    model_no_id = FakeICModel(model_metadata={"index2label": {}, "id": None})

    with pytest.warns(
        UserWarning,
        match="use_cache was requested but caching is disabled because at least one of the following is None:",
    ):
        cached_tasks.predict(
            model=model_no_id,
            dataset=fake_ic_dataset_default,
            use_cache=True,
        )


def test_predict_cache_warning_no_dataset_id(fake_ic_model_default):
    """Test that a warning is emitted when predict() is called with use_cache=True but dataset has no ID."""
    from checkmaite import cached_tasks
    from tests.conftest import FakeICDataset

    # Create a dataset without an ID
    dataset_no_id = FakeICDataset(dataset_metadata={"index2label": {}, "id": None})

    with pytest.warns(
        UserWarning,
        match="use_cache was requested but caching is disabled because at least one of the following is None:",
    ):
        cached_tasks.predict(
            model=fake_ic_model_default,
            dataset=dataset_no_id,
            use_cache=True,
        )


def test_predict_no_cache_warning_when_use_cache_false(recwarn, fake_ic_model_default, fake_ic_dataset_default):
    """Test that no warning is emitted when predict() is called with use_cache=False."""
    from checkmaite import cached_tasks
    from tests.conftest import FakeICDataset

    # Create a dataset without an ID
    dataset_no_id = FakeICDataset(dataset_metadata={"index2label": {}, "id": None})

    cached_tasks.predict(
        model=fake_ic_model_default,
        dataset=dataset_no_id,
        use_cache=False,
    )
    assert len(recwarn) == 0


def test_evaluate_from_predictions_cache_warning_no_metric_id(fake_ic_model_default, fake_ic_dataset_default):
    """Test that a warning is emitted when evaluate_from_predictions() is called with use_cache=True but metric has no ID."""
    import torch

    from checkmaite import cached_tasks
    from tests.conftest import FakeICMetric

    # Create a metric without an ID
    metric_no_id = FakeICMetric(
        calculated_metrics={"result": torch.Tensor([0.5])},
        metric_metadata={"id": None},
        return_key="result",
    )

    predictions = [[torch.tensor([0, 1, 0])]]
    targets = [[torch.tensor([0, 1, 1])]]

    with pytest.warns(
        UserWarning,
        match="use_cache was requested but caching is disabled because at least one of the following is None:",
    ):
        cached_tasks.evaluate_from_predictions(
            metric=metric_no_id,
            predictions=predictions,
            targets=targets,
            model=fake_ic_model_default,
            dataset=fake_ic_dataset_default,
            use_cache=True,
        )


def test_evaluate_from_predictions_cache_warning_no_model_id(fake_ic_dataset_default):
    """Test that a warning is emitted when evaluate_from_predictions() is called with use_cache=True but model has no ID."""
    import torch

    from checkmaite import cached_tasks
    from tests.conftest import FakeICMetric, FakeICModel

    metric = FakeICMetric(
        calculated_metrics={"result": torch.Tensor([0.5])},
        metric_metadata={"id": "test_metric"},
        return_key="result",
    )

    model_no_id = FakeICModel(model_metadata={"index2label": {}, "id": None})

    predictions = [[torch.tensor([0, 1, 0])]]
    targets = [[torch.tensor([0, 1, 1])]]

    with pytest.warns(
        UserWarning,
        match="use_cache was requested but caching is disabled because at least one of the following is None:",
    ):
        cached_tasks.evaluate_from_predictions(
            metric=metric,
            predictions=predictions,
            targets=targets,
            model=model_no_id,
            dataset=fake_ic_dataset_default,
            use_cache=True,
        )


def test_evaluate_from_predictions_no_cache_warning_when_use_cache_false(recwarn, fake_ic_model_default):
    """Test that no warning is emitted when evaluate_from_predictions() is called with use_cache=False, even with a metric that has no ID."""

    import torch

    from checkmaite import cached_tasks
    from tests.conftest import FakeICDataset, FakeICMetric

    metric_no_id = FakeICMetric(
        calculated_metrics={"result": torch.Tensor([0.5])},
        metric_metadata={"id": None},  # No ID
        return_key="result",
    )

    dataset_no_id = FakeICDataset(dataset_metadata={"index2label": {}, "id": None})

    predictions = [[torch.tensor([0, 1, 0])]]
    targets = [[torch.tensor([0, 1, 1])]]

    cached_tasks.evaluate_from_predictions(
        metric=metric_no_id,
        predictions=predictions,
        targets=targets,
        model=fake_ic_model_default,
        dataset=dataset_no_id,
        use_cache=False,
    )
    assert len(recwarn) == 0
