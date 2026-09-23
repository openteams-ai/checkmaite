import dataclasses
import functools
from collections.abc import Callable

import pytest


def test_cache_set_is_atomic_when_serialization_fails(tmp_path):
    """A failed replacement cannot leave a corrupt entry that appears valid."""
    from checkmaite.core._cache import Cache

    class TextCache(Cache[str]):
        def path(self, key):
            return tmp_path / key

        def serialize(self, value):
            if value == "invalid":
                raise ValueError("cannot serialize")
            return value.encode()

        def deserialize(self, value):
            return value.decode()

    cache = TextCache()
    cache.set("entry", "valid")
    with pytest.raises(ValueError, match="cannot serialize"):
        cache.set("entry", "invalid")

    assert cache.contains("entry")
    assert cache.get("entry") == "valid"


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


def test_predict_warns_and_skips_lossy_metadata_cache(fake_ic_model_default):
    from pathlib import Path

    from checkmaite import cached_tasks
    from tests.conftest import FakeICDataset

    dataset = FakeICDataset(
        datum_metadata=[{"id": index, "source": Path("image.png")} for index in range(20)],
        dataset_metadata={"id": "lossy-metadata-dataset", "index2label": {}},
    )
    model = CountingModel(fake_ic_model_default, model_id="lossy-metadata-model")

    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.predict(model=model, dataset=dataset, strict_cache_serialization=True)
    calls_after_first_run = model.calls
    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.predict(model=model, dataset=dataset, strict_cache_serialization=True)

    assert model.calls > calls_after_first_run


def test_compatibility_mode_allows_lossy_metadata_cache(fake_ic_model_default):
    from pathlib import Path

    from checkmaite import cached_tasks
    from tests.conftest import FakeICDataset

    dataset = FakeICDataset(
        datum_metadata=[{"id": index, "source": Path("image.png")} for index in range(20)],
        dataset_metadata={"id": "compatibility-metadata-dataset", "index2label": {}},
    )
    model = CountingModel(fake_ic_model_default, model_id="compatibility-metadata-model")

    cached_tasks.predict(model=model, dataset=dataset)
    calls_after_first_run = model.calls
    cached_tasks.predict(model=model, dataset=dataset)

    assert model.calls == calls_after_first_run


@pytest.mark.parametrize("predict_domain_fixture", ["IC"], indirect=True)
def test_strict_mode_reuses_lossless_prediction_cache(predict_domain_fixture):
    from checkmaite import cached_tasks

    model = CountingModel(
        predict_domain_fixture.model,
        model_id=f"{predict_domain_fixture.model.metadata['id']}-strict-cache",
    )
    cached_tasks.predict(
        model=model,
        dataset=predict_domain_fixture.dataset,
        strict_cache_serialization=True,
    )
    calls_after_first_run = model.calls
    cached_tasks.predict(
        model=model,
        dataset=predict_domain_fixture.dataset,
        strict_cache_serialization=True,
    )

    assert model.calls == calls_after_first_run


def test_strict_mode_rejects_unadapted_od_target_classes(fake_od_model_default, fake_od_dataset_default):
    from checkmaite import cached_tasks

    model = CountingModel(fake_od_model_default, model_id="strict-unadapted-od-model")
    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.predict(
            model=model,
            dataset=fake_od_dataset_default,
            strict_cache_serialization=True,
        )
    calls_after_first_run = model.calls
    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.predict(
            model=model,
            dataset=fake_od_dataset_default,
            strict_cache_serialization=True,
        )

    assert model.calls > calls_after_first_run


def test_strict_mode_validates_predictions_and_has_a_distinct_identity(fake_ic_dataset_default):
    import torch
    from torchvision import tv_tensors

    from checkmaite import cached_tasks

    class TVTensorModel:
        metadata = {"id": "strict-tvtensor-model", "index2label": {}}

        def __init__(self):
            self.calls = 0

        def __call__(self, inputs):
            self.calls += 1
            return [tv_tensors.Image(torch.zeros((1, 2, 2))) for _ in inputs]

    model = TVTensorModel()
    cached_tasks.predict(model=model, dataset=fake_ic_dataset_default)
    calls_after_compatibility_run = model.calls

    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.predict(
            model=model,
            dataset=fake_ic_dataset_default,
            strict_cache_serialization=True,
        )

    assert model.calls > calls_after_compatibility_run


def test_strict_mode_skips_lossy_evaluation_results(fake_ic_model_default, fake_ic_dataset_default):
    from pathlib import Path

    from checkmaite import cached_tasks

    class PathMetric:
        metadata = {"id": "strict-path-result"}

        def __init__(self):
            self.computes = 0

        def reset(self):
            pass

        def update(self, predictions, targets, metadata):
            pass

        def compute(self):
            self.computes += 1
            return {"artifact": Path("result.json")}

    model = CountingModel(fake_ic_model_default, model_id="strict-result-model")
    metric = PathMetric()
    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.evaluate(
            model=model,
            dataset=fake_ic_dataset_default,
            metric=metric,
            strict_cache_serialization=True,
        )
    calls_after_first_run = model.calls
    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.evaluate(
            model=model,
            dataset=fake_ic_dataset_default,
            metric=metric,
            strict_cache_serialization=True,
        )

    assert model.calls == calls_after_first_run
    assert metric.computes == 2


@pytest.mark.parametrize("task_name", ["predict", "evaluate"])
@pytest.mark.parametrize("invalid_value", ["none", "targets", "all"])
def test_return_augmented_data_rejects_string_modes(
    task_name, invalid_value, fake_ic_model_default, fake_ic_dataset_default
):
    """CheckMAITE follows MAITE's boolean-only public contract."""
    from checkmaite import cached_tasks

    task = getattr(cached_tasks, task_name)
    with pytest.raises(TypeError, match="must be a bool"):
        task(
            model=fake_ic_model_default,
            dataset=fake_ic_dataset_default,
            return_augmented_data=invalid_value,
            use_cache=False,
        )


def test_prediction_cache_is_reused_by_later_evaluation(fake_ic_model_default, fake_ic_dataset_default):
    """Inference identity does not depend on whether the first caller requested a metric."""
    from checkmaite import cached_tasks
    from tests.conftest import FakeICMetric

    model = CountingModel(fake_ic_model_default, model_id="predict-then-evaluate-model")
    cached_tasks.predict(model=model, dataset=fake_ic_dataset_default, use_cache=True)
    calls_after_prediction = model.calls

    result, _, _ = cached_tasks.evaluate(
        model=model,
        dataset=fake_ic_dataset_default,
        metric=FakeICMetric(metric_metadata={"id": "predict-then-evaluate-metric"}),
        use_cache=True,
    )

    assert result
    assert model.calls == calls_after_prediction


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


def test_falsey_prediction_postprocessor_is_executed(fake_ic_model_default, fake_ic_dataset_default):
    import torch

    from checkmaite import cached_tasks

    class FalseyPostprocessor:
        def __init__(self):
            self.calls = 0

        def __bool__(self):
            return False

        def __call__(self, predictions):
            self.calls += 1
            return [[torch.zeros_like(prediction) for prediction in batch] for batch in predictions]

    class SumMetric:
        metadata = {"id": "falsey-postprocessor-metric"}

        def reset(self):
            self.total = 0.0

        def update(self, predictions, targets, metadata):
            self.total += sum(float(prediction.sum()) for prediction in predictions)

        def compute(self):
            return {"sum": self.total}

    postprocessor = FalseyPostprocessor()
    result, _, _ = cached_tasks.evaluate(
        model=fake_ic_model_default,
        dataset=fake_ic_dataset_default,
        metric=SumMetric(),
        cpu_prediction_postprocessor=postprocessor,
        cpu_prediction_postprocessor_id="falsey",
        use_cache=False,
    )

    assert postprocessor.calls == 1
    assert result == {"sum": 0.0}


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


def test_evaluate_postprocessor_without_cache_retains_targets(fake_ic_model_default, fake_ic_dataset_default):
    """Deferred scoring can use targets without prediction caching."""
    import torch

    from checkmaite import cached_tasks

    class SumMetric:
        metadata = {"id": "uncached-postprocessor-sum"}

        def reset(self):
            self.total = 0.0

        def update(self, preds, targets, metadata):
            self.total += sum(float(prediction.sum()) for prediction in preds)

        def compute(self):
            return {"sum": self.total}

    result, _, _ = cached_tasks.evaluate(
        model=fake_ic_model_default,
        dataset=fake_ic_dataset_default,
        metric=SumMetric(),
        cpu_prediction_postprocessor=lambda predictions: [
            [torch.zeros_like(prediction) for prediction in batch] for batch in predictions
        ],
        use_cache=False,
    )

    assert result == {"sum": 0.0}


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


def test_augmentation_without_id_disables_caching(fake_ic_model_default, fake_ic_dataset_default):
    from checkmaite import cached_tasks

    class Augmentation:
        metadata = {"id": None}

        def __call__(self, batch):
            return batch

    model = CountingModel(fake_ic_model_default, model_id="unidentified-augmentation-model")
    for _ in range(2):
        with pytest.warns(UserWarning, match="augmentation has no metadata ID"):
            cached_tasks.predict(
                model=model,
                dataset=fake_ic_dataset_default,
                augmentation=Augmentation(),
            )

    assert model.calls > len(fake_ic_dataset_default)


def test_dataloader_calls_disable_caching_without_batching_identity(fake_ic_model_default, fake_ic_dataset_default):
    from checkmaite import cached_tasks

    data = [fake_ic_dataset_default[index] for index in range(2)]
    inputs, targets, metadata = zip(*data, strict=True)
    dataloader = [(inputs, targets, metadata)]
    model = CountingModel(fake_ic_model_default, model_id="dataloader-cache-model")

    with pytest.warns(UserWarning, match="batching has no stable identity"):
        cached_tasks.predict(model=model, dataloader=dataloader, dataset_id="dataloader-dataset")
    calls_after_first_run = model.calls
    with pytest.warns(UserWarning, match="batching has no stable identity"):
        cached_tasks.predict(model=model, dataloader=dataloader, dataset_id="dataloader-dataset")

    assert model.calls > calls_after_first_run


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
            metadata_batches=[[{"id": "datum"}]],
            model=fake_ic_model_default,
            dataset=fake_ic_dataset_default,
            inference_id="no-metric-id-inference",
            use_cache=True,
        )


def test_evaluate_from_predictions_cache_warning_without_inference_id(fake_ic_dataset_default):
    """Standalone evaluation caching requires an explicit inference identity."""
    import torch

    from checkmaite import cached_tasks
    from tests.conftest import FakeICMetric

    metric = FakeICMetric(
        calculated_metrics={"result": torch.Tensor([0.5])},
        metric_metadata={"id": "test_metric"},
        return_key="result",
    )

    with pytest.warns(
        UserWarning,
        match="use_cache was requested but caching is disabled because at least one of the following is None:",
    ):
        cached_tasks.evaluate_from_predictions(
            metric=metric,
            predictions=[[torch.tensor([0, 1, 0])]],
            targets=[[torch.tensor([0, 1, 1])]],
            metadata_batches=[[{"id": "datum"}]],
            model_id="standalone-model",
            dataset=fake_ic_dataset_default,
            use_cache=True,
        )


def test_evaluate_from_predictions_requires_metadata_batches(fake_ic_metric_default):
    """Callers must explicitly supply metadata rather than silently scoring without it."""
    from checkmaite import cached_tasks

    with pytest.raises(TypeError, match="metadata_batches"):
        cached_tasks.evaluate_from_predictions(
            metric=fake_ic_metric_default,
            predictions=[[1]],
            targets=[[1]],
            use_cache=False,
        )


@pytest.mark.parametrize(
    ("targets", "metadata_batches", "message"),
    [
        ([], [[{"id": "datum"}]], "same number of batches"),
        ([[1]], [], "same number of batches"),
        ([[]], [[{"id": "datum"}]], "same length"),
        ([[1]], [[]], "same length"),
    ],
)
def test_evaluate_from_predictions_validates_batch_alignment(
    fake_ic_metric_default, targets, metadata_batches, message
):
    """Predictions, targets, and metadata must remain positionally aligned."""
    from checkmaite import cached_tasks

    with pytest.raises(ValueError, match=message):
        cached_tasks.evaluate_from_predictions(
            metric=fake_ic_metric_default,
            predictions=[[1]],
            targets=targets,
            metadata_batches=metadata_batches,
            use_cache=False,
        )


def test_evaluation_cache_distinguishes_inference_identities():
    """Standalone result caches cannot cross explicitly identified prediction sets."""
    import torch

    from checkmaite import cached_tasks

    class Metric:
        metadata = {"id": "metadata-presence"}

        def reset(self):
            self.total = 0

        def update(self, preds, targets, metadata):
            self.total += sum(item.get("weight", 0) for item in metadata)

        def compute(self):
            return {"total": self.total}

    predictions = [[torch.tensor([1.0])]]
    targets = [[torch.tensor([1.0])]]
    first = cached_tasks.evaluate_from_predictions(
        metric=Metric(),
        predictions=predictions,
        targets=targets,
        metadata_batches=[[{"id": "datum", "weight": 7}]],
        inference_id="first-inference",
    )
    second = cached_tasks.evaluate_from_predictions(
        metric=Metric(),
        predictions=predictions,
        targets=targets,
        metadata_batches=[[{"id": "datum", "weight": 3}]],
        inference_id="second-inference",
    )

    assert first == {"total": 7}
    assert second == {"total": 3}


def test_full_augmented_data_does_not_publish_prediction_cache(
    monkeypatch, fake_ic_model_default, fake_ic_dataset_default
):
    """A fresh full-data realization must not replace ordinary cached predictions."""
    from checkmaite import cached_tasks
    from checkmaite.core import cached_tasks as cached_tasks_module

    def reject_cache_write(*args, **kwargs):
        raise AssertionError("full-data requests must not publish prediction artifacts")

    monkeypatch.setattr(cached_tasks_module._predict_cache, "set", reject_cache_write)
    _, augmented_data = cached_tasks.predict(
        model=fake_ic_model_default,
        dataset=fake_ic_dataset_default,
        return_augmented_data=True,
        use_cache=True,
    )

    assert augmented_data


def test_full_augmented_evaluation_does_not_replace_cached_result(fake_ic_dataset_default):
    """Fresh full-data metrics do not overwrite ordinary cached evaluation results."""
    import torch

    from checkmaite import cached_tasks

    class ChangingModel:
        metadata = {"id": "full-evaluation-cache-model", "index2label": {}}

        def __init__(self):
            self.calls = 0

        def __call__(self, inputs):
            self.calls += 1
            return [torch.tensor(float(self.calls)) for _ in inputs]

    class SumMetric:
        metadata = {"id": "full-evaluation-cache-metric"}

        def reset(self):
            self.total = 0.0

        def update(self, predictions, targets, metadata):
            self.total += sum(float(prediction) for prediction in predictions)

        def compute(self):
            return {"sum": self.total}

    model = ChangingModel()
    ordinary, _, _ = cached_tasks.evaluate(
        model=model, dataset=fake_ic_dataset_default, metric=SumMetric(), use_cache=True
    )
    fresh, _, augmented_data = cached_tasks.evaluate(
        model=model,
        dataset=fake_ic_dataset_default,
        metric=SumMetric(),
        return_augmented_data=True,
        use_cache=True,
    )
    calls_after_fresh = model.calls
    cached, _, _ = cached_tasks.evaluate(
        model=model, dataset=fake_ic_dataset_default, metric=SumMetric(), use_cache=True
    )

    assert augmented_data
    assert fresh != ordinary
    assert cached == ordinary
    assert model.calls == calls_after_fresh


def test_cold_evaluate_scores_inside_maite_evaluate(monkeypatch, fake_ic_model_default, fake_ic_dataset_default):
    """A cold evaluation does not split inference from metric evaluation."""
    import maite.tasks

    from checkmaite import cached_tasks
    from tests.conftest import FakeICMetric

    def reject_evaluate_from_predictions(**kwargs):
        raise AssertionError("cold evaluation should score in maite.tasks.evaluate")

    monkeypatch.setattr(maite.tasks, "evaluate_from_predictions", reject_evaluate_from_predictions)

    result, _, _ = cached_tasks.evaluate(
        model=fake_ic_model_default,
        dataset=fake_ic_dataset_default,
        metric=FakeICMetric(),
        use_cache=False,
    )

    assert result


def test_evaluate_allows_different_class_map_names(fake_ic_model_default):
    """Generic evaluation does not require task-specific class-map names."""
    from checkmaite import cached_tasks
    from tests.conftest import FakeICDataset, FakeICMetric

    model = CountingModel(fake_ic_model_default, model_id="class-map-model")
    model.metadata["index2label"] = {0: "cat"}
    dataset = FakeICDataset(dataset_metadata={"id": "class-map-dataset", "index2label": {0: "feline"}})

    result, _, _ = cached_tasks.evaluate(model=model, dataset=dataset, metric=FakeICMetric(), use_cache=False)

    assert result


def test_missing_prediction_artifact_does_not_reuse_old_evaluation(fake_ic_dataset_default):
    """Fresh predictions are never paired with a result from a missing artifact."""
    import torch

    from checkmaite import cached_tasks
    from checkmaite.core import cached_tasks as cached_tasks_module

    class ChangingModel:
        metadata = {"id": "missing-prediction-model", "index2label": {}}

        def __init__(self):
            self.calls = 0

        def __call__(self, inputs):
            self.calls += 1
            return [torch.tensor(float(self.calls)) for _ in inputs]

    class SumMetric:
        metadata = {"id": "missing-prediction-metric"}

        def reset(self):
            self.total = 0.0

        def update(self, predictions, targets, metadata):
            self.total += sum(float(prediction) for prediction in predictions)

        def compute(self):
            return {"sum": self.total}

    model = ChangingModel()
    first, _, _ = cached_tasks.evaluate(
        model=model,
        dataset=fake_ic_dataset_default,
        metric=SumMetric(),
        use_cache=True,
    )
    config = cached_tasks_module._PredictConfig(
        model_id=model.metadata["id"],
        dataset_id=fake_ic_dataset_default.metadata["id"],
        augmentation_id=None,
        batch_size=1,
    )
    assert config.cache_key is not None
    cached_tasks_module._predict_cache.path(config.cache_key).unlink()

    second, predictions, _ = cached_tasks.evaluate(
        model=model,
        dataset=fake_ic_dataset_default,
        metric=SumMetric(),
        return_preds=True,
        use_cache=True,
    )

    assert predictions
    assert second != first


def test_cached_predictions_preserve_metadata_for_later_metrics(fake_ic_model_default):
    """A second metric receives cacheable metadata while reusing inference."""
    import numpy as np

    from checkmaite import cached_tasks
    from tests.conftest import FakeICDataset

    datum_metadata = [
        {
            "id": index,
            "weight": index + 1,
            "source": f"camera-{index}.jpg",
            "region": [index, index + 1],
            "attributes": {"camera": "left"},
            "mask": np.asarray([True, False]),
        }
        for index in range(20)
    ]
    dataset = FakeICDataset(
        datum_metadata=datum_metadata,
        dataset_metadata={"id": "metadata-cache-dataset", "index2label": {}},
    )
    model = CountingModel(fake_ic_model_default, model_id="metadata-cache-model")

    class MetadataMetric:
        def __init__(self, metric_id):
            self.metadata = {"id": metric_id}

        def reset(self):
            self.total = 0

        def update(self, preds, targets, metadata):
            assert all(isinstance(item["source"], str) for item in metadata)
            assert all(isinstance(item["region"], list) for item in metadata)
            assert all(item["attributes"] == {"camera": "left"} for item in metadata)
            assert all(isinstance(item["mask"], np.ndarray) for item in metadata)
            self.total += sum(item["weight"] for item in metadata)

        def compute(self):
            return {"total": self.total}

    first, _, _ = cached_tasks.evaluate(
        model=model,
        dataset=dataset,
        metric=MetadataMetric("metadata-first"),
        use_cache=True,
    )
    calls_after_first = model.calls
    second, _, _ = cached_tasks.evaluate(
        model=model,
        dataset=dataset,
        metric=MetadataMetric("metadata-second"),
        use_cache=True,
    )

    assert model.calls == calls_after_first
    assert first == {"total": 210}
    assert second == first


def test_prediction_cache_escapes_binary_reference_like_strings(fake_ic_model_default):
    """User strings cannot be confused with generated binary references."""
    from checkmaite import cached_tasks
    from tests.conftest import FakeICDataset

    dataset = FakeICDataset(
        datum_metadata=[
            {"id": index, "source": "binary+numpy://00000000-0000-0000-0000-000000000000"} for index in range(20)
        ],
        dataset_metadata={"id": "reference-like-string-dataset", "index2label": {}},
    )
    model = CountingModel(fake_ic_model_default, model_id="reference-like-string-model")

    cached_tasks.predict(model=model, dataset=dataset, use_cache=True)
    calls_after_first = model.calls
    cached_tasks.predict(model=model, dataset=dataset, use_cache=True)

    assert model.calls == calls_after_first


def test_prediction_cache_disables_for_unsupported_metadata_types(fake_ic_model_default):
    """Unsupported metadata disables caching instead of changing on a cache hit."""
    from checkmaite import cached_tasks
    from tests.conftest import FakeICDataset

    class UnsupportedMetadataValue:
        pass

    dataset = FakeICDataset(
        datum_metadata=[{"id": index, "custom": UnsupportedMetadataValue()} for index in range(20)],
        dataset_metadata={"id": "unsupported-metadata-dataset", "index2label": {}},
    )

    model = CountingModel(fake_ic_model_default, model_id="unsupported-metadata-model")
    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.predict(model=model, dataset=dataset, use_cache=True)
    calls_after_first = model.calls
    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.predict(model=model, dataset=dataset, use_cache=True)

    assert model.calls > calls_after_first


def test_prediction_cache_disables_for_cyclic_metadata(fake_ic_model_default):
    """Cyclic metadata warns and continues evaluation without publication."""
    from checkmaite import cached_tasks
    from tests.conftest import FakeICDataset

    cyclic = []
    cyclic.append(cyclic)
    dataset = FakeICDataset(
        datum_metadata=[{"id": index, "cyclic": cyclic} for index in range(20)],
        dataset_metadata={"id": "cyclic-metadata-dataset", "index2label": {}},
    )
    model = CountingModel(fake_ic_model_default, model_id="cyclic-metadata-model")

    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.predict(model=model, dataset=dataset, use_cache=True)
    calls_after_first = model.calls
    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.predict(model=model, dataset=dataset, use_cache=True)

    assert model.calls > calls_after_first


def test_cold_evaluate_validates_prediction_target_alignment():
    """Cold and cache-backed evaluations enforce the same batch contract."""
    import numpy as np

    from checkmaite import cached_tasks

    class Dataset:
        metadata = {"id": "invalid-batch-dataset", "index2label": {}}

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return np.zeros((1, 2, 2)), np.asarray([1.0]), {"id": index}

    class Model:
        metadata = {"id": "invalid-batch-model", "index2label": {}}

        def __call__(self, inputs):
            return []

    class Metric:
        metadata = {"id": "invalid-batch-metric"}

        def reset(self):
            pass

        def update(self, preds, targets, metadata):
            pass

        def compute(self):
            return {}

    with pytest.raises(ValueError, match="prediction and target batches"):
        cached_tasks.evaluate(model=Model(), dataset=Dataset(), metric=Metric(), use_cache=True)


def test_predict_allows_empty_data():
    """Prediction keeps MAITE's empty-dataset behavior."""
    from checkmaite import cached_tasks

    class EmptyDataset:
        metadata = {"id": "empty-predict-dataset", "index2label": {}}

        def __len__(self):
            return 0

        def __getitem__(self, index):
            raise IndexError(index)

    class Model:
        metadata = {"id": "empty-predict-model", "index2label": {}}

        def __call__(self, inputs):
            return []

    predictions, augmented_data = cached_tasks.predict(
        model=Model(),
        dataset=EmptyDataset(),
        return_augmented_data=False,
        use_cache=False,
    )

    assert predictions == []
    assert augmented_data == []


def test_cold_evaluate_rejects_empty_data():
    """Cold evaluation matches evaluate-from-predictions empty-input semantics."""
    from checkmaite import cached_tasks

    class EmptyDataset:
        metadata = {"id": "empty-dataset", "index2label": {}}

        def __len__(self):
            return 0

        def __getitem__(self, index):
            raise IndexError(index)

    class Model:
        metadata = {"id": "empty-model", "index2label": {}}

        def __call__(self, inputs):
            return []

    class Metric:
        metadata = {"id": "empty-metric"}

        def reset(self):
            pass

        def update(self, preds, targets, metadata):
            pass

        def compute(self):
            return {}

    with pytest.raises(ValueError, match="at least one element"):
        cached_tasks.evaluate(model=Model(), dataset=EmptyDataset(), metric=Metric(), use_cache=True)


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
        metadata_batches=[[{"id": "datum"}]],
        model=fake_ic_model_default,
        dataset=dataset_no_id,
        use_cache=False,
    )
    assert len(recwarn) == 0


def _native_yolo_cache_fixture(*, dataset_id: str, model_id: str):
    from pathlib import Path

    from maite.protocols import ModelMetadata

    from checkmaite.core.object_detection.dataset_loaders import load_yolo_detection_dataset
    from tests.conftest import FakeODModel

    dataset_root = Path(__file__).parents[1] / "data_for_tests" / "yolo_dataset"
    dataset = load_yolo_detection_dataset(dataset_root / "dataset.yaml", dataset_id=dataset_id)
    model = CountingModel(
        FakeODModel(model_metadata=ModelMetadata(id="fake_od_model", index2label=dataset.metadata["index2label"])),
        model_id=model_id,
    )
    return dataset, model


class _RecordingMetric:
    """Keeps every datum-metadata mapping it is shown, so a cache hit can be compared with a miss."""

    def __init__(self, metric_id: str):
        self.metadata = {"id": metric_id}
        self.seen: list[dict] = []

    def reset(self):
        self.seen = []

    def update(self, preds, targets, metadata):
        self.seen.extend(dict(item) for item in metadata)

    def compute(self):
        return {"count": len(self.seen)}


def test_cached_predictions_preserve_native_dataset_datum_metadata():
    """A metric run from cached predictions sees the same datum metadata as the fresh run.

    datamaite-native datasets carry real datum metadata — COCO ``images[]``
    extras, per-box VisDrone factors, YOLO provenance — and dataeval reads it as
    bias factors. If the cache round trip dropped it, the first metric would see
    rich metadata and every later metric only ``{"id": ...}``.
    """
    from checkmaite.core import cached_tasks

    dataset, model = _native_yolo_cache_fixture(dataset_id="native-yolo-metadata", model_id="fake-od-native-metadata")
    fresh_metric = _RecordingMetric("native-metadata-fresh")
    cached_metric = _RecordingMetric("native-metadata-cached")

    cached_tasks.evaluate(model=model, dataset=dataset, metric=fresh_metric)
    calls_after_miss = model.calls
    cached_tasks.evaluate(model=model, dataset=dataset, metric=cached_metric)

    assert model.calls == calls_after_miss, "second evaluation should reuse the cached predictions"
    assert len(fresh_metric.seen) == len(cached_metric.seen) == len(dataset)

    for fresh_metadata, cached_metadata in zip(fresh_metric.seen, cached_metric.seen, strict=True):
        assert {"yolo_bbox", "source_line", "label_file"} <= set(fresh_metadata)
        assert set(cached_metadata) == set(fresh_metadata)
        assert cached_metadata["id"] == fresh_metadata["id"]
        assert cached_metadata["source_line"] == fresh_metadata["source_line"]
        # The cache is JSON-backed, so datamaite's per-box tuples read back as
        # lists. That normalization is part of the contract, not an accident.
        assert cached_metadata["yolo_bbox"] == [list(box) for box in fresh_metadata["yolo_bbox"]]


def test_strict_cache_mode_declines_native_yolo_datasets():
    """Strict (lossless) caching warns and re-runs inference for native YOLO datasets.

    Two independent reasons, both by design of the strict policy: datamaite's
    ``ObjectDetectionTarget`` is not a registered strict cache type (no OD target
    class is), and ``yolo_bbox`` provenance is a list of tuples, which JSON cannot
    round-trip losslessly. Compatibility mode, the default, caches both.
    """
    from checkmaite.core import cached_tasks
    from checkmaite.core._cached_metadata import validate_metadata_value

    dataset, model = _native_yolo_cache_fixture(dataset_id="native-yolo-strict", model_id="fake-od-native-strict")

    datum_metadata = dict(dataset[0][2])
    with pytest.raises(TypeError, match=r"yolo_bbox\[0\]"):
        validate_metadata_value(datum_metadata, strict=True)
    validate_metadata_value({key: value for key, value in datum_metadata.items() if key != "yolo_bbox"}, strict=True)

    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.predict(model=model, dataset=dataset, strict_cache_serialization=True)
    calls_after_first_run = model.calls
    with pytest.warns(UserWarning, match="Cache publication is disabled"):
        cached_tasks.predict(model=model, dataset=dataset, strict_cache_serialization=True)

    assert model.calls > calls_after_first_run
