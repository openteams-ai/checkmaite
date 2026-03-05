import dataclasses
import functools
from collections.abc import Callable

import pytest


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
def test_predict(mocker, predict_domain_fixture):
    from maite import tasks as tasks

    from checkmaite import cached_tasks

    actual = tasks.predict(model=predict_domain_fixture.model, dataset=predict_domain_fixture.dataset)
    expected = cached_tasks.predict(
        model=predict_domain_fixture.model, dataset=predict_domain_fixture.dataset, return_augmented_data=False
    )

    predict_domain_fixture.assert_closeness_fn(actual, expected)

    mocker.patch(
        "maite.tasks.predict",
        side_effect=AssertionError("maite.tasks.predict() was called although a cache hit was expected"),
    )
    cached = cached_tasks.predict(model=predict_domain_fixture.model, dataset=predict_domain_fixture.dataset)

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
