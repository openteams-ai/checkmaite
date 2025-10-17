import dataclasses
import functools
from collections.abc import Callable

import pytest
import torch
from maite import tasks as tasks
from maite._internals.protocols.generic import Dataset, Model

from jatic_ri import cached_tasks


@dataclasses.dataclass
class DomainFixture:
    model: Model
    dataset: Dataset
    assert_closeness_fn: Callable


def _assert_predict_close(actual, expected, /, *, assert_target_close_fn, **kwargs):
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
