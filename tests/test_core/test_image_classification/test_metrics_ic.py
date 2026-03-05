import pytest
from maite.tasks import evaluate

from checkmaite.core.image_classification.metrics import (
    InvalidMetricTypeError,
    TorchICMulticlassMetric,
    accuracy_multiclass_torch_metric_factory,
    f1score_multiclass_torch_metric_factory,
)


@pytest.mark.filterwarnings(
    "ignore:The ``compute`` method of metric MulticlassAccuracy was called before the ``update`` method:UserWarning"
)
def test_ic_accuracy_defaults() -> None:
    accuracy_metric = accuracy_multiclass_torch_metric_factory(num_classes=12, average="micro")

    assert accuracy_metric._ic_metric.num_classes == 12

    assert accuracy_metric._ic_metric.average == "micro"

    # Assert that the wrapper class compute returns a dictionary with correct return key and underlying metric's initial compute result
    assert accuracy_metric.compute()["accuracy"] == accuracy_metric._ic_metric.compute()


@pytest.mark.filterwarnings(
    "ignore:The ``compute`` method of metric MulticlassF1Score was called before the ``update`` method:UserWarning"
)
def test_ic_f1score_defaults() -> None:
    f1score_metric = f1score_multiclass_torch_metric_factory(num_classes=12, average="macro")

    assert f1score_metric._ic_metric.num_classes == 12

    assert f1score_metric._ic_metric.average == "macro"

    # Assert that the wrapper class compute returns a dictionary with correct return key and underlying metric's initial compute result
    assert f1score_metric.compute()["f1_score"] == f1score_metric._ic_metric.compute()


def test_calculate_accuracy_micro(fake_ic_dataset_default, fake_ic_model_default):
    accuracy_metric = accuracy_multiclass_torch_metric_factory(num_classes=10, average="micro")

    results, _, _ = evaluate(dataset=fake_ic_dataset_default, model=fake_ic_model_default, metric=accuracy_metric)

    assert results["accuracy"] == 0.9  # 18 of 20 total predictions in test data are accurate


def test_calculate_f1_score_macro(fake_ic_dataset_default, fake_ic_model_default):
    f1score_metric = f1score_multiclass_torch_metric_factory(num_classes=10, average="macro")

    results, _, _ = evaluate(dataset=fake_ic_dataset_default, model=fake_ic_model_default, metric=f1score_metric)

    assert results["f1_score"] == pytest.approx(0.85)


def test_error_with_non_multiclass_metric():
    from torchmetrics import Accuracy

    multilabel_metric = Accuracy(task="multilabel", num_classes=10, num_labels=10)

    with pytest.raises(InvalidMetricTypeError):
        _ = TorchICMulticlassMetric(multilabel_metric, return_key="_", metric_id="test_multilabel_metric")
