import pytest
from jatic_ri.image_classification.metrics import TorchICMulticlassMetric, accuracy_multiclass_torch_metric_factory, f1score_multiclass_torch_metric_factory, InvalidMetricTypeError
from maite.protocols import image_classification as ic
from maite.workflows import evaluate
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from tests.fake_ic_classes import FakeICDataset, FakeICModel

@pytest.mark.filterwarnings("ignore:The ``compute`` method of metric MulticlassAccuracy was called before the ``update`` method:UserWarning")
def test_ic_accuracy_defaults() -> None:
    "Tests that a TorchMetric Accuracy wrapper factory initiates with default parameters."

    accuracy_metric: ic.Metric = accuracy_multiclass_torch_metric_factory(num_classes=12)

    assert isinstance(accuracy_metric, TorchICMulticlassMetric)
    # torchmetrics.Accuracy is a wrapper.  initiating with type='multiclass' returns torchmetrics.classification.MulticlassAccuracy object)
    assert isinstance(accuracy_metric._ic_metric, MulticlassAccuracy)
    assert accuracy_metric._ic_metric.num_classes == 12
    # 'micro' calculates accuracy over all classes.  'macro' averages per-class accuracy
    assert accuracy_metric._ic_metric.average == "micro"
    # Assert that the wrapper class compute returns a dictionary with correct return key and underlying metric's initial compute result
    assert accuracy_metric.compute()["accuracy"] == accuracy_metric._ic_metric.compute()

@pytest.mark.filterwarnings("ignore:The ``compute`` method of metric MulticlassF1Score was called before the ``update`` method:UserWarning")
def test_ic_f1score_defaults() -> None:
    "Tests that a TorchMetric Accuracy wrapper factory initiates with default parameters."

    f1score_metric: ic.Metric = f1score_multiclass_torch_metric_factory(num_classes=12)

    assert isinstance(f1score_metric, TorchICMulticlassMetric)
    # torchmetrics.F1Score is a wrapper.  initiating with type='multiclass' returns torchmetrics.classification.MulticlassF1Score object)
    assert isinstance(f1score_metric._ic_metric, MulticlassF1Score)
    assert f1score_metric._ic_metric.num_classes == 12
    # 'micro' calculates accuracy over all classes.  'macro' averages per-class accuracy
    assert f1score_metric._ic_metric.average == "macro"
    # Assert that the wrapper class compute returns a dictionary with correct return key and underlying metric's initial compute result
    assert f1score_metric.compute()["f1_score"] == f1score_metric._ic_metric.compute()

def test_calcualate_accuracy_micro(fake_ic_dataset_default: FakeICDataset, fake_ic_model_default: FakeICModel ) -> None:
    """
    Tests that a default 'micro' accuracy (i.e. not an average of per-class accuracies) returns as expected.
    metric_batch_input_data_ic["predictions"] is a Sequence[Sequence[Tensor(10,)]].  Each item in the first sequence represents one batch ic.TargetBatchType,
    i.e. a Sequence of Tensors, each 1-dim tensor corresponds to one prediction of logits (or pseudo-probs) for each class
    metric_batch_input_data_ic["targets"] has a similar structure and represents ground truth one-hots.

    Evaluates to 0.9 as 18 of 20 predictions are correct in the fake data.
    """
    accuracy_metric: ic.Metric = accuracy_multiclass_torch_metric_factory(num_classes=10)

    results, _, _ = evaluate(dataset=fake_ic_dataset_default, model=fake_ic_model_default, metric=accuracy_metric)
    assert results["accuracy"] == 0.9 # 18 of 20 total predictions in test data are accurate

def test_calcualate_accuracy_macro(fake_ic_dataset_default: FakeICDataset, fake_ic_model_default: FakeICModel ) -> None:
    """
    Tests the non-default 'macro' accuracy (an average of per-class accuracy scores).

    In the test data, 8 of 10 classes were either predicted or target (so other two classes ignored).
    Of those, six classes were all correct, one class was all in correct, and one class was 2 of 3 (.6667) correct.
    So (1 + 1 + 1 + 1 + 1 + 1 + 0 + .66667) / 8 = .833333333...
    """
    accuracy_metric: ic.Metric = accuracy_multiclass_torch_metric_factory(num_classes=10, average='macro')

    results, _, _ = evaluate(dataset=fake_ic_dataset_default, model=fake_ic_model_default, metric=accuracy_metric)
    assert results["accuracy"] == pytest.approx(.8333333333)


def test_calculate_f1_score_macro(fake_ic_dataset_default: FakeICDataset, fake_ic_model_default: FakeICModel ) -> None:
    """
    Tests an F1Score metric with defaults for 'type' and 'average' against the fake data.
    """

    f1score_metric: ic.Metric = f1score_multiclass_torch_metric_factory(num_classes=10)

    results, _, _ = evaluate(dataset=fake_ic_dataset_default, model=fake_ic_model_default, metric=f1score_metric)

    assert results["f1_score"] == pytest.approx(.85)

def test_error_with_non_multiclass_metric() -> None:
    """
    Tests that the wrapper raises an error if the provided TorchMetric is not a multiclass IC metric.
    """
    multilabel_metric = Accuracy(task="multilabel", num_classes=10, num_labels=10)

    with pytest.raises(InvalidMetricTypeError):
        _ = TorchICMulticlassMetric(multilabel_metric,return_key="_")
