"""Test baseline evalutation"""

import maite.protocols.object_detection as od

from jatic_ri.object_detection.test_stages.impls.baseline_evaluation import (
    BaselineEvaluation,
)


def test_baseline_evaluation() -> None:
    """Test BaselineEvaluation implementation"""

    # fake model class
    class FakeModel(od.Model):
        def __call__(self) -> None:
            return None

    model = FakeModel()

    # fake dataset class
    class FakeDataset(od.Dataset):
        def __len__(self) -> None:
            pass

        def __getitem__(self) -> None:
            pass

    dataset = FakeDataset()

    # fake metric class
    class FakeMetric(od.Metric):
        def update(self) -> None:
            pass

        def compute(self) -> None:
            pass

        def reset(self) -> None:
            pass

    metric = FakeMetric()

    test = BaselineEvaluation()
    # load the maite compliant model
    test.load_model(model=model, model_id="fake_model_1")
    test.load_metric(metric=metric, metric_id="fake_metric_1")
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dataset, dataset_id="fake_dataset_1")
    test.run()
    test.collect_report_consumables()
