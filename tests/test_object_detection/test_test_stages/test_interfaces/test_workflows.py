from jatic_ri.object_detection.test_stages.interfaces.test_workflows import SingleDataset, SingleModelDatasetMetricThreshold, TwoDataset, MultiModelSingleDataset, DatasetMetricThreshold, SingleModelSingleDataset
from typing import Any


def test_model_dataset_metric_threshold(dummy_model, dummy_dataset, dummy_metric) -> None:

    class TestImpl(SingleModelDatasetMetricThreshold):
        """Dummy implementation class"""
        def run(self, use_cache: bool = True) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()
    stage = TestImpl()

    stage.load_model(
        model=dummy_model,
        model_id="dummy1",
        )

    stage.load_dataset(
        dataset=dummy_dataset,
        dataset_id="dummy1",
    )

    stage.load_metric(
        metric=dummy_metric,
        metric_id = 'dummy1',
        )

    dummy_threshold = 99.99
    stage.load_threshold(threshold = dummy_threshold)

def test_single_dataset(dummy_dataset) -> None:

    class TestImpl(SingleDataset):
        """Dummy implementation class"""
        def run(self, use_cache: bool = True) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()
    stage = TestImpl()

    stage.load_dataset(
        dataset=dummy_dataset,
        dataset_id="dummy1",
    )

def test_two_dataset(dummy_dataset) -> None:

    class TestImpl(TwoDataset):
        """Dummy implementation class"""
        def run(self, use_cache: bool = True) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()
    stage = TestImpl()

    stage.load_datasets(
        dataset_1=dummy_dataset,
        dataset_1_id="dummy1",
        dataset_2=dummy_dataset,
        dataset_2_id="dummy2",
    )

def test_single_model_dataset(dummy_model, dummy_dataset) -> None:

    class TestImpl(SingleModelSingleDataset):
        """Dummy implementation class"""
        def run(self, use_cache: bool = True) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()
    stage = TestImpl()

    stage.load_model(
        model=dummy_model,
        model_id="dummy1",
        )

    stage.load_dataset(
        dataset=dummy_dataset,
        dataset_id="dummy1",
    )


def test_multi_model_single_dataset(dummy_model, dummy_dataset) -> None:

    class TestImpl(MultiModelSingleDataset):
        """Dummy implementation class"""
        def run(self, use_cache: bool = True) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()
    stage = TestImpl()

    models = {
        "dummy1": dummy_model,
        "dummy2": dummy_model,
        }
    stage.load_models(models=models)

    stage.load_dataset(
        dataset=dummy_dataset,
        dataset_id="dummy1",
    )

def test_dataset_metric_threshold(dummy_dataset, dummy_metric) -> None:

    class TestImpl(DatasetMetricThreshold):
        """Dummy implementation class"""
        def run(self, use_cache: bool = True) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()
    stage = TestImpl()

    stage.load_dataset(
        dataset=dummy_dataset,
        dataset_id="dummy1",
    )

    stage.load_metric(
        metric=dummy_metric,
        metric_id = 'dummy1',
        )

    dummy_threshold = 99.99
    stage.load_threshold(threshold = dummy_threshold)
