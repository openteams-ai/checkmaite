from jatic_ri.object_detection.test_stages.interfaces.test_workflows import SingleDataset, ModelDatasetMetricThreshold
from typing import Any


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

def test_model_dataset_metric_threshold(dummy_dataset) -> None:

    class TestImpl(ModelDatasetMetricThreshold):
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
