from typing import Any

from jatic_ri._common.test_stages.interfaces.test_stage import TestStage
from jatic_ri.image_classification.test_stages.interfaces.plugins import (
    MetricPlugin,
    MultiModelPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    TwoDatasetPlugin,
)


def test_model_dataset_metric(dummy_model_ic, dummy_dataset_ic, dummy_metric_ic) -> None:
    class TestImpl(TestStage[None], SingleModelPlugin, SingleDatasetPlugin, MetricPlugin):
        """Dummy implementation class"""

        def _run(self) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()

    stage = TestImpl()

    stage.load_model(
        model=dummy_model_ic,
        model_id="dummy1",
    )

    stage.load_dataset(
        dataset=dummy_dataset_ic,
        dataset_id="dummy1",
    )

    stage.load_metric(
        metric=dummy_metric_ic,
        metric_id="dummy1",
    )


def test_single_dataset(dummy_dataset_ic) -> None:
    class TestImpl(TestStage[None], SingleDatasetPlugin):
        """Dummy implementation class"""

        def _run(self) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()

    stage = TestImpl()

    stage.load_dataset(
        dataset=dummy_dataset_ic,
        dataset_id="dummy1",
    )


def test_two_dataset(dummy_dataset_ic) -> None:
    class TestImpl(TestStage[None], TwoDatasetPlugin):
        """Dummy implementation class"""

        def _run(self) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()

    stage = TestImpl()

    stage.load_datasets(
        dataset_1=dummy_dataset_ic,
        dataset_1_id="dummy1",
        dataset_2=dummy_dataset_ic,
        dataset_2_id="dummy2",
    )


def test_single_model_dataset(dummy_model_ic, dummy_dataset_ic) -> None:
    class TestImpl(TestStage[None], SingleModelPlugin, SingleDatasetPlugin):
        """Dummy implementation class"""

        def _run(self) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()

    stage = TestImpl()

    stage.load_model(
        model=dummy_model_ic,
        model_id="dummy1",
    )

    stage.load_dataset(
        dataset=dummy_dataset_ic,
        dataset_id="dummy1",
    )


def test_multi_model_single_dataset(dummy_model_ic, dummy_dataset_ic) -> None:
    class TestImpl(TestStage[None], MultiModelPlugin, SingleDatasetPlugin):
        """Dummy implementation class"""

        def _run(self) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()

    stage = TestImpl()

    models = {
        "dummy1": dummy_model_ic,
        "dummy2": dummy_model_ic,
    }
    stage.load_models(models=models)

    stage.load_dataset(
        dataset=dummy_dataset_ic,
        dataset_id="dummy1",
    )


def test_dataset_metric(dummy_dataset_ic, dummy_metric_ic) -> None:
    class TestImpl(TestStage[None], SingleDatasetPlugin, MetricPlugin):
        """Dummy implementation class"""

        def _run(self) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()

    stage = TestImpl()

    stage.load_dataset(
        dataset=dummy_dataset_ic,
        dataset_id="dummy1",
    )

    stage.load_metric(
        metric=dummy_metric_ic,
        metric_id="dummy1",
    )


def test_test_stage_no_default_cache() -> None:
    class TestImpl(TestStage[None]):
        """Dummy implementation class"""

        def _run(self) -> None:
            pass

        def collect_report_consumables(self) -> list[dict[str, Any]]:
            return super().collect_report_consumables()

    stage = TestImpl()

    assert not stage.cache_path
    assert not stage.cache
