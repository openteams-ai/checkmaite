from typing import Any

import pytest

from jatic_ri._common.test_stages.interfaces.plugins import MetricPlugin, SingleModelPlugin, TwoDatasetPlugin
from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, Number, OutputsBase, RunBase, TestStage


class MockConfig(ConfigBase):
    pass


class MockOutputs(OutputsBase):
    result: bool


class MockRun(RunBase):
    config: MockConfig
    outputs: MockOutputs


class MockTestStage(TestStage[bool]):
    """Mock test stage for testing"""

    _RUN_TYPE = MockRun

    def _create_config(self):
        return MockConfig()

    def _run(self):
        return MockOutputs(result=True)

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        "Mock report consumables"
        if self._stored_run is None:
            raise RuntimeError("TestStage must be run before accessing outputs")
        outputs = self._stored_run.outputs
        return [{"report": outputs.result}]


class MockTestStagePlugins(TestStage[bool], SingleModelPlugin, MetricPlugin, TwoDatasetPlugin):
    """Mock test stage for testing"""

    _RUN_TYPE = MockRun

    def _create_config(self):
        return MockConfig()

    def _run(self):
        return MockOutputs(result=True)

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        "Mock report consumables"
        if self._stored_run is None:
            raise RuntimeError("TestStage must be run before accessing outputs")
        outputs = self._stored_run.outputs
        return [{"report": outputs.result}]


def test_teststage_collect_with_run() -> None:
    m = MockTestStage()
    m.run()
    report = m.collect_report_consumables()
    assert report
    assert m.supports_datasets == Number.ZERO
    assert m.supports_models == Number.ZERO
    assert m.supports_metric == Number.ZERO


def test_teststage_collect_without_run() -> None:
    m = MockTestStage()
    with pytest.raises(RuntimeError):
        m.collect_report_consumables()


def test_teststage_plugins() -> None:
    m = MockTestStagePlugins()

    assert m.supports_datasets == Number.TWO
    assert m.supports_models == Number.ONE
    assert m.supports_metric == Number.ONE
