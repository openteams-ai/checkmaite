from typing import Any

import pytest

from jatic_ri._common.test_stages.interfaces.test_stage import TestStage


class MockTestStage(TestStage[bool]):
    """Mock test stage for testing"""

    def _run(self) -> bool:
        return True

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        "Mock report consumables"
        return [{"report": self.outputs}]


def test_teststage_collect_with_run() -> None:
    m = MockTestStage()
    m.run()
    report = m.collect_report_consumables()
    assert report


def test_teststage_collect_without_run() -> None:
    m = MockTestStage()
    with pytest.raises(RuntimeError):
        m.collect_report_consumables()
