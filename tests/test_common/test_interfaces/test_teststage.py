import json
from typing import Any

import pytest
from pydantic import BaseModel

from jatic_ri._common.test_stages.interfaces.test_stage import Run, TestStage


class FakeRun(Run):
    def serialize_outputs(self, outputs: dict[str, list[str]]) -> str:
        return json.dumps(self.outputs)

    @classmethod
    def deserialize_outputs(cls, data: str) -> dict[str, list[str]]:
        return json.loads(data)


class FakeConfig(BaseModel):
    pass


def test_run_implementation() -> None:
    config = FakeConfig()
    model_ids = ["1234"]
    dataset_ids = ["4321"]
    metric_id = "aaaa"
    outputs = {"fake-item-1": ["fake-result"]}

    run = FakeRun(config=config, model_ids=model_ids, dataset_ids=dataset_ids, metric_id=metric_id, outputs=outputs)

    serialized = run.model_dump()

    assert len(serialized) == 5

    deserialized_outputs = run.deserialize_outputs(serialized["outputs"])

    assert deserialized_outputs == run.outputs


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
