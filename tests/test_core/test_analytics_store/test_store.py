import pytest

from checkmaite.core.analytics_store import AnalyticsStore, BaseRecord, ParquetBackend
from checkmaite.core.capability_core import CapabilityConfigBase, CapabilityOutputsBase, CapabilityRunBase


class StoreMetricRecord(BaseRecord, table_name="store_metrics"):
    dataset_id: str
    metric_name: str
    metric_value: float


class StoreOtherRecord(BaseRecord, table_name="store_other"):
    value: int


class StoreOutputs(CapabilityOutputsBase):
    records: list[BaseRecord]


class StoreRun(CapabilityRunBase[CapabilityConfigBase, StoreOutputs]):
    def extract(self) -> list[BaseRecord]:
        return self.outputs.records


def _run(records: list[BaseRecord] | None = None) -> StoreRun:
    return StoreRun(
        capability_id="store-capability",
        config=CapabilityConfigBase(),
        dataset_metadata=[{"id": "dataset-1"}],
        model_metadata=[{"id": "model-1"}],
        metric_metadata=[{"id": "accuracy"}],
        outputs=StoreOutputs(records=records or []),
    )


@pytest.fixture
def store(tmp_path) -> AnalyticsStore:
    return AnalyticsStore(ParquetBackend(str(tmp_path / "store")))


def test_analytics_store_writes_payloads_with_queryable_run_metadata(store: AnalyticsStore) -> None:
    run = _run()
    run.outputs.records.append(
        StoreMetricRecord(run_uid=run.run_uid, dataset_id="dataset-1", metric_name="accuracy", metric_value=0.9)
    )

    receipt = store.write_with_receipt([run])
    result = store.query_sql("""
        SELECT m.metric_value
        FROM store_metrics m
        JOIN runs r ON m.run_uid = r.run_uid
        WHERE r.entity_id = 'dataset-1'
    """)

    assert result["metric_value"].to_list() == [0.9]
    assert set(store.list_tables()) == {"runs", "store_metrics"}
    assert store.describe_table("store_metrics")["metric_value"] == "Float64"
    assert store.get_run_uri(run.run_uid) == receipt.resolve_run_uri(run.run_uid)


def test_analytics_store_empty_write_is_a_noop(store: AnalyticsStore) -> None:
    receipt = store.write_with_receipt([])
    store.write([])

    assert receipt.resolve_run_uri("missing") is None
    assert store.list_tables() == []


def test_analytics_store_rejects_runs_that_extract_multiple_payload_tables(store: AnalyticsStore) -> None:
    run = _run(
        [
            StoreMetricRecord(run_uid="mixed-run", dataset_id="dataset-1", metric_name="accuracy", metric_value=0.9),
            StoreOtherRecord(run_uid="mixed-run", value=1),
        ]
    )

    with pytest.raises(ValueError, match="exactly one payload table"):
        store.write([run])
