import pytest

from checkmaite.core.analytics_store import (
    AnalyticsStore,
    BaseRecord,
    ParquetBackend,
    configure_provenance,
    get_provenance_defaults,
    reset_provenance,
)
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


@pytest.fixture(autouse=True)
def _reset_provenance_defaults() -> None:
    reset_provenance()
    yield
    reset_provenance()


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


def test_analytics_store_persists_provenance_on_runs(store: AnalyticsStore) -> None:
    run = _run()
    run.outputs.records.append(
        StoreMetricRecord(run_uid=run.run_uid, dataset_id="dataset-1", metric_name="accuracy", metric_value=0.9)
    )

    store.write_with_receipt(
        [run],
        provenance={
            "user_id": "alice",
            "workspace_id": "workspace-a",
            "job_id": "job-1",
            "backend": "ray",
            "environment": "databricks",
            "executor": "ray",
            "cluster_id": "cluster-42",
            "request_id": "req-123",
            "run_event_id": "invoke-1",
        },
    )

    result = store.query_sql("""
        SELECT DISTINCT user_id, workspace_id, job_id, backend, environment,
                        executor, cluster_id, request_id, run_event_id
        FROM runs
    """)

    assert result.to_dicts() == [
        {
            "user_id": "alice",
            "workspace_id": "workspace-a",
            "job_id": "job-1",
            "backend": "ray",
            "environment": "databricks",
            "executor": "ray",
            "cluster_id": "cluster-42",
            "request_id": "req-123",
            "run_event_id": "invoke-1",
        }
    ]


def test_analytics_store_uses_explicit_configured_defaults_with_write_provenance(store: AnalyticsStore) -> None:
    configure_provenance(user_id="alice", workspace_id="workspace-a", environment="databricks")
    run = _run()
    run.outputs.records.append(
        StoreMetricRecord(run_uid=run.run_uid, dataset_id="dataset-1", metric_name="accuracy", metric_value=0.9)
    )

    provenance = get_provenance_defaults().merge({"job_id": "job-1", "backend": "ray"})
    store.write_with_receipt([run], provenance=provenance)

    result = store.query_sql("SELECT DISTINCT user_id, workspace_id, environment, job_id, backend FROM runs")
    assert result.to_dicts() == [
        {
            "user_id": "alice",
            "workspace_id": "workspace-a",
            "environment": "databricks",
            "job_id": "job-1",
            "backend": "ray",
        }
    ]


def test_analytics_store_does_not_read_process_defaults_implicitly(store: AnalyticsStore) -> None:
    configure_provenance(user_id="alice")
    run = _run()
    run.outputs.records.append(
        StoreMetricRecord(run_uid=run.run_uid, dataset_id="dataset-1", metric_name="accuracy", metric_value=0.9)
    )

    store.write_with_receipt([run])

    result = store.query_sql("SELECT DISTINCT user_id FROM runs")
    assert result["user_id"].to_list() == [None]


def test_repeated_local_writes_dedupe_run_history_and_payload(store: AnalyticsStore) -> None:
    run = _run()
    run.outputs.records.append(
        StoreMetricRecord(run_uid=run.run_uid, dataset_id="dataset-1", metric_name="accuracy", metric_value=0.9)
    )

    store.write([run])
    store.write([run])

    payload_rows = store.query_sql("SELECT * FROM store_metrics")
    history = store.query_sql("SELECT DISTINCT run_event_id FROM runs")
    runs_rows = store.query_sql("SELECT * FROM runs")

    assert payload_rows.shape[0] == 1
    assert history["run_event_id"].to_list() == [None]
    assert runs_rows.shape[0] == 3


def test_explicit_run_event_ids_preserve_run_history_without_duplicating_payload(store: AnalyticsStore) -> None:
    run = _run()
    run.outputs.records.append(
        StoreMetricRecord(run_uid=run.run_uid, dataset_id="dataset-1", metric_name="accuracy", metric_value=0.9)
    )

    store.write([run], provenance={"run_event_id": "invoke-1"})
    store.write([run], provenance={"run_event_id": "invoke-2"})

    payload_rows = store.query_sql("SELECT * FROM store_metrics")
    history = store.query_sql("SELECT DISTINCT run_event_id FROM runs ORDER BY run_event_id")
    runs_rows = store.query_sql("SELECT * FROM runs")

    assert payload_rows.shape[0] == 1
    assert history["run_event_id"].to_list() == ["invoke-1", "invoke-2"]
    assert runs_rows.shape[0] == 6


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
