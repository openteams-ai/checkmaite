import polars as pl
import pytest

from jatic_ri.core.analytics_store._schema import BaseRecord, RunRecord
from jatic_ri.core.analytics_store._storage._parquet import ParquetBackend


class MetricRecord(BaseRecord, table_name="metrics"):
    dataset_id: str
    metric_name: str
    metric_value: float


class CleaningRecord(BaseRecord, table_name="cleaning"):
    dataset_id: str
    duplicate_count: int
    outlier_ratio: float
    note: str | None = None


@pytest.fixture
def backend(tmp_path) -> ParquetBackend:
    return ParquetBackend(str(tmp_path / "store"))


def _metric(
    run_uid: str = "abc123", dataset_id: str = "ds1", name: str = "accuracy", value: float = 0.95
) -> MetricRecord:
    return MetricRecord(run_uid=run_uid, dataset_id=dataset_id, metric_name=name, metric_value=value)


def _cleaning(run_uid: str = "abc123", dataset_id: str = "ds1", dup: int = 5, ratio: float = 0.1) -> CleaningRecord:
    return CleaningRecord(run_uid=run_uid, dataset_id=dataset_id, duplicate_count=dup, outlier_ratio=ratio)


def test_write_and_query_single_table(backend: ParquetBackend) -> None:
    backend.write([_metric(), _metric(run_uid="def456", value=0.88)])

    result = backend.query_sql("SELECT * FROM metrics ORDER BY metric_value")
    assert result.shape[0] == 2
    assert result["metric_value"].to_list() == [0.88, 0.95]


def test_write_and_query_multiple_tables(backend: ParquetBackend) -> None:
    backend.write([_metric(), _cleaning()])

    result_m = backend.query_sql("SELECT * FROM metrics")
    result_c = backend.query_sql("SELECT * FROM cleaning")
    assert result_m.shape[0] == 1
    assert result_c.shape[0] == 1
    assert result_c["duplicate_count"].to_list() == [5]


def test_cross_table_join(backend: ParquetBackend) -> None:
    backend.write(
        [
            _metric(run_uid="r1", dataset_id="ds1", value=0.9),
            _metric(run_uid="r2", dataset_id="ds2", value=0.8),
            _cleaning(run_uid="r1", dataset_id="ds1", dup=3),
            _cleaning(run_uid="r2", dataset_id="ds2", dup=10),
        ]
    )

    result = backend.query_sql("""
        SELECT m.metric_value, c.duplicate_count
        FROM metrics m
        JOIN cleaning c ON m.dataset_id = c.dataset_id
        ORDER BY m.metric_value
    """)
    assert result.shape[0] == 2
    assert result["duplicate_count"].to_list() == [10, 3]


def test_write_across_multiple_calls(backend: ParquetBackend) -> None:
    """Data from separate write() calls is visible in a single query."""
    backend.write([_metric(run_uid="r1", value=0.9)])
    backend.write([_metric(run_uid="r2", value=0.8)])

    result = backend.query_sql("SELECT * FROM metrics")
    assert result.shape[0] == 2


def test_sql_filtering(backend: ParquetBackend) -> None:
    backend.write(
        [
            _metric(run_uid="r1", name="accuracy", value=0.9),
            _metric(run_uid="r2", name="precision", value=0.85),
            _metric(run_uid="r3", name="accuracy", value=0.7),
        ]
    )

    result = backend.query_sql("SELECT * FROM metrics WHERE metric_name = 'accuracy'")
    assert result.shape[0] == 2


def test_sql_aggregation(backend: ParquetBackend) -> None:
    backend.write(
        [
            _metric(run_uid="r1", name="accuracy", value=0.9),
            _metric(run_uid="r2", name="accuracy", value=0.8),
        ]
    )

    result = backend.query_sql("SELECT AVG(metric_value) AS avg_val FROM metrics")
    assert abs(result["avg_val"][0] - 0.85) < 1e-9


def test_run_records_queryable(backend: ParquetBackend) -> None:
    """RunRecord (now a BaseRecord subclass) writes to the runs table and is queryable."""
    run_rec = RunRecord(
        run_uid="r1",
        capability_id="cap1",
        capability_table="metrics",
        entity_type="dataset",
        entity_id="ds1",
    )
    backend.write([_metric(run_uid="r1"), run_rec])

    result = backend.query_sql("""
        SELECT m.metric_value
        FROM metrics m
        JOIN runs r ON m.run_uid = r.run_uid
        WHERE r.entity_id = 'ds1'
    """)
    assert result.shape[0] == 1
    assert result["metric_value"][0] == 0.95


def test_datetime_preserved(backend: ParquetBackend) -> None:
    """created_at should survive write → read as a datetime, not a string."""
    backend.write([_metric()])

    result = backend.query_sql("SELECT created_at FROM metrics")
    assert result.dtypes[0] == pl.Datetime or result.dtypes[0].is_(pl.Datetime)


def test_deduplication_by_run_uid(backend: ParquetBackend) -> None:
    backend.write([_metric(run_uid="r1", value=0.9)])
    backend.write([_metric(run_uid="r1", value=0.99)])  # same run_uid, different value

    result = backend.query_sql("SELECT * FROM metrics")
    assert result.shape[0] == 1
    assert result["metric_value"][0] == 0.9  # first write wins


def test_deduplication_idempotent_within_single_write(backend: ParquetBackend) -> None:
    """First write with duplicate run_uids should write all (dedup is cross-call only)."""
    backend.write([_metric(run_uid="r1", value=0.9), _metric(run_uid="r1", value=0.99)])

    result = backend.query_sql("SELECT * FROM metrics")
    # Both rows written because dedup only checks existing files, not within-batch
    assert result.shape[0] == 2


def test_schema_mismatch_raises(backend: ParquetBackend) -> None:
    """Writing a record whose column type conflicts with existing data raises TypeError."""
    backend.write([_metric(run_uid="r1")])

    # Create a record type with same table_name but incompatible field type
    class BadMetricRecord(BaseRecord, table_name="metrics"):
        dataset_id: str
        metric_name: str
        metric_value: int  # was float

    bad = BadMetricRecord(run_uid="r2", dataset_id="ds1", metric_name="acc", metric_value=1)
    with pytest.raises(TypeError, match="Schema mismatch"):
        backend.write([bad])


def test_schema_evolution_optional_field_within_batch(backend: ParquetBackend) -> None:
    """Optional fields work correctly when mixed None/non-None values are in the same batch.

    Within a single write() call, Polars infers the column as String (not Null)
    because at least one row has a non-None value.
    """
    rec1 = CleaningRecord(run_uid="r1", dataset_id="ds1", duplicate_count=5, outlier_ratio=0.1, note="first")
    rec2 = CleaningRecord(run_uid="r2", dataset_id="ds2", duplicate_count=3, outlier_ratio=0.05)
    backend.write([rec1, rec2])

    result = backend.query_sql("SELECT * FROM cleaning ORDER BY run_uid")
    assert result.shape[0] == 2
    notes = result.sort("run_uid")["note"].to_list()
    assert notes[0] == "first"
    assert notes[1] is None


def test_schema_evolution_null_then_string_allowed(backend: ParquetBackend) -> None:
    """Existing Null column (all-None) accepts a subsequent String write."""
    backend.write([_cleaning(run_uid="r1")])  # note=None → Null dtype

    rec = CleaningRecord(run_uid="r2", dataset_id="ds2", duplicate_count=0, outlier_ratio=0.0, note="hello")
    backend.write([rec])

    result = backend.query_sql("SELECT run_uid, note FROM cleaning ORDER BY run_uid")
    assert result.shape[0] == 2
    assert result["note"].to_list() == [None, "hello"]


def test_schema_evolution_string_then_null_allowed(backend: ParquetBackend) -> None:
    """Existing String column accepts a new batch where that column is all-None.

    Polars infers the column as Null when every value is None, but this is not a
    real type conflict — it just means the new batch has no information for that
    column.
    """
    rec1 = CleaningRecord(run_uid="r1", dataset_id="ds1", duplicate_count=5, outlier_ratio=0.1, note="first")
    backend.write([rec1])

    rec2 = CleaningRecord(run_uid="r2", dataset_id="ds2", duplicate_count=0, outlier_ratio=0.0)
    backend.write([rec2])  # should not raise

    result = backend.query_sql("SELECT run_uid, note FROM cleaning ORDER BY run_uid")
    assert result.shape[0] == 2
    assert result["note"].to_list() == ["first", None]


def test_idempotent_write_with_optional_none_does_not_raise(backend: ParquetBackend) -> None:
    """Re-writing a deduplicated record with an all-None optional column must not
    raise a schema mismatch error. The schema check should treat Null in the new
    batch the same way it treats Null in existing data: as absence of type info."""
    rec = CleaningRecord(run_uid="r1", dataset_id="ds1", duplicate_count=5, outlier_ratio=0.1, note="hello")
    backend.write([rec])

    # Re-write the same run_uid but with note=None — Polars infers Null dtype.
    # This must not raise, and the row count must stay at 1 (deduplicated).
    dup = CleaningRecord(run_uid="r1", dataset_id="ds1", duplicate_count=5, outlier_ratio=0.1)
    backend.write([dup])

    result = backend.query_sql("SELECT * FROM cleaning")
    assert result.shape[0] == 1


def test_list_tables_empty(backend: ParquetBackend) -> None:
    assert backend.list_tables() == []


def test_list_tables_after_write(backend: ParquetBackend) -> None:
    backend.write([_metric(), _cleaning()])
    tables = sorted(backend.list_tables())
    assert tables == ["cleaning", "metrics"]


def test_describe_table(backend: ParquetBackend) -> None:
    backend.write([_metric()])
    schema = backend.describe_table("metrics")
    assert "run_uid" in schema
    assert "metric_value" in schema
    assert "metric_name" in schema


def test_describe_table_nonexistent_raises(backend: ParquetBackend) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        backend.describe_table("no_such_table")


def test_write_empty_list(backend: ParquetBackend) -> None:
    """Writing an empty list is a no-op."""
    backend.write([])
    assert backend.list_tables() == []


def test_query_empty_store(backend: ParquetBackend) -> None:
    """Querying before any writes returns an empty DataFrame."""
    result = backend.query_sql("SELECT 1")
    assert result.is_empty()


def test_query_invalid_sql_raises(backend: ParquetBackend) -> None:
    backend.write([_metric()])
    with pytest.raises(ValueError, match="SQL query failed"):
        backend.query_sql("SELECT * FROM nonexistent_table")
