import pytest

from checkmaite.core._common.dataeval_feasibility_record import DataevalFeasibilityRecord
from checkmaite.core.analytics_store._storage._parquet import ParquetBackend


@pytest.fixture
def backend(tmp_path) -> ParquetBackend:
    return ParquetBackend(str(tmp_path / "store"))


def _ic_record(
    run_uid: str = "feas_ic_1",
    dataset_id: str = "ds1",
    ber_upper: float = 0.15,
    ber_lower: float = 0.08,
) -> DataevalFeasibilityRecord:
    return DataevalFeasibilityRecord(
        run_uid=run_uid,
        dataset_id=dataset_id,
        ber_upper=ber_upper,
        ber_lower=ber_lower,
    )


def _od_record(
    run_uid: str = "feas_od_1",
    dataset_id: str = "ds1",
    ber_upper: float = 0.25,
    ber_lower: float = 0.12,
    num_instances: int = 500,
    num_classes: int = 10,
    small_object_ratio: float = 0.05,
    truncated_bbox_ratio: float = 0.03,
    overlap_image_ratio: float = 0.02,
    health_warning_count: int = 0,
) -> DataevalFeasibilityRecord:
    return DataevalFeasibilityRecord(
        run_uid=run_uid,
        dataset_id=dataset_id,
        ber_upper=ber_upper,
        ber_lower=ber_lower,
        num_instances=num_instances,
        num_classes=num_classes,
        small_object_ratio=small_object_ratio,
        truncated_bbox_ratio=truncated_bbox_ratio,
        overlap_image_ratio=overlap_image_ratio,
        health_warning_count=health_warning_count,
    )


def test_write_and_query_ic_records(backend: ParquetBackend) -> None:
    backend.write([_ic_record(), _ic_record(run_uid="feas_ic_2", dataset_id="ds2", ber_upper=0.30)])

    result = backend.query_sql("SELECT * FROM dataeval_feasibility ORDER BY ber_upper")
    assert result.shape[0] == 2
    assert result["dataset_id"].to_list() == ["ds1", "ds2"]
    assert result["ber_upper"].to_list() == [0.15, 0.30]


def test_write_and_query_od_records(backend: ParquetBackend) -> None:
    backend.write([_od_record(), _od_record(run_uid="feas_od_2", dataset_id="ds2", num_instances=200)])

    result = backend.query_sql("SELECT * FROM dataeval_feasibility ORDER BY num_instances")
    assert result.shape[0] == 2
    assert result["num_instances"].to_list() == [200, 500]


def test_ic_records_have_null_od_fields(backend: ParquetBackend) -> None:
    """IC records leave OD-specific fields as None."""
    backend.write([_ic_record()])

    result = backend.query_sql("SELECT num_instances, num_classes, small_object_ratio FROM dataeval_feasibility")
    assert result.shape[0] == 1
    assert result["num_instances"][0] is None
    assert result["num_classes"][0] is None
    assert result["small_object_ratio"][0] is None


def test_mixed_ic_and_od_records(backend: ParquetBackend) -> None:
    """IC and OD records coexist in the same table."""
    backend.write([_ic_record(run_uid="ic1", dataset_id="ds_ic"), _od_record(run_uid="od1", dataset_id="ds_od")])

    result = backend.query_sql(
        "SELECT dataset_id, ber_upper, num_instances FROM dataeval_feasibility ORDER BY ber_upper"
    )
    assert result.shape[0] == 2
    assert result["dataset_id"].to_list() == ["ds_ic", "ds_od"]
    assert result["num_instances"][0] is None
    assert result["num_instances"][1] == 500


def test_deduplication_by_run_uid(backend: ParquetBackend) -> None:
    """Writing the same run_uid twice across separate writes is a no-op."""
    backend.write([_ic_record(run_uid="dup1", ber_upper=0.15)])
    backend.write([_ic_record(run_uid="dup1", ber_upper=0.99)])

    result = backend.query_sql("SELECT * FROM dataeval_feasibility")
    assert result.shape[0] == 1
    assert result["ber_upper"][0] == pytest.approx(0.15)


def test_cross_table_join_with_bias(backend: ParquetBackend) -> None:
    """Feasibility records can JOIN with other capability tables via dataset_id."""
    from checkmaite.core._common.dataeval_bias_capability import DataevalBiasRecord

    feasibility = _ic_record(run_uid="r1", dataset_id="shared_ds")
    bias = DataevalBiasRecord(
        run_uid="r2",
        dataset_id="shared_ds",
        coverage_total=100,
        coverage_uncovered_count=5,
        coverage_uncovered_ratio=0.05,
        coverage_radius=0.85,
    )
    backend.write([feasibility, bias])

    result = backend.query_sql("""
        SELECT f.ber_upper, b.coverage_uncovered_ratio
        FROM dataeval_feasibility f
        JOIN dataeval_bias b ON f.dataset_id = b.dataset_id
    """)
    assert result.shape[0] == 1
    assert result["ber_upper"][0] == pytest.approx(0.15)
    assert result["coverage_uncovered_ratio"][0] == pytest.approx(0.05)
