import pytest

from checkmaite.core._common.dataeval_shift_capability import DataevalShiftRecord
from checkmaite.core.analytics_store._storage._parquet import ParquetBackend


@pytest.fixture
def backend(tmp_path) -> ParquetBackend:
    return ParquetBackend(str(tmp_path / "store"))


def _shift_record(
    run_uid: str = "shift_1",
    reference_dataset_id: str = "ds_ref",
    evaluation_dataset_id: str = "ds_eval",
    mmd_drifted: bool = True,
    mmd_distance: float = 0.45,
    mmd_p_val: float = 0.01,
    mmd_threshold: float = 0.05,
    cvm_drifted: bool = True,
    cvm_distance: float = 0.38,
    cvm_p_val: float = 0.02,
    cvm_threshold: float = 0.05,
    cvm_feature_drift_count: int = 5,
    ks_drifted: bool = False,
    ks_distance: float = 0.12,
    ks_p_val: float = 0.15,
    ks_threshold: float = 0.05,
    ks_feature_drift_count: int = 2,
    ood_count: int = 3,
    ood_total: int = 50,
    ood_ratio: float = 0.06,
    ood_mean_instance_score: float = 0.72,
    ood_std_instance_score: float = 0.15,
    ood_max_instance_score: float = 1.05,
) -> DataevalShiftRecord:
    return DataevalShiftRecord(
        run_uid=run_uid,
        reference_dataset_id=reference_dataset_id,
        evaluation_dataset_id=evaluation_dataset_id,
        mmd_drifted=mmd_drifted,
        mmd_distance=mmd_distance,
        mmd_p_val=mmd_p_val,
        mmd_threshold=mmd_threshold,
        cvm_drifted=cvm_drifted,
        cvm_distance=cvm_distance,
        cvm_p_val=cvm_p_val,
        cvm_threshold=cvm_threshold,
        cvm_feature_drift_count=cvm_feature_drift_count,
        ks_drifted=ks_drifted,
        ks_distance=ks_distance,
        ks_p_val=ks_p_val,
        ks_threshold=ks_threshold,
        ks_feature_drift_count=ks_feature_drift_count,
        ood_count=ood_count,
        ood_total=ood_total,
        ood_ratio=ood_ratio,
        ood_mean_instance_score=ood_mean_instance_score,
        ood_std_instance_score=ood_std_instance_score,
        ood_max_instance_score=ood_max_instance_score,
    )


def test_write_and_query_records(backend: ParquetBackend) -> None:
    backend.write([_shift_record(), _shift_record(run_uid="shift_2", mmd_distance=0.80)])

    result = backend.query_sql("SELECT * FROM dataeval_shift ORDER BY mmd_distance")
    assert result.shape[0] == 2
    assert result["mmd_distance"].to_list() == [0.45, 0.80]


def test_drift_fields_queryable(backend: ParquetBackend) -> None:
    backend.write([_shift_record()])

    result = backend.query_sql("""
        SELECT mmd_drifted, mmd_p_val, cvm_drifted, cvm_p_val, ks_drifted, ks_p_val
        FROM dataeval_shift
    """)
    assert result.shape[0] == 1
    assert result["mmd_drifted"][0] is True
    assert result["mmd_p_val"][0] == pytest.approx(0.01)
    assert result["cvm_drifted"][0] is True
    assert result["ks_drifted"][0] is False


def test_ood_fields_queryable(backend: ParquetBackend) -> None:
    backend.write([_shift_record()])

    result = backend.query_sql("""
        SELECT ood_count, ood_total, ood_ratio, ood_mean_instance_score
        FROM dataeval_shift
    """)
    assert result.shape[0] == 1
    assert result["ood_count"][0] == 3
    assert result["ood_total"][0] == 50
    assert result["ood_ratio"][0] == pytest.approx(0.06)
    assert result["ood_mean_instance_score"][0] == pytest.approx(0.72)


def test_drift_thresholds_queryable(backend: ParquetBackend) -> None:
    """Drift thresholds allow distinguishing results at different significance levels."""
    backend.write([_shift_record(mmd_threshold=0.05, cvm_threshold=0.005, ks_threshold=0.005)])

    result = backend.query_sql("SELECT mmd_threshold, cvm_threshold, ks_threshold FROM dataeval_shift")
    assert result.shape[0] == 1
    assert result["mmd_threshold"][0] == pytest.approx(0.05)
    assert result["cvm_threshold"][0] == pytest.approx(0.005)
    assert result["ks_threshold"][0] == pytest.approx(0.005)


def test_feature_drift_counts_queryable(backend: ParquetBackend) -> None:
    """Per-feature drift counts show how widespread drift is across embedding dimensions."""
    backend.write([_shift_record(cvm_feature_drift_count=12, ks_feature_drift_count=3)])

    result = backend.query_sql("SELECT cvm_feature_drift_count, ks_feature_drift_count FROM dataeval_shift")
    assert result.shape[0] == 1
    assert result["cvm_feature_drift_count"][0] == 12
    assert result["ks_feature_drift_count"][0] == 3


def test_ood_distribution_stats_queryable(backend: ParquetBackend) -> None:
    """OOD std and max capture score distribution shape beyond the mean."""
    backend.write([_shift_record(ood_std_instance_score=0.25, ood_max_instance_score=1.8)])

    result = backend.query_sql("SELECT ood_std_instance_score, ood_max_instance_score FROM dataeval_shift")
    assert result.shape[0] == 1
    assert result["ood_std_instance_score"][0] == pytest.approx(0.25)
    assert result["ood_max_instance_score"][0] == pytest.approx(1.8)


def test_two_dataset_ids(backend: ParquetBackend) -> None:
    """Shift records have both reference and evaluation dataset IDs."""
    backend.write([_shift_record(reference_dataset_id="ref_ds", evaluation_dataset_id="eval_ds")])

    result = backend.query_sql("SELECT reference_dataset_id, evaluation_dataset_id FROM dataeval_shift")
    assert result["reference_dataset_id"][0] == "ref_ds"
    assert result["evaluation_dataset_id"][0] == "eval_ds"


def test_deduplication_by_run_uid(backend: ParquetBackend) -> None:
    """Writing the same run_uid twice across separate writes is a no-op."""
    backend.write([_shift_record(run_uid="dup1", mmd_distance=0.45)])
    backend.write([_shift_record(run_uid="dup1", mmd_distance=0.99)])

    result = backend.query_sql("SELECT * FROM dataeval_shift")
    assert result.shape[0] == 1
    assert result["mmd_distance"][0] == pytest.approx(0.45)


def test_cross_table_join_with_feasibility(backend: ParquetBackend) -> None:
    """Shift records can JOIN with single-dataset capabilities via reference_dataset_id."""
    from checkmaite.core._common.dataeval_feasibility_record import DataevalFeasibilityRecord

    shift = _shift_record(run_uid="r1", reference_dataset_id="shared_ds")
    feasibility = DataevalFeasibilityRecord(
        run_uid="r2",
        dataset_id="shared_ds",
        ber_upper=0.15,
        ber_lower=0.08,
    )
    backend.write([shift, feasibility])

    result = backend.query_sql("""
        SELECT s.mmd_drifted, f.ber_upper
        FROM dataeval_shift s
        JOIN dataeval_feasibility f ON s.reference_dataset_id = f.dataset_id
    """)
    assert result.shape[0] == 1
    assert result["mmd_drifted"][0] is True
    assert result["ber_upper"][0] == pytest.approx(0.15)
