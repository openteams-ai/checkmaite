import pytest

from checkmaite.core._common.dataeval_bias_capability import DataevalBiasRecord
from checkmaite.core.analytics_store._storage._parquet import ParquetBackend


@pytest.fixture
def backend(tmp_path) -> ParquetBackend:
    return ParquetBackend(str(tmp_path / "store"))


def _bias_record(
    run_uid: str = "bias_run_1",
    dataset_id: str = "ds1",
    coverage_total: int = 100,
    uncovered: int = 5,
    radius: float = 0.85,
    with_balance: bool = True,
    with_diversity: bool = True,
) -> DataevalBiasRecord:
    return DataevalBiasRecord(
        run_uid=run_uid,
        dataset_id=dataset_id,
        coverage_total=coverage_total,
        coverage_uncovered_count=uncovered,
        coverage_uncovered_ratio=uncovered / coverage_total if coverage_total > 0 else 0.0,
        coverage_radius=radius,
        balance_num_factors=3 if with_balance else None,
        balance_mean=0.25 if with_balance else None,
        balance_max=0.45 if with_balance else None,
        balance_factors_above_05=0 if with_balance else None,
        diversity_num_factors=3 if with_diversity else None,
        diversity_mean=0.7 if with_diversity else None,
        diversity_min=0.3 if with_diversity else None,
        diversity_factors_below_04=1 if with_diversity else None,
    )


def test_write_and_query_bias_records(backend: ParquetBackend) -> None:
    backend.write([_bias_record(), _bias_record(run_uid="bias_run_2", dataset_id="ds2", uncovered=10)])

    result = backend.query_sql("SELECT * FROM dataeval_bias ORDER BY coverage_uncovered_count")
    assert result.shape[0] == 2
    assert result["dataset_id"].to_list() == ["ds1", "ds2"]
    assert result["coverage_uncovered_count"].to_list() == [5, 10]


def test_bias_records_with_null_balance_diversity(backend: ParquetBackend) -> None:
    """Records with no metadata factors (balance/diversity = None) write and query correctly."""
    backend.write([_bias_record(with_balance=False, with_diversity=False)])

    result = backend.query_sql("SELECT balance_mean, diversity_mean FROM dataeval_bias")
    assert result.shape[0] == 1
    assert result["balance_mean"][0] is None
    assert result["diversity_mean"][0] is None


def test_cross_table_join_with_cleaning(backend: ParquetBackend) -> None:
    """Bias records can JOIN with other capability tables via dataset_id."""
    from checkmaite.core._common.dataeval_cleaning_capability import DataevalCleaningRecord

    bias = _bias_record(run_uid="r1", dataset_id="shared_ds")
    cleaning = DataevalCleaningRecord(
        run_uid="r2",
        dataset_id="shared_ds",
        exact_duplicate_count=3,
        exact_duplicate_ratio=0.03,
        near_duplicate_count=1,
        near_duplicate_ratio=0.01,
        image_outlier_count=2,
        image_outlier_ratio=0.02,
        class_count=5,
        label_count=100,
        image_count=100,
        mean_width=224.0,
        mean_height=224.0,
        std_aspect_ratio=0.1,
        mean_brightness=0.5,
        mean_contrast=0.5,
        mean_sharpness=0.5,
        class_imbalance_ratio=1.5,
        min_class_image_count=15,
        max_class_image_count=25,
        mean_labels_per_image=1.0,
    )
    backend.write([bias, cleaning])

    result = backend.query_sql("""
        SELECT b.coverage_uncovered_ratio, c.exact_duplicate_ratio
        FROM dataeval_bias b
        JOIN dataeval_cleaning c ON b.dataset_id = c.dataset_id
    """)
    assert result.shape[0] == 1
    assert result["coverage_uncovered_ratio"][0] == pytest.approx(0.05)
    assert result["exact_duplicate_ratio"][0] == pytest.approx(0.03)
