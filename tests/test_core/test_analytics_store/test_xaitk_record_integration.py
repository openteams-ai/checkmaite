import pytest

from checkmaite.core._common.xaitk_explainable_capability import XaitkExplainableRecord
from checkmaite.core.analytics_store._storage._parquet import ParquetBackend


@pytest.fixture
def backend(tmp_path) -> ParquetBackend:
    return ParquetBackend(str(tmp_path / "store"))


def _xaitk_record(
    run_uid: str = "xaitk_ic_1",
    dataset_id: str = "ds1",
    model_id: str = "model_a",
    saliency_generator_type: str = "RISEStack",
    image_index: int = 0,
    gt_label: str | None = "cat",
    mean_saliency: float = 0.25,
    max_saliency: float = 0.85,
    std_saliency: float = 0.18,
    positive_saliency_ratio: float = 0.60,
) -> XaitkExplainableRecord:
    return XaitkExplainableRecord(
        run_uid=run_uid,
        dataset_id=dataset_id,
        model_id=model_id,
        saliency_generator_type=saliency_generator_type,
        image_index=image_index,
        gt_label=gt_label,
        mean_saliency=mean_saliency,
        max_saliency=max_saliency,
        std_saliency=std_saliency,
        positive_saliency_ratio=positive_saliency_ratio,
    )


def _od_record(
    run_uid: str = "xaitk_od_1",
    dataset_id: str = "ds_od",
    model_id: str = "model_b",
    saliency_generator_type: str = "DRISEStack",
    image_index: int = 0,
    image_id: str | None = "img_001",
    detection_index: int | None = 0,
    predicted_label: str | None = "car",
    confidence: float | None = 0.92,
    mean_saliency: float = 0.31,
    max_saliency: float = 0.78,
    std_saliency: float = 0.21,
    positive_saliency_ratio: float = 0.55,
) -> XaitkExplainableRecord:
    return XaitkExplainableRecord(
        run_uid=run_uid,
        dataset_id=dataset_id,
        model_id=model_id,
        saliency_generator_type=saliency_generator_type,
        image_index=image_index,
        image_id=image_id,
        detection_index=detection_index,
        predicted_label=predicted_label,
        confidence=confidence,
        mean_saliency=mean_saliency,
        max_saliency=max_saliency,
        std_saliency=std_saliency,
        positive_saliency_ratio=positive_saliency_ratio,
    )


def test_write_and_query_ic_records(backend: ParquetBackend) -> None:
    backend.write([_xaitk_record(), _xaitk_record(run_uid="xaitk_ic_2", gt_label="dog", mean_saliency=0.40)])

    result = backend.query_sql("SELECT * FROM xaitk_explainable ORDER BY mean_saliency")
    assert result.shape[0] == 2
    assert result["gt_label"].to_list() == ["cat", "dog"]
    assert result["mean_saliency"].to_list() == pytest.approx([0.25, 0.40])


def test_write_and_query_od_records(backend: ParquetBackend) -> None:
    backend.write([_od_record(), _od_record(run_uid="xaitk_od_2", image_id="img_002", confidence=0.75)])

    result = backend.query_sql("SELECT * FROM xaitk_explainable ORDER BY confidence")
    assert result.shape[0] == 2
    assert result["image_id"].to_list() == ["img_002", "img_001"]
    assert result["confidence"].to_list() == pytest.approx([0.75, 0.92])


def test_multi_image_ic_records(backend: ParquetBackend) -> None:
    """Multiple images produce separate records, each with their own image_index."""
    records = [
        _xaitk_record(run_uid=f"xaitk_ic_{i}", image_index=i, gt_label="cat", mean_saliency=0.1 * (i + 1))
        for i in range(4)
    ]
    backend.write(records)

    result = backend.query_sql("SELECT image_index, mean_saliency FROM xaitk_explainable ORDER BY image_index")
    assert result.shape[0] == 4
    assert result["image_index"].to_list() == [0, 1, 2, 3]


def test_multi_detection_od_records(backend: ParquetBackend) -> None:
    """Multiple detections within one image each get their own record with detection_index."""
    records = [
        _od_record(
            run_uid=f"xaitk_od_det_{i}",
            image_index=0,
            detection_index=i,
            predicted_label="car",
            confidence=0.9 - 0.1 * i,
        )
        for i in range(3)
    ]
    backend.write(records)

    result = backend.query_sql("SELECT detection_index, confidence FROM xaitk_explainable ORDER BY detection_index")
    assert result.shape[0] == 3
    assert result["detection_index"].to_list() == [0, 1, 2]
    assert result["confidence"][0] == pytest.approx(0.9)


def test_saliency_stats_queryable(backend: ParquetBackend) -> None:
    backend.write(
        [_xaitk_record(mean_saliency=0.25, max_saliency=0.85, std_saliency=0.18, positive_saliency_ratio=0.60)]
    )

    result = backend.query_sql(
        "SELECT mean_saliency, max_saliency, std_saliency, positive_saliency_ratio FROM xaitk_explainable"
    )
    assert result.shape[0] == 1
    assert result["mean_saliency"][0] == pytest.approx(0.25)
    assert result["max_saliency"][0] == pytest.approx(0.85)
    assert result["std_saliency"][0] == pytest.approx(0.18)
    assert result["positive_saliency_ratio"][0] == pytest.approx(0.60)


def test_generator_type_filter(backend: ParquetBackend) -> None:
    """Records can be filtered by saliency_generator_type."""
    backend.write(
        [
            _xaitk_record(run_uid="rise_1", saliency_generator_type="RISEStack"),
            _od_record(run_uid="drise_1", saliency_generator_type="DRISEStack"),
        ]
    )

    result = backend.query_sql("SELECT run_uid FROM xaitk_explainable WHERE saliency_generator_type = 'RISEStack'")
    assert result.shape[0] == 1
    assert result["run_uid"][0] == "rise_1"


def test_ic_records_have_null_od_fields(backend: ParquetBackend) -> None:
    """IC records leave OD-specific fields as None."""
    backend.write([_xaitk_record()])

    result = backend.query_sql("SELECT image_id, detection_index, predicted_label, confidence FROM xaitk_explainable")
    assert result.shape[0] == 1
    assert result["image_id"][0] is None
    assert result["detection_index"][0] is None
    assert result["predicted_label"][0] is None
    assert result["confidence"][0] is None


def test_deduplication_by_run_uid(backend: ParquetBackend) -> None:
    """Writing the same run_uid twice across separate writes is a no-op."""
    backend.write([_xaitk_record(run_uid="dup1", mean_saliency=0.25)])
    backend.write([_xaitk_record(run_uid="dup1", mean_saliency=0.99)])

    result = backend.query_sql("SELECT * FROM xaitk_explainable")
    assert result.shape[0] == 1
    assert result["mean_saliency"][0] == pytest.approx(0.25)


def test_cross_table_join_with_cleaning(backend: ParquetBackend) -> None:
    """XaitkExplainable records can JOIN with other capability tables via dataset_id."""
    from checkmaite.core._common.dataeval_cleaning_capability import DataevalCleaningRecord

    xaitk = _xaitk_record(run_uid="r1", dataset_id="shared_ds")
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
    backend.write([xaitk, cleaning])

    result = backend.query_sql("""
        SELECT x.mean_saliency, c.exact_duplicate_ratio
        FROM xaitk_explainable x
        JOIN dataeval_cleaning c ON x.dataset_id = c.dataset_id
    """)
    assert result.shape[0] == 1
    assert result["mean_saliency"][0] == pytest.approx(0.25)
    assert result["exact_duplicate_ratio"][0] == pytest.approx(0.03)
