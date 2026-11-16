import pytest
import torch

from checkmaite.core._common.nrtk_robustness_capability import (
    NrtkRobustnessConfig,
    NrtkRobustnessOutputs,
    NrtkRobustnessPoint,
    NrtkRobustnessRecord,
    NrtkRobustnessRun,
)
from checkmaite.core.analytics_store._storage._parquet import ParquetBackend


@pytest.fixture
def backend(tmp_path) -> ParquetBackend:
    return ParquetBackend(str(tmp_path / "store"))


def _nrtk_record(
    run_uid: str = "nrtk_1",
    dataset_id: str = "ds1",
    model_id: str = "resnet50",
    metric_id: str = "coco_metrics",
    perturber_class: str = "BrightnessPerturber",
    perturber_type: str = "Brightness Perturber",
    theta_key: str = "factor",
    theta_index: int = 0,
    theta_value: float = 1.0,
    metric_key: str = "accuracy",
    metric_value: float = 0.95,
    baseline_value: float = 0.97,
    is_primary: bool = True,
) -> NrtkRobustnessRecord:
    return NrtkRobustnessRecord(
        run_uid=run_uid,
        dataset_id=dataset_id,
        model_id=model_id,
        metric_id=metric_id,
        perturber_class=perturber_class,
        perturber_type=perturber_type,
        theta_key=theta_key,
        theta_index=theta_index,
        theta_value=theta_value,
        metric_key=metric_key,
        metric_value=metric_value,
        baseline_value=baseline_value,
        is_primary=is_primary,
    )


def _robustness_curve_records(run_uid: str = "nrtk_curve") -> list[NrtkRobustnessRecord]:
    """Create a small robustness curve: 3 theta points x 2 metrics = 6 records."""
    records = []
    # (theta_index, theta_value, accuracy)
    thetas = [(0, 1.0, 0.95), (1, 3.0, 0.82), (2, 5.0, 0.65)]
    for theta_index, theta_value, accuracy in thetas:
        records.append(
            _nrtk_record(
                run_uid=run_uid,
                theta_index=theta_index,
                theta_value=theta_value,
                metric_key="accuracy",
                metric_value=accuracy,
                is_primary=True,
            )
        )
        records.append(
            _nrtk_record(
                run_uid=run_uid,
                theta_index=theta_index,
                theta_value=theta_value,
                metric_key="f1_score",
                metric_value=accuracy - 0.05,
                is_primary=False,
            )
        )
    return records


def test_write_and_query_single_record(backend: ParquetBackend) -> None:
    backend.write([_nrtk_record()])

    result = backend.query_sql("SELECT * FROM nrtk_robustness")
    assert result.shape[0] == 1
    assert result["dataset_id"][0] == "ds1"
    assert result["metric_value"][0] == pytest.approx(0.95)


def test_robustness_curve_reconstruction(backend: ParquetBackend) -> None:
    """Full curve can be reconstructed via SQL ORDER BY theta_index."""
    backend.write(_robustness_curve_records())

    result = backend.query_sql("""
        SELECT theta_value, metric_value FROM nrtk_robustness
        WHERE is_primary = true
        ORDER BY theta_index
    """)
    assert result.shape[0] == 3
    assert result["theta_value"].to_list() == [1.0, 3.0, 5.0]
    assert result["metric_value"].to_list() == [pytest.approx(0.95), pytest.approx(0.82), pytest.approx(0.65)]


def test_multi_metric_records(backend: ParquetBackend) -> None:
    """Each theta point produces one record per metric key."""
    backend.write(_robustness_curve_records())

    result = backend.query_sql("""
        SELECT metric_key, COUNT(*) AS cnt FROM nrtk_robustness
        GROUP BY metric_key ORDER BY metric_key
    """)
    assert result.shape[0] == 2
    assert result["metric_key"].to_list() == ["accuracy", "f1_score"]
    assert result["cnt"].to_list() == [3, 3]


def test_is_primary_filter(backend: ParquetBackend) -> None:
    """is_primary flag correctly distinguishes metric_key from secondary metrics."""
    backend.write(_robustness_curve_records())

    primary = backend.query_sql("SELECT COUNT(*) AS n FROM nrtk_robustness WHERE is_primary = true")
    secondary = backend.query_sql("SELECT COUNT(*) AS n FROM nrtk_robustness WHERE is_primary = false")
    assert primary["n"][0] == 3
    assert secondary["n"][0] == 3


def test_perturber_fields(backend: ParquetBackend) -> None:
    backend.write([_nrtk_record()])

    result = backend.query_sql("SELECT perturber_class, perturber_type, theta_key FROM nrtk_robustness")
    assert result["perturber_class"][0] == "BrightnessPerturber"
    assert result["perturber_type"][0] == "Brightness Perturber"
    assert result["theta_key"][0] == "factor"


def test_deduplication_by_run_uid(backend: ParquetBackend) -> None:
    """Writing the same run_uid twice across separate writes is a no-op."""
    backend.write([_nrtk_record(run_uid="dup1", metric_value=0.95)])
    backend.write([_nrtk_record(run_uid="dup1", metric_value=0.99)])

    result = backend.query_sql("SELECT * FROM nrtk_robustness")
    assert result.shape[0] == 1
    assert result["metric_value"][0] == pytest.approx(0.95)


def test_cross_table_join_with_feasibility(backend: ParquetBackend) -> None:
    """NRTK records can JOIN with other capabilities via dataset_id."""
    from checkmaite.core._common.dataeval_feasibility_record import DataevalFeasibilityRecord

    nrtk = _nrtk_record(run_uid="r1", dataset_id="shared_ds")
    feasibility = DataevalFeasibilityRecord(
        run_uid="r2",
        dataset_id="shared_ds",
        ber_upper=0.15,
        ber_lower=0.08,
    )
    backend.write([nrtk, feasibility])

    result = backend.query_sql("""
        SELECT n.metric_value, f.ber_upper
        FROM nrtk_robustness n
        JOIN dataeval_feasibility f ON n.dataset_id = f.dataset_id
    """)
    assert result.shape[0] == 1
    assert result["metric_value"][0] == pytest.approx(0.95)
    assert result["ber_upper"][0] == pytest.approx(0.15)


def _make_run(
    perturbations: list[dict[str, object]], metric_key: str = "accuracy", baseline: dict[str, object] | None = None
) -> NrtkRobustnessRun:
    """Construct a minimal NrtkRobustnessRun for testing extract(), with thetas 1.0, 2.0, ..."""
    baseline = baseline if baseline is not None else {key: 1.0 for key in perturbations[0]}
    outputs = NrtkRobustnessOutputs(
        perturber="BrightnessPerturber",
        theta_key="factor",
        baseline=baseline,
        points=[
            NrtkRobustnessPoint(theta=float(index + 1), metrics=metrics) for index, metrics in enumerate(perturbations)
        ],
        metric_key=metric_key,
    )
    return NrtkRobustnessRun(
        capability_id="test.NrtkRobustness",
        config=NrtkRobustnessConfig(),
        dataset_metadata=[{"id": "ds1"}],
        model_metadata=[{"id": "resnet50"}],
        metric_metadata=[{"id": "coco_metrics"}],
        outputs=outputs,
    )


def test_extract_produces_records_per_theta_and_metric() -> None:
    """extract() returns one record per (theta, metric_key) pair."""
    run = _make_run(
        perturbations=[
            {"accuracy": 0.95, "f1_score": 0.90},
            {"accuracy": 0.82, "f1_score": 0.78},
        ],
    )
    records = run.extract()

    assert len(records) == 4  # 2 thetas x 2 metrics
    assert all(isinstance(r, NrtkRobustnessRecord) for r in records)
    assert all(r.run_uid == run.run_uid for r in records)
    assert all(r.dataset_id == "ds1" for r in records)
    assert all(r.model_id == "resnet50" for r in records)


def test_extract_is_primary_flag() -> None:
    """extract() sets is_primary=True only for the metric_key metric."""
    run = _make_run(perturbations=[{"accuracy": 0.9, "f1_score": 0.8}])
    records = run.extract()

    primary = [r for r in records if r.is_primary]
    secondary = [r for r in records if not r.is_primary]
    assert len(primary) == 1
    assert primary[0].metric_key == "accuracy"
    assert len(secondary) == 1
    assert secondary[0].metric_key == "f1_score"


def test_extract_handles_tensor_values() -> None:
    """extract() converts torch.Tensor metric values to float."""
    run = _make_run(perturbations=[{"accuracy": torch.tensor(0.95)}])
    records = run.extract()

    assert len(records) == 1
    assert records[0].metric_value == pytest.approx(0.95)
    assert isinstance(records[0].metric_value, float)


def test_extract_perturber_fields() -> None:
    """extract() correctly parses perturber class name and label from config."""
    run = _make_run(perturbations=[{"accuracy": 0.9}])
    records = run.extract()

    assert records[0].perturber_class == "BrightnessPerturber"
    assert records[0].perturber_type == "Brightness Perturber"
    assert records[0].theta_key == "factor"


def test_extract_records_the_baseline_for_each_metric() -> None:
    """extract() puts the unperturbed score for the same metric key on every record."""
    run = _make_run(
        perturbations=[{"accuracy": 0.8, "f1_score": 0.7}, {"accuracy": 0.6, "f1_score": 0.5}],
        baseline={"accuracy": torch.tensor(0.9), "f1_score": 0.85},
    )
    records = run.extract()

    assert [(r.metric_key, r.theta_value) for r in records] == [
        ("accuracy", 1.0),
        ("f1_score", 1.0),
        ("accuracy", 2.0),
        ("f1_score", 2.0),
    ]
    assert [r.baseline_value for r in records] == pytest.approx([0.9, 0.85, 0.9, 0.85])


def test_extract_skips_results_that_are_not_single_numbers() -> None:
    """An unfiltered mAP metric also returns per-class arrays, which have no single value to record."""
    results = {
        "map": torch.tensor(0.3),
        "map_50": torch.tensor(0.5),
        "map_per_class": torch.tensor([0.1, 0.2]),
        "classes": torch.tensor([0, 1], dtype=torch.int32),
    }
    run = _make_run(perturbations=[results], metric_key="map_50", baseline=results)

    records = run.extract()

    assert sorted(r.metric_key for r in records) == ["map", "map_50"]
    assert [r.metric_key for r in records if r.is_primary] == ["map_50"]


def test_extract_skips_per_class_results_that_are_single_numbers() -> None:
    """With per-class metrics off or one class, torchmetrics returns 0-d per-class tensors."""
    results = {"map": torch.tensor(0.3), "map_per_class": torch.tensor(-1.0), "classes": torch.tensor(3)}
    run = _make_run(perturbations=[results], metric_key="map", baseline=results)

    assert [r.metric_key for r in run.extract()] == ["map"]
