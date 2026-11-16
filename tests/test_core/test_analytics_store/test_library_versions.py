"""Analytics rows from before and after a library upgrade stay apart and can be told apart."""

import importlib.metadata

import numpy as np
import pytest

from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend, RunVersionRecord
from checkmaite.core.image_classification import DataevalCleaning


class _SmallImageDataset:
    metadata = {"id": "upgrade-images", "index2label": {0: "a", 1: "b"}}

    def __len__(self) -> int:
        return 4

    def __getitem__(self, i: int):
        return np.full((3, 32, 32), 25 + i, dtype=np.uint8), np.eye(2)[i % 2], {"id": i}


@pytest.fixture
def store(tmp_path) -> AnalyticsStore:
    return AnalyticsStore(ParquetBackend(str(tmp_path / "store")))


def _pin_dataeval(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    real_version = importlib.metadata.version

    def fake_version(distribution: str) -> str:
        return version if distribution == "dataeval" else real_version(distribution)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)


def test_upgraded_run_gets_a_new_run_uid_and_keeps_the_old_rows(monkeypatch, store) -> None:
    capability = DataevalCleaning()
    dataset = _SmallImageDataset()
    installed = capability.cache_versions()["dataeval"]

    with monkeypatch.context() as patch:
        _pin_dataeval(patch, "1.0.6")
        old = capability.run(datasets=[dataset])
    store.write([old])

    # The cache entry written under the old version must not be reused.
    new = capability.run(datasets=[dataset])
    store.write([new])

    assert old.versions["dataeval"] == "1.0.6"
    assert new.versions["dataeval"] == installed != "1.0.6"
    assert new.run_uid != old.run_uid

    payload = store.query_sql("SELECT run_uid FROM dataeval_cleaning")
    assert sorted(payload["run_uid"].to_list()) == sorted([old.run_uid, new.run_uid])

    by_version = store.query_sql("""
        SELECT v.version AS dataeval_version, c.run_uid
        FROM dataeval_cleaning c
        JOIN run_versions v ON c.run_uid = v.run_uid
        WHERE v.kind = 'library' AND v.name = 'dataeval'
        ORDER BY dataeval_version
    """)
    assert dict(zip(by_version["dataeval_version"], by_version["run_uid"], strict=True)) == {
        "1.0.6": old.run_uid,
        installed: new.run_uid,
    }


def test_run_versions_table_has_one_row_per_library(store) -> None:
    run = DataevalCleaning().run(datasets=[_SmallImageDataset()])
    store.write([run])

    rows = store.query_sql("SELECT run_uid, kind, name, version FROM run_versions").rows()

    assert rows == [
        (run.run_uid, "library", "dataeval", run.versions["dataeval"]),
    ]


def test_runs_table_holds_only_inputs(store) -> None:
    run = DataevalCleaning().run(datasets=[_SmallImageDataset()])
    store.write([run])

    rows = store.query_sql("SELECT entity_type, entity_id FROM runs").rows()

    assert rows == [("dataset", "upgrade-images")]
    assert "entity_version" not in store.describe_table("runs")


def test_version_rows_are_written_once_per_run(store) -> None:
    run = DataevalCleaning().run(datasets=[_SmallImageDataset()])
    store.write([run, run])
    store.write([run], provenance={"run_event_id": "job-2"})

    count = store.query_sql("SELECT COUNT(*) AS n FROM run_versions")["n"][0]

    assert count == 1  # dataeval


def test_version_table_is_not_a_run_payload(store) -> None:
    run = DataevalCleaning().run(datasets=[_SmallImageDataset()])
    receipt = store.write_with_receipt([run])

    assert set(receipt.run_table_files[run.run_uid]) == {"dataeval_cleaning"}
    assert "/dataeval_cleaning/" in store.get_run_uri(run.run_uid)


def test_unknown_version_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        RunVersionRecord(run_uid="abc", kind="capability", name="x", version="1")


def test_a_capability_without_libraries_writes_no_version_rows(
    store, fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default
) -> None:
    from checkmaite.core.image_classification import MaiteEvaluation

    run = MaiteEvaluation().run(
        models=[fake_ic_model_default], datasets=[fake_ic_dataset_default], metrics=[fake_ic_metric_default]
    )
    store.write([run])

    assert "capability" in run.versions  # still part of the run UID
    assert "run_versions" not in store.list_tables()
