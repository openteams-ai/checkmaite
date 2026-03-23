from pathlib import Path
from typing import Any

import numpy as np
import pytest
from maite.protocols import DatasetMetadata
from PIL import Image as PILImage

from checkmaite.core._common.dataeval_bias_capability import (
    DataevalBiasBalanceOutputs,
    DataevalBiasBase,
    DataevalBiasConfig,
    DataevalBiasCoverageOutputs,
    DataevalBiasDiversityOutputs,
    DataevalBiasOutputs,
    DataevalBiasRecord,
    DataevalBiasRun,
)
from checkmaite.core.object_detection.dataset_loaders import CocoDetectionDataset
from checkmaite.core.report._gradient import HAS_GRADIENT


@pytest.fixture
def run(dummy_dataset_ic, fake_image):
    return DataevalBiasRun(
        capability_id="DataevalBiasCapability",
        dataset_metadata=[dummy_dataset_ic.metadata],
        model_metadata=[],
        metric_metadata=[],
        config=DataevalBiasConfig(device="cpu"),
        outputs=DataevalBiasOutputs(
            balance=DataevalBiasBalanceOutputs(
                balance=np.array([0.99999822, 0.13363788, 0.04505382, 0.02994455]),
                factors=np.array(
                    [
                        [0.99999843, 0.04133555, 0.09725766],
                        [0.04133555, 0.08433558, 0.1301489],
                        [0.09725766, 0.1301489, 0.99999856],
                    ]
                ),
                classwise=np.array([[0.99999822, 0.13363788, 0.0, 0.0], [0.99999822, 0.13363788, 0.0, 0.0]]),
                factor_names=[],
                class_names=[],
                image_metadata=fake_image,
                image_classwise=fake_image,
            ),
            diversity=DataevalBiasDiversityOutputs(
                diversity_index=np.array([0.18103448, 0.18103448, 0.88636364]),
                classwise=np.array([[0.17241379, 0.39473684], [0.2, 0.2]]),
                factor_names=[],
                class_names=[],
                image=fake_image,
            ),
            coverage=DataevalBiasCoverageOutputs(
                total=len(dummy_dataset_ic),
                uncovered_indices=np.array([447, 412, 8, 32, 63]),
                coverage_radius=0.8459038956941765,
                critical_value_radii=np.arange(10),
                image=fake_image,
            ),
        ),
    )


@pytest.fixture
def test_run_ic(fake_ic_dataset_default) -> Any:
    capability = DataevalBiasBase()

    return capability.run(use_cache=False, datasets=[fake_ic_dataset_default])  # smoke test


def test_collect_md_report_ic(test_run_ic):
    md = test_run_ic.collect_md_report(threshold=0.5)
    assert md  # smoke test


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_collect_report_consumables_ic(test_run_ic):
    with pytest.warns(DeprecationWarning):
        consumables = test_run_ic.collect_report_consumables(threshold=0.5)
    assert consumables  # smoke test


class TestOdDataevalBiasCapability:
    ROOT = Path(__file__).parent.parent.parent / "data_for_tests"
    coco_dataset_dir = ROOT / "coco_resized_val2017"

    @pytest.fixture
    def test_run_od(self, fake_od_dataset_default) -> Any:
        capability = DataevalBiasBase()

        return capability.run(use_cache=False, datasets=[fake_od_dataset_default])  # smoke test

    def test_collect_md_report_od(self, test_run_od):
        md = test_run_od.collect_md_report(threshold=0.5)
        assert md  # smoke test

    @pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
    def test_collect_report_consumables_od(self, test_run_od):
        with pytest.warns(DeprecationWarning):
            consumables = test_run_od.collect_report_consumables(threshold=0.5)
        assert consumables  # smoke test

    def test_coco(self):
        coco_dataset = CocoDetectionDataset(
            root=str(self.coco_dataset_dir),
            ann_file=str(self.coco_dataset_dir.joinpath("instances_val2017_resized_6.json")),
        )

        capability = DataevalBiasBase()

        capability.run(use_cache=False, datasets=[coco_dataset])
        pass  # no explosions

    def test_no_metadata(self):
        coco_dataset = CocoDetectionDataset(
            root=str(self.coco_dataset_dir),
            ann_file=str(self.coco_dataset_dir.joinpath("instances_val2017_resized_6.json")),
        )

        coco_keys = ["license", "file_name", "coco_url", "height", "width", "date_captured", "flickr_url", "id"]
        for _, _, metadata in coco_dataset:
            assert set(metadata.keys()) == set(coco_keys)

        metadata_to_exclude = [
            "license",
            "file_name",
            "coco_url",
            "height",
            "width",
            "date_captured",
            "flickr_url",
            "id",
        ]

        config = DataevalBiasConfig(metadata_to_exclude=metadata_to_exclude)

        capability = DataevalBiasBase()

        capability.run(use_cache=False, datasets=[coco_dataset], config=config)
        pass  # no explosions


def test_bias_record_accepts_valid_fields():
    record = DataevalBiasRecord(
        run_uid="abc123",
        dataset_id="test_dataset",
        coverage_total=100,
        coverage_uncovered_count=5,
        coverage_uncovered_ratio=0.05,
        coverage_radius=0.85,
        balance_num_factors=3,
        balance_mean=0.25,
        balance_max=0.45,
        balance_factors_above_05=0,
        diversity_num_factors=3,
        diversity_mean=0.7,
        diversity_min=0.3,
        diversity_factors_below_04=1,
    )
    assert record.table_name == "dataeval_bias"
    assert record.dataset_id == "test_dataset"
    assert record.coverage_total == 100


def test_bias_record_accepts_null_balance_and_diversity():
    """When no metadata factors exist, balance and diversity fields are None."""
    record = DataevalBiasRecord(
        run_uid="abc123",
        dataset_id="test_dataset",
        coverage_total=100,
        coverage_uncovered_count=5,
        coverage_uncovered_ratio=0.05,
        coverage_radius=0.85,
    )
    assert record.balance_num_factors is None
    assert record.balance_mean is None
    assert record.diversity_num_factors is None
    assert record.diversity_mean is None


@pytest.fixture
def bias_run_with_all_outputs():
    """A DataevalBiasRun with all three output groups populated, using inline values."""
    _dummy_img = PILImage.new("RGB", (1, 1))
    return DataevalBiasRun(
        capability_id="DataevalBiasCapability",
        dataset_metadata=[DatasetMetadata(id="test_dataset", index2label={0: "cat", 1: "dog"})],
        model_metadata=[],
        metric_metadata=[],
        config=DataevalBiasConfig(device="cpu"),
        outputs=DataevalBiasOutputs(
            balance=DataevalBiasBalanceOutputs(
                balance=np.array([0.99999822, 0.13363788, 0.04505382, 0.02994455]),
                factors=np.array([[1.0, 0.04], [0.04, 1.0]]),
                classwise=np.array([[0.5, 0.1], [0.5, 0.1]]),
                factor_names=["factor_a", "factor_b", "factor_c", "factor_d"],
                class_names=["cat", "dog"],
                image_metadata=_dummy_img,
                image_classwise=_dummy_img,
            ),
            diversity=DataevalBiasDiversityOutputs(
                diversity_index=np.array([0.18103448, 0.18103448, 0.88636364]),
                classwise=np.array([[0.17, 0.39], [0.2, 0.2]]),
                factor_names=["factor_a", "factor_b", "factor_c"],
                class_names=["cat", "dog"],
                image=_dummy_img,
            ),
            coverage=DataevalBiasCoverageOutputs(
                total=200,
                uncovered_indices=np.array([10, 20, 30, 40, 50]),
                coverage_radius=0.8459038956941765,
                critical_value_radii=np.arange(10),
            ),
        ),
    )


def test_extract_returns_one_record(bias_run_with_all_outputs):
    records = bias_run_with_all_outputs.extract()
    assert len(records) == 1
    assert isinstance(records[0], DataevalBiasRecord)


def test_extract_coverage_fields(bias_run_with_all_outputs):
    record = bias_run_with_all_outputs.extract()[0]
    assert record.dataset_id == "test_dataset"
    assert record.coverage_total == 200
    assert record.coverage_uncovered_count == 5
    assert record.coverage_radius == pytest.approx(0.8459038956941765)
    assert record.coverage_uncovered_ratio == pytest.approx(5 / 200)


def test_extract_balance_fields(bias_run_with_all_outputs):
    record = bias_run_with_all_outputs.extract()[0]
    # balance array is [0.99999822, 0.13363788, 0.04505382, 0.02994455]
    assert record.balance_num_factors == 4
    assert record.balance_mean == pytest.approx(float(np.mean([0.99999822, 0.13363788, 0.04505382, 0.02994455])))
    assert record.balance_max == pytest.approx(0.99999822)
    assert record.balance_factors_above_05 == 1  # only 0.99999822 >= 0.5


def test_extract_diversity_fields(bias_run_with_all_outputs):
    record = bias_run_with_all_outputs.extract()[0]
    # diversity_index is [0.18103448, 0.18103448, 0.88636364]
    assert record.diversity_num_factors == 3
    assert record.diversity_mean == pytest.approx(float(np.mean([0.18103448, 0.18103448, 0.88636364])))
    assert record.diversity_min == pytest.approx(0.18103448)
    assert record.diversity_factors_below_04 == 2  # 0.18103448 and 0.18103448 are < 0.4


def test_extract_without_balance_or_diversity():
    """When metadata factors are absent, balance and diversity fields are None."""
    run_no_meta = DataevalBiasRun(
        capability_id="DataevalBiasCapability",
        dataset_metadata=[DatasetMetadata(id="no_meta_dataset", index2label={0: "a", 1: "b"})],
        model_metadata=[],
        metric_metadata=[],
        config=DataevalBiasConfig(device="cpu"),
        outputs=DataevalBiasOutputs(
            balance=None,
            diversity=None,
            coverage=DataevalBiasCoverageOutputs(
                total=50,
                uncovered_indices=np.array([1, 2]),
                coverage_radius=0.5,
                critical_value_radii=np.arange(5),
            ),
        ),
    )
    records = run_no_meta.extract()
    assert len(records) == 1
    record = records[0]
    assert record.dataset_id == "no_meta_dataset"
    assert record.coverage_total == 50
    assert record.coverage_uncovered_count == 2
    assert record.balance_num_factors is None
    assert record.balance_mean is None
    assert record.balance_max is None
    assert record.balance_factors_above_05 is None
    assert record.diversity_num_factors is None
    assert record.diversity_mean is None
    assert record.diversity_min is None
    assert record.diversity_factors_below_04 is None


def test_extract_zero_total_coverage():
    """Edge case: coverage_total == 0 should not cause division by zero."""
    run_empty = DataevalBiasRun(
        capability_id="DataevalBiasCapability",
        dataset_metadata=[DatasetMetadata(id="empty_ds", index2label={})],
        model_metadata=[],
        metric_metadata=[],
        config=DataevalBiasConfig(device="cpu"),
        outputs=DataevalBiasOutputs(
            balance=None,
            diversity=None,
            coverage=DataevalBiasCoverageOutputs(
                total=0,
                uncovered_indices=np.array([]),
                coverage_radius=0.0,
                critical_value_radii=np.array([]),
            ),
        ),
    )
    record = run_empty.extract()[0]
    assert record.coverage_total == 0
    assert record.coverage_uncovered_count == 0
    assert record.coverage_uncovered_ratio == 0.0
