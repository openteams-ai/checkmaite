from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
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


def _create_fake_bias_run(
    dataset,
    with_balance: bool = True,
    with_diversity: bool = True,
    num_factors: int = 0,
):
    fake_image = PILImage.new("RGB", (1, 1))
    index2label = dataset.metadata["index2label"]
    factor_names = [f"factor_{i}" for i in range(num_factors)]
    balance_df = pl.from_dict(
        {"Factor": ["class_label", *factor_names], "Balance Score": [1.0, *[0.1 * (i + 1) for i in range(num_factors)]]}
    )
    cbs = list(combinations(factor_names, 2))
    factors_df = pl.from_dict(
        {
            "factor1": [value[0] for value in cbs],
            "factor2": [value[1] for value in cbs],
            "mi_value": [0.1 * i for i, _ in enumerate(cbs)],
            "is_correlated": [False for _ in cbs],
        }
    )
    class_names = list(index2label.values())
    prod = list(product(class_names, [*factor_names, "class_label"]))
    classwise_df = pl.from_dict(
        {
            "class_name": [value[0] for value in prod],
            "factor_name": [value[1] for value in prod],
            "mi_value": [1.0 for _ in prod],
            "is_imbalanced": [False for _ in prod],
        }
    )
    balance = (
        DataevalBiasBalanceOutputs(
            balance=balance_df,
            factors=factors_df,
            classwise=classwise_df,
            image_metadata=None if num_factors > 1 else fake_image,
            image_classwise=fake_image,
        )
        if with_balance
        else None
    )
    factors_df = pl.from_dict(
        {
            "Factor": ["class_label", *factor_names],
            "Diversity Index": [0.7] * num_factors + [0.1],
        }
    )
    prod = list(product(class_names, factor_names))
    classwise_df = pl.from_dict(
        {
            "class_name": [value[0] for value in prod],
            "factor_name": [value[1] for value in prod],
            "mi_value": [1.0 for _ in prod],
            "is_imbalanced": [False for _ in prod],
        }
    )
    diversity = (
        DataevalBiasDiversityOutputs(
            factors=factors_df,
            classwise=classwise_df,
            image=fake_image,
        )
        if with_diversity
        else None
    )
    coverage = DataevalBiasCoverageOutputs(
        total=len(dataset),
        uncovered_indices=np.array([447, 412, 8, 32, 63]),
        coverage_radius=0.8459038956941765,
        critical_value_radii=np.arange(10),
        image=fake_image,
    )
    return DataevalBiasRun(
        capability_id="DataevalBiasCapability",
        dataset_metadata=[dataset.metadata],
        model_metadata=[],
        metric_metadata=[],
        config=DataevalBiasConfig(device="cpu"),
        outputs=DataevalBiasOutputs(
            balance=balance,
            diversity=diversity,
            coverage=coverage,
        ),
    )


def do_smoke_run(dataset):
    capability = DataevalBiasBase()
    return capability.run(use_cache=False, datasets=[dataset])  # smoke test


@pytest.fixture
def test_run_ic(fake_ic_dataset_default) -> Any:
    return do_smoke_run(fake_ic_dataset_default)


def test_collect_md_report_ic(test_run_ic):
    md = test_run_ic.collect_md_report(threshold=0.5)
    assert md  # smoke test

    expected_data = [
        "| Potentially under-represented images | 1 of 20 (5.0%) |",
        "| Coverage radius |",
    ]
    md_str = str(md)
    for expected in expected_data:
        assert expected in md_str, md_str


def test_bias_output(fake_ic_dataset_cifar10_metadata):
    output = do_smoke_run(fake_ic_dataset_cifar10_metadata)

    assert output.outputs.balance.balance.to_dicts() == [
        {"Factor": "class_label", "Balance Score": 1.0},
        {"Factor": "batch_id", "Balance Score": 0.0},
    ]
    assert len(output.outputs.balance.factors) == 0
    assert output.outputs.balance.classwise.shape == (14, 4)
    assert isinstance(output.outputs.balance.image_classwise, PILImage.Image)
    assert output.outputs.balance.image_metadata is None

    assert output.outputs.diversity.factors.to_dicts() == [
        {"Diversity Index": pytest.approx(0.759259), "Factor": "class_label"},
        {"Diversity Index": 0.0, "Factor": "batch_id"},
    ]
    assert output.outputs.diversity.classwise.shape == (7, 4)

    assert output.outputs.coverage.total == 20


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
def bias_run_with_all_outputs(fake_ic_dataset_default):
    """A DataevalBiasRun with all three output groups populated, using inline values."""

    return _create_fake_bias_run(
        fake_ic_dataset_default,
        with_balance=True,
        with_diversity=True,
        num_factors=4,
    )


def test_extract_returns_one_record(bias_run_with_all_outputs):
    records = bias_run_with_all_outputs.extract()
    assert len(records) == 1
    assert isinstance(records[0], DataevalBiasRecord)


def test_extract_fields(bias_run_with_all_outputs):
    record = bias_run_with_all_outputs.extract()[0]
    assert record.dataset_id == "fake_id_dataset"
    assert record.coverage_total == 20
    assert record.coverage_uncovered_count == 5
    assert record.coverage_radius == pytest.approx(0.84590389)
    assert record.coverage_uncovered_ratio == pytest.approx(5 / 20)

    assert record.balance_num_factors == 5
    assert record.balance_mean == 0.4  # avg of Balance Score data: [1.0, 0.1, 0.2, 0.3, 0.4]
    assert record.balance_max == 1.0
    assert record.balance_factors_above_05 == 1  # only 0.99999822 >= 0.5

    assert record.diversity_num_factors == 5
    assert record.diversity_mean == 0.58  # avg of Diversity Index data: [0.7] * 4 + [0.1]
    assert record.diversity_min == 0.1
    assert record.diversity_factors_below_04 == 1


def test_extract_without_balance_or_diversity(fake_ic_dataset_default):
    """When metadata factors are absent, balance and diversity fields are None."""
    run_no_meta = _create_fake_bias_run(
        fake_ic_dataset_default,
        with_balance=False,
        with_diversity=False,
        num_factors=0,
    )

    records = run_no_meta.extract()
    assert len(records) == 1
    record = records[0]
    assert record.dataset_id == "fake_id_dataset"
    assert record.coverage_total == 20
    assert record.coverage_uncovered_count == 5
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
