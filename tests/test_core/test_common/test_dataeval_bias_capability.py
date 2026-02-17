from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jatic_ri.core._common.dataeval_bias_capability import (
    DataevalBiasBalanceOutputs,
    DataevalBiasBase,
    DataevalBiasConfig,
    DataevalBiasCoverageOutputs,
    DataevalBiasDiversityOutputs,
    DataevalBiasOutputs,
    DataevalBiasRun,
)
from jatic_ri.core.object_detection.dataset_loaders import CocoDetectionDataset
from jatic_ri.core.report._gradient import HAS_GRADIENT


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
