"""Test baseline evalutation"""

import os

import pandas as pd

from jatic_ri.object_detection.test_stages.impls.dataeval_drift_test_stage import (
    DatasetDriftTestStage,
)


class TestDatasetDriftTestStage:
    """Tests DatasetDriftTestStage correctly handles caching, results, and gradient consumables"""

    def test_drift(self, dummy_dataset_od) -> None:
        """Test DataEval implementation"""

        dev_dataset = dummy_dataset_od
        op_dataset = dummy_dataset_od
        op_dataset.images *= 0.5

        stage = DatasetDriftTestStage()
        stage.load_datasets(dataset_1=dev_dataset, dataset_2=op_dataset, dataset_1_id="dev", dataset_2_id="op")
        stage.run(use_cache=False)
        report = stage.collect_report_consumables()

        assert report
        assert len(report) == 1  # Drift results are bundled into one slide

        drift_args = report[0]

        # Gradient requires these 3 keys
        assert all(required_key in drift_args for required_key in ("deck", "layout_name", "layout_arguments"))

        # Only calculated data is checked. Text information is arbitrary
        drift_df: pd.DataFrame = drift_args["layout_arguments"]["table"]

        assert len(drift_df) == 3
        assert all(drift_df.columns == ["Method", "Has drifted?", "Test statistic", "P-value"])

    def test_drift_cache(self, dummy_dataset_od, tmp_path) -> None:
        """Tests outputs is saved into a file"""

        stage = DatasetDriftTestStage()
        stage.cache_base_path = tmp_path
        stage.load_datasets(
            dataset_1=dummy_dataset_od,
            dataset_2=dummy_dataset_od,
            dataset_1_id="dev",
            dataset_2_id="op",
        )
        stage.run()

        assert os.path.exists(stage.cache_path)

    def test_drift_no_outputs(self) -> None:
        """Tests no consumable is generated if run is not called"""

        assert DatasetDriftTestStage().collect_report_consumables() == []
