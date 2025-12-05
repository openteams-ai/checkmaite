import pytest

pytest.importorskip("reallabel")

from reallabel import ColumnNameConfig  # noqa: E402

from jatic_ri.core.object_detection.reallabel_labelling_capability import (  # noqa: E402
    ReallabelLabelling,
    ReallabelLabellingConfig,
)


@pytest.mark.unsupported
def test_run_and_collect(fake_od_dataset_reallabel_only, fake_od_model_default):
    capability = ReallabelLabelling()

    config = ReallabelLabellingConfig(
        deduplication_algorithm="wbf",
        column_names=ColumnNameConfig(
            unique_identifier_columns=["id"],
            calibrated_confidence_column="score",
        ),
        run_confidence_calibration=False,
        keep_likely_corrects=True,
        deduplication_iou_threshold=0.5,
        minimum_confidence_threshold=0.1,
        threshold_max_aggregated_confidence_fp=0.01,
    )

    output = capability.run(
        use_cache=True,
        models=[fake_od_model_default, fake_od_model_default],
        datasets=[fake_od_dataset_reallabel_only],
        config=config,
    )

    assert output.model_dump()  # smoke test

    assert output.collect_report_consumables(threshold=0.5)  # smoke test
