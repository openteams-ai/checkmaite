import pytest

# Skip all tests in this file if reallabel isn't available
pytest.importorskip("reallabel")

# Module-level imports after importorskip to prevent collection errors
from checkmaite.ui.configuration_pages.object_detection.reallabel_app import RealLabelApp  # noqa: E402


@pytest.mark.parametrize(
    ("exp_iou_threshold", "exp_confidence_thresholds", "exp_class_agnostic", "exp_rwgt"),
    [(0.6, (0.3, 0.4), True, False), (0.9, (0.2, 0.8), False, True)],
    ids=["Agnostic_Not_Ground_Truth", "Ground_Truth_Not_Agnostic"],
)
@pytest.mark.unsupported
def test_run_export(
    exp_iou_threshold: float, exp_confidence_thresholds: tuple[float, float], exp_class_agnostic: bool, exp_rwgt: bool
) -> None:
    """Test RealLabelApp._run_export() with all possible similarity strategies."""
    # Arrange
    app = RealLabelApp()
    app.panel()  # test building the UI even though we can't see it here
    app.run_with_ground_truth = exp_rwgt
    app.iou_threshold = exp_iou_threshold
    app.likely_wrong_max_confidence = exp_confidence_thresholds[1]
    app.likely_missed_min_confidence = exp_confidence_thresholds[0]
    app.class_agnostic = exp_class_agnostic
    exp_output = {
        "TYPE": "RealLabelTestStage",
        "CONFIG": {
            "additional_outputs": [
                "sequence_priority_score",
                "sequence_priority_score_balanced",
                "classification_disagreements",
                "wanrs",
                "aggregated_confidence_df",
            ],
            "run_with_ground_truth": exp_rwgt,
            "deduplication_iou_threshold": exp_iou_threshold,
            "threshold_max_aggregated_confidence_fp": exp_confidence_thresholds[1],
            "threshold_min_aggregated_confidence_fn": exp_confidence_thresholds[0],
            "class_agnostic": exp_class_agnostic,
            "use_thresholds": True,
            "run_confidence_calibration": False,
            "column_names": {
                "calibrated_confidence_column": "score",
                "unique_identifier_columns": ["id"],
            },
        },
    }

    # Act
    app._run_export()
    output = app.output_test_stages["reallabel_test_stage"]

    # Assert
    assert output == exp_output
