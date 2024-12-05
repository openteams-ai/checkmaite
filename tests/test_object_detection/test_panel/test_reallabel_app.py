from jatic_ri.object_detection._panel.configurations.reallabel_app import RealLabelApp
import panel as pn
import pytest


def test_settings_pane() -> None:
    """Test RealLabelApp.settings_pane() returns panel with widgets in correct order and with correct values."""
    # Arrange
    app = RealLabelApp()
    exp_types = [pn.param.ParamMethod, pn.widgets.FloatInput, pn.Row, pn.widgets.Switch, pn.Row,
                 pn.widgets.Switch, pn.widgets.FloatInput, pn.widgets.FloatInput]
    exp_values = [None, 0.5, None, True, None, False, 0.5, 0.5]

    # Act
    result = app.settings_pane()

    # Assert
    assert isinstance(result, pn.Column)
    assert result.width == app.page_width
    assert len(result.objects) == len(exp_types)
    for obj, obj_type, obj_value in zip(result.objects, exp_types, exp_values):
        assert isinstance(obj, obj_type)
        # ParamMethod and Row don't have value attributes, so if value == None, don't try these assert statements.
        if obj_value is not None:
            assert obj.value == obj_value


def test_panel() -> None:
    """Test RealLabelApp.panel() returns panel with widgets in correct order and with correct values."""
    # Arrange
    app = RealLabelApp()
    exp_styles = {"background": app.color_dark_blue}
    exp_types = [pn.param.ParamMethod, pn.param.ParamMethod]

    # Act
    result = app.panel()

    # Assert
    assert isinstance(result, pn.Column)
    assert result.width == app.page_width
    assert result.styles == exp_styles
    assert len(result.objects) == len(exp_types)
    for obj, obj_type in zip(result.objects, exp_types):
        assert isinstance(obj, obj_type)


@pytest.mark.parametrize(
    ("exp_iou_threshold", "exp_confidence_thresholds", "exp_class_agnostic", "exp_rwgt"),
    [
        (0.6, (0.3, 0.4), True, False),
        (0.9, (0.2, 0.8), False, True)
    ],
    ids=["Agnostic_Not_Ground_Truth", "Ground_Truth_Not_Agnostic"]
)
def test_run_export(
        exp_iou_threshold: float,
        exp_confidence_thresholds: tuple[float, float],
        exp_class_agnostic: bool,
        exp_rwgt: bool
) -> None:
    """Test RealLabelApp._run_export() with all possible similarity strategies."""
    # Arrange
    app = RealLabelApp()
    app.run_with_ground_truth = exp_rwgt
    app.iou_threshold = exp_iou_threshold
    app.likely_wrong_max_confidence = exp_confidence_thresholds[1]
    app.likely_missed_min_confidence = exp_confidence_thresholds[0]
    app.class_agnostic = exp_class_agnostic
    exp_output = {
        "TYPE": "RealLabelTestStage",
        "CONFIG": {
            "additional_outputs": [
                "results",
                "verbose_df",
                "classification_disagreements_df",
                "sequence_priority_score_df",
                "wanrs_df",
                "sequence_priority_score_balanced_df",
            ],
            "run_with_ground_truth": exp_rwgt,
            "deduplication_iou_threshold": exp_iou_threshold,
            "threshold_max_aggregated_confidence_fp": exp_confidence_thresholds[1],
            "threshold_min_aggregated_confidence_fn": exp_confidence_thresholds[0],
            "class_agnostic": exp_class_agnostic,
            "use_thresholds": True,
        }
    }

    # Act
    app._run_export()
    output = app.output_test_stages["reallabel_test_stage"]

    # Assert
    assert output == exp_output
