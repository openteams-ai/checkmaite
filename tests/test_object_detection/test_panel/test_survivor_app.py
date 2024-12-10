from typing import Any, Optional, Union

from jatic_ri.object_detection._panel.configurations.survivor_app import SurvivorApp
import panel as pn
import pytest


@pytest.mark.parametrize(
    ("strategy", "exp_type", "exp_value", "exp_description", "is_widget"),
    [
        ("Binned", pn.widgets.TextInput, "0, 0.25, 0.5, 1.0",
         "Edges of the bins to sort model metrics into. Should all be within metric range i.e (0-1), (1-100)", True),
        ("Rounded", pn.widgets.IntInput, 2,
         "Number of decimal places to round model metrics to.", True),
        ("Exact", pn.Row, None, None, False),

    ],
    ids=["Binned", "Rounded", "Exact"],
)
def test_similarity_option_pane(
        strategy: str, exp_type: type, exp_value: Any, exp_description: str, is_widget: bool
) -> None:
    """Test SurvivorApp.similarity_option_pane() with all possible similarity_strategies."""
    # Arrange
    app = SurvivorApp()
    app.similarity_strategy = strategy

    # Act
    result = app.similarity_option_pane()

    # Assert
    assert isinstance(result, exp_type)
    if is_widget:
        assert result.value == exp_value
        assert result.description == exp_description
        assert result.width == app.widget_width
        assert result.stylesheets == [app.widget_stylesheet, app.info_button_style]
        assert result.margin == (0, 40)


@pytest.mark.parametrize(
    ("input_str", "exp_output", "exp_fail"),
    [
        ("0", 0, False),
        ("12321345", 12321345, False),
        ("1,2,3", [1, 2, 3], False),
        ("0.5", [0.5], False),
        ("1.2, 3.4, 12,45", [1.2, 3.4, 12, 45], False),
        ("1., 5., 6.", [1, 5, 6], False),
        ("12...3", [0, 0.25, 0.5, 1.0], True),
        ("Hi", [0, 0.25, 0.5, 1.0], True),
        ("12, 4, g, 89", [0, 0.25, 0.5, 1.0], True),
        (".23", [0, 0.25, 0.5, 1.0], True)
    ],
    ids=["Zero", "Large_Number", "No_Spaces", "Single_Float", "Floats_And_Ints", "Hanging_Decimal_Points",
         "Multiple_Decimal_points", "Single_Word", "Word_And_Numbers", "Leading_Decimal"]
)
def test_parse_bin_string(input_str: str, exp_output: Optional[Union[int, list[float]]], exp_fail: bool) -> None:
    """Test SurvivorApp.parse_bin_string() sets SurvivorApp._bins correctly."""
    # Arrange
    app = SurvivorApp()
    app.bins = input_str

    # Act
    app._parse_bin_string()

    # Assert
    assert app._bins == exp_output
    assert app.status_text == ("Invalid Bins argument" if exp_fail else "Waiting for input...")


def test_settings_pane() -> None:
    """Test SurvivorApp.settings_pane() returns panel with widgets in correct order and with correct values."""
    # Arrange
    app = SurvivorApp()
    exp_types = [pn.param.ParamMethod, pn.widgets.FloatInput, pn.widgets.FloatInput, pn.widgets.Select]
    exp_values = [None, 0.5, 0.5, "Exact"]
    exp_descriptions = [
        None,
        "Upper threshold of model agreement for data to be considered 'On the Bubble'.",
        "Threshold of model score for data to be considered 'Easy' or 'Hard'.",
        "Strategy to use to discretize model metrics."
    ]

    # Act
    result = app.settings_pane()

    # Assert
    assert isinstance(result, pn.Column)
    assert result.width == app.page_width
    assert len(result.objects) == len(exp_types)
    for obj, obj_type, obj_value, obj_desc in zip(result.objects, exp_types, exp_values, exp_descriptions):
        assert isinstance(obj, obj_type)
        # ParamMethod doesn't have these attributes, so if value == None, don't try these assert statements.
        if obj_value is not None:
            assert obj.value == obj_value
            assert obj.description == obj_desc
            assert obj.width == app.widget_width
            assert obj.stylesheets == [app.widget_stylesheet, app.info_button_style]


def test_panel() -> None:
    """Test SurvivorApp.panel() returns panel with widgets in correct order and with correct values."""
    # Arrange
    app = SurvivorApp()
    exp_styles = {"background": app.color_dark_blue}
    exp_types = [pn.param.ParamMethod, pn.param.ParamMethod, pn.param.ParamMethod]

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
    ("similarity_strategy", "bins", "exp_conversion_type", "exp_conversion_args"),
    [
        ["Exact", "1", "original", {}],
        ["Binned", "1, 2, 3", "binned", {"conversion_args": {"bin_edges": [1, 2, 3]}}],
        ["Rounded", "2", "rounded", {"conversion_args": {"decimals_to_round": 9}}],
    ],
    ids=["Exact", "Binned", "Rounded"]
)
def test_run_export(
        similarity_strategy: str,
        bins: str,
        exp_conversion_type: str,
        exp_conversion_args: dict[str, Any],

) -> None:
    """Test SurvivorApp._run_export() with all possible similarity strategies."""
    # Arrange
    app = SurvivorApp()
    app.similarity_strategy = similarity_strategy
    app.bins = bins
    app.round_precision = 9
    app.otb_threshold = 0.7
    app.difficulty_threshold = 0.9
    exp_output = {
        "TYPE": "SurvivorTestStage",
        "CONFIG": {
                      "metric_column": "metric",
                      "unique_identifier_columns": ["image_id"],
                      "conversion_type": exp_conversion_type,
                      "otb_threshold": app.otb_threshold,
                      "easy_hard_threshold": app.difficulty_threshold,
                  } | exp_conversion_args
    }

    # Act
    app._run_export()
    output = app.output_test_stages["survivor_test_stage"]

    # Assert
    assert output == exp_output

def test_roundtrip() -> None:
    from jatic_ri.object_detection.test_stages.impls.survivor_test_stage import SurvivorTestStage
    app = SurvivorApp()
    app._run_export()
    output_config = app.output_test_stages['survivor_test_stage']["CONFIG"]
    SurvivorTestStage(config=output_config)
