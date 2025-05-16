from typing import Any, Optional, Union

import panel as pn
import pytest

from jatic_ri.object_detection._panel.configurations.survivor_app import SurvivorApp


@pytest.mark.parametrize(
    ("strategy", "exp_type", "exp_value", "exp_description", "is_widget"),
    [
        (
            "Binned",
            pn.widgets.TextInput,
            "0, 0.25, 0.5, 1.0",
            "Edges of the bins to sort model metrics into. Should all be within metric range i.e (0-1), (1-100)",
            True,
        ),
        ("Rounded", pn.widgets.IntInput, 2, "Number of decimal places to round model metrics to.", True),
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
        (".23", [0, 0.25, 0.5, 1.0], True),
    ],
    ids=[
        "Zero",
        "Large_Number",
        "No_Spaces",
        "Single_Float",
        "Floats_And_Ints",
        "Hanging_Decimal_Points",
        "Multiple_Decimal_points",
        "Single_Word",
        "Word_And_Numbers",
        "Leading_Decimal",
    ],
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
    assert app.status_source.current_value == ("Invalid Bins argument" if exp_fail else "Waiting for input...")


@pytest.mark.parametrize(
    ("similarity_strategy", "bins", "exp_conversion_type", "exp_conversion_args"),
    [
        ["Exact", "1", "original", {}],
        ["Binned", "1, 2, 3", "binned", {"conversion_args": {"bin_edges": [1, 2, 3]}}],
        ["Rounded", "2", "rounded", {"conversion_args": {"decimals_to_round": 9}}],
    ],
    ids=["Exact", "Binned", "Rounded"],
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
    app.easy_hard_threshold = 0.9
    exp_output = {
        "TYPE": "SurvivorTestStage",
        "CONFIG": {
            "metric_column": "metric",
            "conversion_type": exp_conversion_type,
            "otb_threshold": app.otb_threshold,
            "easy_hard_threshold": app.easy_hard_threshold,
        }
        | exp_conversion_args,
    }

    # Act
    app._run_export()
    output = app.output_test_stages["survivor_test_stage"]

    # Assert
    assert output == exp_output


def test_roundtrip() -> None:
    from jatic_ri.object_detection.test_stages.impls.survivor_test_stage import SurvivorTestStage

    app = SurvivorApp()
    app.panel()  # test constructing the UI even though we can't see it
    app._run_export()
    output_config = app.output_test_stages["survivor_test_stage"]["CONFIG"]
    SurvivorTestStage(config=output_config)
