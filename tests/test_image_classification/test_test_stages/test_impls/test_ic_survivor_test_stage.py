"""Test image classification survivor test stage."""

from typing import Any

import pytest

# Skip all tests in this file if survivor isn't available
pytest.importorskip("survivor")

# Module-level imports after importorskip to prevent collection errors
import pandas as pd  # noqa: E402
from gradient import SubText, Text  # noqa: E402
from gradient.templates_and_layouts.create_deck import create_deck  # noqa: E402
from maite.protocols import image_classification as ic  # noqa: E402
from survivor.enums import ScoreConversionType  # noqa: E402

from jatic_ri.image_classification.test_stages import SurvivorConfig, SurvivorTestStage  # noqa: E402
from tests.fake_ic_classes import FakeICDataset, FakeICMetric, FakeICModel  # noqa: E402

_DICT_CONFIG = "dict_config"
_SURVIVOR_CONFIG = "config"


@pytest.fixture(scope="session")
def survivor_test_stage_args(
    fake_ic_model_default: FakeICModel, fake_ic_dataset_default: FakeICDataset, fake_ic_metric_default: FakeICMetric
) -> dict[str, Any]:
    """Default arguments for RealLabelTestStage."""
    fake_model: ic.Model = fake_ic_model_default
    model_dict: dict[str, FakeICModel] = {
        "fake_model": fake_model,
    }
    detection_dataset: ic.Dataset = fake_ic_dataset_default
    map_metric: ic.Metric = fake_ic_metric_default

    config = SurvivorConfig(
        metric_column="fake_metric",
        otb_threshold=0.9,
        easy_hard_threshold=0.5,
        conversion_type=ScoreConversionType.ROUNDED.value,
        conversion_args={"decimals_to_round": 2},
        heatmap_plot_columns=None,
    )

    dict_config = {
        "metric_column": "fake_metric",
        "otb_threshold": 0.9,
        "easy_hard_threshold": 0.5,
        "conversion_type": ScoreConversionType.ROUNDED.value,
        "conversion_args": {"decimals_to_round": 2},
        "heatmap_plot_columns": None,
    }

    return {
        _SURVIVOR_CONFIG: config,
        "dataset": detection_dataset,
        "metric": map_metric,
        "models": model_dict,
        _DICT_CONFIG: dict_config,
    }


@pytest.fixture(name="test_stage")
def create_test_stage(survivor_test_stage_args: dict, request: pytest.FixtureRequest):
    """Create a SurvivorTestStage object and load in all required args.

    Can load in both the `dict_config` and `config` configurations in `survivor_test_stage_args` depending on the
    string input to `request.param` (set through indirect parametrization of `test_stage`, defaults to `config`).
    """
    # Create and configure SurvivorTestStage
    return SurvivorTestStage(config=survivor_test_stage_args[getattr(request, "param", _SURVIVOR_CONFIG)])


@pytest.mark.parametrize(
    "test_stage",
    [_SURVIVOR_CONFIG, _DICT_CONFIG],
    ids=["Using SurvivorConfig", "Using dict config"],
    indirect=True,
)
def test_survivor_test_stage_run_caches(
    mocker, test_stage: SurvivorTestStage, survivor_test_stage_args, tmp_cache_path
) -> None:
    """Test SurvivorTestStage generates a cache object that can be read correctly."""

    models = list(survivor_test_stage_args["models"].values())
    datasets = [survivor_test_stage_args["dataset"]]
    metrics = [survivor_test_stage_args["metric"]]

    run = test_stage.run(use_stage_cache=True, models=models, metrics=metrics, datasets=datasets)

    mocker.patch.object(test_stage, "_run", side_effect=AssertionError("_run() called while cache hit was expected"))
    cached_run = test_stage.run(use_stage_cache=True, models=models, metrics=metrics, datasets=datasets)
    assert cached_run is not run

    reallabel_outputs = run.outputs
    cached_realabel_outputs = cached_run.outputs

    pd.testing.assert_frame_equal(cached_realabel_outputs.raw_output_df, reallabel_outputs.raw_output_df)
    pd.testing.assert_frame_equal(
        cached_realabel_outputs.metrics_with_survivor_label_df, reallabel_outputs.metrics_with_survivor_label_df
    )


def test_survivor_collect_report_consumables(
    test_stage: SurvivorTestStage, survivor_test_stage_args, artifact_dir
) -> None:
    """Test collect_report_consumables."""
    # Arrange
    expected_deck = "image_classification_survivor"
    expected_layout_name = "TwoItem"
    expected_content_left = Text(
        content=[
            SubText("Types of Data\n", bold=True),
            "• Easy: Models score the same and perform well.\n"
            "• Hard: Models score the same and perform poorly.\n"
            "• On the Bubble: Models score differently.\n\n"
            "• Ideally, a dataset would be primarily On the Bubble, so all data is helping distinguish between model "
            "performance.\n\n"
            "• This dataset had 0.0% Easy, 100.0% Hard, and "
            "0.0% On the Bubble data.",
        ],
        fontsize=22,
    )
    expected_title = "Survivor Dataset Breakdown"

    models = list(survivor_test_stage_args["models"].values())
    datasets = [survivor_test_stage_args["dataset"]]
    metrics = [survivor_test_stage_args["metric"]]

    # Run test stage once to ensure cache is present
    test_stage.run(use_stage_cache=True, models=models, metrics=metrics, datasets=datasets)

    # Run again to use cache
    test_stage.run(use_stage_cache=True, models=models, metrics=metrics, datasets=datasets)

    # Act
    slide_content = test_stage.collect_report_consumables()
    output_consumables = test_stage.collect_report_consumables()[0]

    # Assert
    assert output_consumables["deck"] == expected_deck
    assert output_consumables["layout_name"] == expected_layout_name
    assert output_consumables["layout_arguments"]["title"] == expected_title
    assert output_consumables["layout_arguments"]["left_item"].content == expected_content_left.content
    assert output_consumables["layout_arguments"]["right_item"].is_file()

    filename = create_deck(slide_content, artifact_dir, "survivor")
    assert filename.exists()


def test_survivor_test_stage_collect_report_consumables_error(
    test_stage: SurvivorTestStage,
) -> None:
    """Test collect_report_consumables error when run not called."""
    # No Arrange

    # Act and Assert
    with pytest.raises(RuntimeError, match="TestStage must be run before accessing outputs"):
        test_stage.collect_report_consumables()
