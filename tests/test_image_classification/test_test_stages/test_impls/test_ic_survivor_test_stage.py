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

    return {
        "config": config,
        "dataset": detection_dataset,
        "metric": map_metric,
        "models": model_dict,
    }


@pytest.mark.unsupported
def test_survivor_test_stage_run_caches(mocker, survivor_test_stage_args, tmp_cache_path) -> None:
    """Test SurvivorTestStage generates a cache object that can be read correctly."""

    test_stage = SurvivorTestStage()
    models = list(survivor_test_stage_args["models"].values())
    datasets = [survivor_test_stage_args["dataset"]]
    metrics = [survivor_test_stage_args["metric"]]
    config = survivor_test_stage_args["config"]

    run = test_stage.run(config=config, use_stage_cache=True, models=models, metrics=metrics, datasets=datasets)

    mocker.patch.object(test_stage, "_run", side_effect=AssertionError("_run() called while cache hit was expected"))
    cached_run = test_stage.run(config=config, use_stage_cache=True, models=models, metrics=metrics, datasets=datasets)
    assert cached_run is not run

    survivor_outputs = run.outputs
    cached_survivor_outputs = cached_run.outputs

    pd.testing.assert_frame_equal(cached_survivor_outputs.raw_output_df, survivor_outputs.raw_output_df)
    pd.testing.assert_frame_equal(
        cached_survivor_outputs.metrics_with_survivor_label_df, survivor_outputs.metrics_with_survivor_label_df
    )


def test_survivor_collect_report_consumables(survivor_test_stage_args, artifact_dir) -> None:
    """Test collect_report_consumables."""
    # Arrange
    expected_deck = "jatic_ri.image_classification.test_stages._impls.survivor_test_stage.SurvivorTestStage"
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

    test_stage = SurvivorTestStage()
    models = list(survivor_test_stage_args["models"].values())
    datasets = [survivor_test_stage_args["dataset"]]
    metrics = [survivor_test_stage_args["metric"]]
    config = survivor_test_stage_args["config"]

    # Run test stage once to ensure cache is present
    run = test_stage.run(config=config, use_stage_cache=True, models=models, metrics=metrics, datasets=datasets)

    # Run again to use cache
    _ = test_stage.run(config=config, use_stage_cache=True, models=models, metrics=metrics, datasets=datasets)

    # Act
    slide_content = run.collect_report_consumables(threshold=0.5)
    output_consumables = run.collect_report_consumables(threshold=0.5)[0]

    # Assert
    assert output_consumables["deck"] == expected_deck
    assert output_consumables["layout_name"] == expected_layout_name
    assert output_consumables["layout_arguments"]["title"] == expected_title
    assert output_consumables["layout_arguments"]["left_item"].content == expected_content_left.content
    assert output_consumables["layout_arguments"]["right_item"].is_file()

    filename = create_deck(slide_content, artifact_dir, "survivor")
    assert filename.exists()
