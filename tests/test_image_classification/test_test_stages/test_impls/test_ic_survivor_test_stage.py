"""Test image classification survivor test stage."""

import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from gradient import Text
from maite.protocols import image_classification as ic
from survivor.enums import ScoreConversionType

from jatic_ri._common.test_stages.impls.survivor_test_stage_cache import (
    _SURVIVOR_CACHE_CONFIGURATION_PATH,
    SurvivorCache,
)
from jatic_ri._common.test_stages.interfaces.test_stage import RIValidationError
from jatic_ri.image_classification.test_stages.impls.survivor_test_stage import SurvivorConfig, SurvivorTestStage
from tests.fake_ic_classes import FakeICDataset, FakeICMetric, FakeICModel
from tests.testing_utilities.testing_utilities import assert_spark_dataframes_equal

# This file is the expected output of Survivor if using all the information found in the survivor_test_stage_args
# fixture, and if any of the data, model, metric, or SurvivorConfig information used by that fixture changes,
# then this file will need to be updated

# KNOWN ISSUES:
# (1) Per the comment above, the expected survivor image file SHOULD need updating given fixtures have changed.  However,
# the tests still passed.  RI issue #232 is tracking
# (2) Higher fidelity tests (i.e. Survivor-relevant variance in the faked model/dataset/metric) are being solutioned - RI issue #229

EXPECTED_SURVIVOR_IMAGE = (
    Path(os.path.abspath(__file__)).parent / "test_data" / "expected_survivor_result_visualization.png"
)

CACHE_DIR = Path(os.path.abspath(__file__)).parent / ".tscache"
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
    )

    dict_config = {
        "metric_column": "fake_metric",
        "otb_threshold": 0.9,
        "easy_hard_threshold": 0.5,
        "conversion_type": ScoreConversionType.ROUNDED.value,
        "conversion_args": {"decimals_to_round": 2},
    }

    return {
        _SURVIVOR_CONFIG: config,
        "dataset": detection_dataset,
        "metric": map_metric,
        "models": model_dict,
        _DICT_CONFIG: dict_config,
    }


@pytest.fixture(name="test_stage")
def create_test_stage(survivor_test_stage_args: dict, request: pytest.FixtureRequest, default_eval_tool_no_cache):
    """Create a SurvivorTestStage object and load in all required args.

    Can load in both the `dict_config` and `config` configurations in `survivor_test_stage_args` depending on the
    string input to `request.param` (set through indirect parametrization of `test_stage`, defaults to `config`).
    """
    # Ensure cache doesn't exist
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(CACHE_DIR)

    # Create and configure SurvivorTestStage
    test_stage = SurvivorTestStage(config=survivor_test_stage_args[getattr(request, "param", _SURVIVOR_CONFIG)])
    test_stage.load_models(models=survivor_test_stage_args["models"])
    test_stage.load_dataset(dataset=survivor_test_stage_args["dataset"], dataset_id="test-dataset")
    test_stage.load_metric(
        metric=survivor_test_stage_args["metric"], metric_id=survivor_test_stage_args["config"].metric_column
    )
    test_stage.load_eval_tool(default_eval_tool_no_cache)

    test_stage.cache_base_path = CACHE_DIR

    yield test_stage

    # Cleanup the cache once test is finished running
    if test_stage.cache_base_path.exists():
        shutil.rmtree(test_stage.cache_base_path)


@pytest.mark.parametrize(
    "test_stage",
    [_SURVIVOR_CONFIG, _DICT_CONFIG],
    ids=["Using SurvivorConfig", "Using dict config"],
    indirect=True,
)
def test_survivor_test_stage_run_caches(test_stage: SurvivorTestStage) -> None:
    """Test SurvivorTestStage generates a cache object that can be read correctly."""
    # Arrange
    expected_cache_location = Path(test_stage.cache_base_path) / test_stage.cache_id
    expected_results_df_path = expected_cache_location / "survivor_standard_results.csv"
    expected_results_png_path = expected_cache_location / "survivor_result_visualization.png"
    expected_results_config_path = expected_cache_location / _SURVIVOR_CACHE_CONFIGURATION_PATH

    survivor_cache = SurvivorCache()

    # test cache should not exist but wipe it if it does so we have a clean slate
    if Path(test_stage.cache_base_path).exists():
        shutil.rmtree(test_stage.cache_base_path)

    # Act - Build the cache
    test_stage.run(use_stage_cache=True)

    actual_cached_results_df, actual_cached_image = survivor_cache.read_cache(cache_path=str(expected_cache_location))

    # Assert

    # Compare the read-from-cache dataframe against the actual dataframe returned from `run()`. Minor issues in type
    # conversion but can't really be helped :/
    assert expected_results_df_path.exists()
    # 'timestamp' column was part of the USA_SUMMER_DATASET in initial tests, but does not exist in our default FakeODDataset
    actual_returned_results_df = test_stage.outputs[
        0
    ]  # .withColumn("timestamp", sf.col("timestamp").cast("timestamp"))
    assert_spark_dataframes_equal(actual_returned_results_df, actual_cached_results_df.toPandas())

    assert expected_results_png_path.exists()

    # Manually check that the cache config was saved properly well since the config isn't returned by read_cache()
    assert expected_results_config_path.exists()
    with expected_results_config_path.open() as file:
        assert test_stage._cache_configuration == json.load(file)


def test_survivor_test_stage_cache_id_generation(test_stage) -> None:
    """Test the SurvivorLabelTestStage cache ID generation against the known ID from the current base test set.

    If the model IDs, Dataset ID, Metric ID, or anything about the SurvivorConfig object from the
    survivor_test_stage_args fixture changes, then the hash in the expected_cache_id variable will need to be updated.
    """
    # Arrange
    expected_cache_id = "survivor_ic_cache_b446f28ba3a4e506fcb68c5c691e92784cf805b411177b87235b189a0f7349e3"

    # Act
    actual_cache_id = test_stage.cache_id

    # Assert
    assert actual_cache_id == expected_cache_id


def test_survivor_collect_report_consumables(test_stage: SurvivorTestStage) -> None:
    """Test collect_report_consumables."""
    # Arrange
    expected_deck = "image_classification_survivor"
    expected_layout_name = "TwoImageTextNoHeader"
    expected_content_left = Text(
        content="**Types of Data**\n"
        "• Easy: Models score the same and perform well.\n"
        "• Hard: Models score the same and perform poorly.\n"
        "• On the Bubble: Models score differently.\n\n"
        "• Ideally, a dataset would be primarily On the Bubble, so all data is helping distinguish between model "
        "performance.\n\n"
        "• This dataset had 0.0% Easy, 100.0% Hard, and "
        "0.0% On the Bubble data.",
        fontsize=22,
    )
    expected_content_right = f"{test_stage.cache_base_path}/{test_stage.cache_id}/survivor_result_visualization.png"
    expected_title = "Survivor Dataset Breakdown"

    # Run test stage once to ensure cache is present
    test_stage.run(use_stage_cache=True)

    # Run again to use cache
    test_stage.run(use_stage_cache=True)

    # Act
    output_consumables = test_stage.collect_report_consumables()[0]

    # Assert
    assert output_consumables["deck"] == expected_deck
    assert output_consumables["layout_name"] == expected_layout_name
    assert output_consumables["layout_arguments"]["title"] == expected_title
    assert output_consumables["layout_arguments"]["content_left"].content == expected_content_left.content
    assert output_consumables["layout_arguments"]["content_right"].as_posix() == expected_content_right


def test_survivor_test_stage_collect_report_consumables_error(
    test_stage: SurvivorTestStage,
) -> None:
    """Test collect_report_consumables error when run not called."""
    # No Arrange

    # Act and Assert
    with pytest.raises(RuntimeError, match="TestStage must be run before accessing outputs"):
        test_stage.collect_report_consumables()


def test_survivor_test_stage_collect_metrics(
    test_stage: SurvivorTestStage,
) -> None:
    """Test collect_metrics."""
    # Arrange
    expected_output = {"Low_Val_Data": 1.0}

    test_stage.run(use_stage_cache=False)

    # Act
    actual_output = test_stage.collect_metrics()

    # Assert
    assert actual_output == expected_output


def test_survivor_test_stage_collect_metrics_error(
    test_stage: SurvivorTestStage,
) -> None:
    """Test collect_metrics error when run not called."""
    # No Arrange

    # Act and Assert
    with pytest.raises(RuntimeError, match="TestStage must be run before accessing outputs"):
        test_stage.collect_metrics()


def test_survivor_test_stage_run_errors(survivor_test_stage_args: dict):
    """Test run() errors."""
    # Arrange
    test_stage_1 = SurvivorTestStage(survivor_test_stage_args["config"])
    test_stage_2 = SurvivorTestStage(survivor_test_stage_args["config"])
    test_stage_3 = SurvivorTestStage(survivor_test_stage_args["config"])

    test_stage_2.load_models(survivor_test_stage_args["models"])

    test_stage_3.load_models(survivor_test_stage_args["models"])
    test_stage_3.load_dataset(survivor_test_stage_args["dataset"], "test-id")

    # Act and Assert
    with pytest.raises(RIValidationError, match=r"'models' not set! Please use `load_models\(\)` function"):
        test_stage_1.run(use_stage_cache=False)

    with pytest.raises(RIValidationError, match=r"'dataset' not set! Please use `load_dataset\(\)` function"):
        test_stage_2.run(use_stage_cache=False)

    with pytest.raises(RIValidationError, match=r"'metric' not set! Please use `load_metric\(\)` function"):
        test_stage_3.run(use_stage_cache=False)


def test_cache_miss_dir_resets(test_stage: SurvivorTestStage, tmp_path) -> None:
    """Test cache miss dir is deleted and resets if it already exists."""
    # Arrange
    test_stage.cache_base_path = tmp_path
    output = tmp_path / "survivor_cache_miss_outputs"
    output.mkdir()
    file = output / "test_file.txt"
    file.touch()

    # Act
    test_stage._run()

    # Assert
    assert output.exists()
    assert not file.exists()
