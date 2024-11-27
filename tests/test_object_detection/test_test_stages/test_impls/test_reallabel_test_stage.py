"""Test RealLabelTestStage."""

import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from gradient.templates_and_layouts.create_deck import create_deck
import pandas as pd
import pyspark.sql.functions as sf
import pytest
from gradient import Text
from matplotlib.testing.compare import compare_images
from reallabel import ColumnNameConfig

from jatic_ri._common.test_stages.interfaces.test_stage import RIValidationError
from jatic_ri.object_detection.test_stages.impls.reallabel_test_stage import (
    _REALLABEL_CACHE_CONFIGURATION_PATH,
    _REALLABEL_CACHE_CSV_PATH,
    _REALLABEL_CACHE_IMAGE_PATH,
    Config,
    RealLabelCache,
    RealLabelTestStage,
)
from tests.testing_utilities.testing_utilities import (
    assert_spark_dataframes_equal,
    minimal_maite_object_detection_dataset_and_model,
)

# This file is the expected output of RealLabel if using all the information found in the survivor_test_stage_args
# fixture, and if any of the data, model, metric, or RealLabelConfig information used by that fixture changes,
# then this file will need to be updated
EXPECTED_REALLABEL_IMAGE = (
    Path(os.path.abspath(__file__)).parent / "reallabel_survivor_shared_data" / "expected_reallabel_output.png"
)

CACHE_DIR = Path(os.path.abspath(__file__)).parent / ".tscache"
_DICT_CONFIG = "dict_config"
_REALLABEL_CONFIG = "config"


@pytest.fixture(scope="session")
def reallabel_test_stage_args() -> dict[str, Any]:
    """Default arguments for RealLabelTestStage."""

    # a single image with groundtruth along with two models with identical predictions
    reallabel_test_stage_dataset, model = minimal_maite_object_detection_dataset_and_model(dataset_length=1)
    reallabel_test_stage_models = {"model_1": model, "model_2": model}

    config = Config(
        deduplication_algorithm="wbf",
        column_names=ColumnNameConfig(
            unique_identifier_columns=["img_filename"],
            calibrated_confidence_column="score",
        ),
        run_confidence_calibration=False,
        keep_true_positives=True,
        deduplication_iou_threshold=0.5,
        minimum_confidence_threshold=0.1,
        threshold_max_aggregated_confidence_fp=0.01,
    )

    dict_config = {
        "deduplication_algorithm": "wbf",
        "column_names": ColumnNameConfig(
            unique_identifier_columns=["img_filename"],
            calibrated_confidence_column="score",
        ),
        "run_confidence_calibration": False,
        "keep_true_positives": True,
        "deduplication_iou_threshold": 0.5,
        "minimum_confidence_threshold": 0.1,
        "threshold_max_aggregated_confidence_fp": 0.01,
    }

    return {
        _REALLABEL_CONFIG: config,
        "dataset": reallabel_test_stage_dataset,
        "models": reallabel_test_stage_models,
        _DICT_CONFIG: dict_config,
    }


@pytest.fixture(name="test_stage")
def create_test_stage(reallabel_test_stage_args: dict, request: pytest.FixtureRequest) -> RealLabelTestStage:
    """Create a RealLabelTestStage object and load in all required args.

    Can load in both the `dict_config` and `config` configurations in `reallabel_test_stage_args` depending on the
    string input to `request.param` (set through indirect parametrization of `test_stage`).
    """

    # Create and yield test_stage
    test_stage = RealLabelTestStage(
        config=reallabel_test_stage_args[getattr(request, "param", _REALLABEL_CONFIG)]
    )
    test_stage.load_models(models=reallabel_test_stage_args["models"])
    test_stage.load_dataset(
        dataset=reallabel_test_stage_args["dataset"],
        dataset_id="test-dataset",
    )

    test_stage.cache_base_path = CACHE_DIR

    # Ensure cache doesn't exist
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(test_stage.cache_base_path)

    yield test_stage

    # Cleanup the cache once test is finished running
    if test_stage.cache_base_path.exists():
        shutil.rmtree(test_stage.cache_base_path)


@pytest.mark.parametrize(
    "test_stage",
    [_REALLABEL_CONFIG, _DICT_CONFIG],
    ids=["Using RealLabelConfig", "Using dict config"],
    indirect=True,
)
def test_reallabel_test_stage_run_caches(test_stage: RealLabelTestStage) -> None:
    """Test RealLabelTestStage generates a cache object correctly."""
    # Arrange
    expected_cache_location = Path(test_stage.cache_base_path) / test_stage.cache_id

    expected_results_df_path = expected_cache_location / _REALLABEL_CACHE_CSV_PATH
    expected_results_png_path = expected_cache_location / _REALLABEL_CACHE_IMAGE_PATH
    expected_results_config_path = expected_cache_location / _REALLABEL_CACHE_CONFIGURATION_PATH

    reallabel_cache = RealLabelCache()

    # Act
    test_stage.run(use_cache=True)

    actual_cached_results_df, actual_cached_image = reallabel_cache.read_cache(cache_path=str(expected_cache_location))

    # Compare the read-from-cache dataframe against the actual dataframe returned from `run()`. Minor issues in type
    # conversion but can't really be helped :/
    assert expected_results_df_path.exists()
    actual_returned_results_df = test_stage.outputs[0].withColumn(
        "classification", sf.col("classification").cast("integer")
    )
    assert_spark_dataframes_equal(actual_returned_results_df, actual_cached_results_df.toPandas())

    # Compare the read-from-cache image against the actual image returned from `run()`
    assert expected_results_png_path.exists()
    compare_images(str(test_stage.outputs[1]), str(actual_cached_image), 0.001)

    # Further compare the image against what we expect the image to look like from a snapshot
    compare_images(str(test_stage.outputs[1]), str(EXPECTED_REALLABEL_IMAGE), 0.001)

    # Manually check that the cache config was saved properly well since the config isn't returned by read_cache()
    assert expected_results_config_path.exists()
    with expected_results_config_path.open() as file:
        assert test_stage._cache_configuration == json.load(file)


def test_reallabel_test_stage_cache_id_generation(test_stage) -> None:
    """Test the RealLabelTestStage cache ID generation against the known ID from the current base test set.

    If the model IDs, Dataset ID, or anything about the RealLabelConfig object from the reallabel_test_stage_args
    fixture changes, then the hash in the expected_cache_id variable will need to be updated.

    This will also need to be updated upon the resolution of the issue that requires Aggregated Confidence to
    """
    # Arrange
    expected_cache_id = "reallabel_od_cache_bd5db062294686177bb33d9f2ab32f47b9e68ebdd4b4fb4e5b949f6fdd03552c"

    # Act
    actual_cache_id = test_stage.cache_id

    # Assert
    assert actual_cache_id == expected_cache_id


def test_reallabel_test_stage_collect_report_consumables(
    test_stage: RealLabelTestStage,
    artifact_dir,
) -> None:
    """Test collect_report_consumables with cached data enabled."""
    # Arrange
    expected_deck = "object_detection_reallabel"
    expected_layout_name = "TwoImageTextNoHeader"
    expected_content_left = Text(
        content="**Description**\n"
        "• RealLabel aids re-labeling efforts by using model ensembling to determine if a label is a:\n"
        "• True Positive Label: probably correct label.\n"
        "• False Positive Label: potentially incorrect label.\n"
        "• False Negative Label: potentially missing label.\n"
        "• In an example subset of the data, RealLabel has found 1 True Positive, "
        "2 False Positive, and 1 False Negative labels.\n"
        "Displayed is an example of a True Positive label.",
        fontsize=22,
    )
    expected_content_right = f"{test_stage.cache_base_path}/{test_stage.cache_id}/{_REALLABEL_CACHE_IMAGE_PATH}"
    expected_title = "RealLabel Label Breakdown"

    # Run test stage once to ensure cache is present
    test_stage.run(use_cache=True)

    # Run again to use cache
    test_stage.run(use_cache=True)

    # Act
    slides = test_stage.collect_report_consumables()
    output_consumables = slides[0]

    # Assert
    assert output_consumables["deck"] == expected_deck
    assert output_consumables["layout_name"] == expected_layout_name
    assert output_consumables["layout_arguments"]["title"] == expected_title
    assert output_consumables["layout_arguments"]["content_left"].content == expected_content_left.content
    assert output_consumables["layout_arguments"]["content_right"].as_posix() == expected_content_right

    filename = create_deck(slides, artifact_dir, 'reallabel')
    assert filename.exists()


def test_reallabel_test_stage_collect_report_consumables_error(
    test_stage: RealLabelTestStage,
) -> None:
    """Test collect_report_consumables error when run not called."""
    # No Arrange

    # Act and Assert
    with pytest.raises(RuntimeError, match="TestStage must be run before accessing outputs"):
        test_stage.collect_report_consumables()


def test_reallabel_test_stage_collect_metrics_cached_data(
    test_stage: RealLabelTestStage,
) -> None:
    """Test collect_metrics."""
    # Arrange
    expected_output = {"NUM_Re-Label": 3}

    test_stage.run()

    # Act
    actual_output = test_stage.collect_metrics()

    # Assert
    assert actual_output == expected_output


def test_reallabel_test_stage_collect_metrics_error(
    test_stage: RealLabelTestStage,
) -> None:
    """Test collect_metrics error when run not called."""
    # No Arrange

    # Act and Assert
    with pytest.raises(RuntimeError, match="TestStage must be run before accessing outputs"):
        test_stage.collect_metrics()


def test_reallabel_test_stage_run_errors(reallabel_test_stage_args: dict):
    """Test run() errors."""
    # Arrange
    test_stage_1 = RealLabelTestStage(reallabel_test_stage_args["config"])
    test_stage_2 = RealLabelTestStage(reallabel_test_stage_args["config"])
    test_stage_2.load_models(reallabel_test_stage_args["models"])

    # Act and Assert
    with pytest.raises(RIValidationError, match=r"'models' not set! Please use `load_models\(\)` function"):
        test_stage_1.run(use_cache=False)

    with pytest.raises(RIValidationError, match=r"'dataset' not set! Please use `load_dataset\(\)` function"):
        test_stage_2.run(use_cache=False)


def test_missing_cache_image_error(tmp_path: Path) -> None:
    """Test error is raised when there is data in the cache but no image."""
    # Arrange
    test_df = pd.DataFrame({"group_winner_box_coords": []})
    test_df.to_csv(tmp_path / "reallabel_standard_results.csv")
    cache = RealLabelCache()

    # Act and Assert
    with pytest.warns(
        UserWarning, match=f"RealLabel cache path {tmp_path} doesn't contain a cached result visualization!"
    ):
        cache.read_cache(str(tmp_path))


def test_cache_miss_dir_resets(test_stage: RealLabelTestStage, tmp_path) -> None:
    """Test cache miss dir is deleted and resets if it already exists."""
    # Arrange
    test_stage.cache_base_path = tmp_path
    output = tmp_path / "reallabel_cache_miss_outputs"
    output.mkdir()
    file = output / "test_file.txt"
    file.touch()

    # Act
    test_stage._run()

    # Assert
    assert output.exists()
    assert not file.exists()


def test_run_error_when_not_ic_dataset(test_stage: RealLabelTestStage) -> None:
    """Test error is raised when dataset doesn't have required attributes."""
    test_stage.load_dataset(dataset=[None], dataset_id="empty_dataset")  # type: ignore

    with pytest.raises(AttributeError, match="need a _dataset_path attribute in the Dataset object"):
        test_stage._run()
