# # COPYRIGHTS AND PERMISSIONS:
# # Copyright 2024 MORSECORP, Inc. All rights reserved.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
# """Test survivor test stage."""
import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
import pytest
from maite.protocols import object_detection as od
from pyspark.sql import functions as sf
from survivor.config import ScoreConversionType
from matplotlib.testing.compare import compare_images

from jatic_ri.object_detection.test_stages.impls.survivor_test_stage import (
    SurvivorConfig,
    SurvivorCache,
    SurvivorTestStage,
    _SURVIVOR_CACHE_CONFIGURATION_PATH,
    _SURVIVOR_CACHE_CSV_PATH,
    _SURVIVOR_CACHE_IMAGE_PATH,
)

from tests.testing_utilities.testing_utilities import assert_spark_dataframes_equal
from tests.testing_utilities.example_maite_objects import (  # noqa: E501
    create_maite_wrapped_metric,
    FMOWDetectionDataset,
    USA_SUMMER_DATA_IMAGERY_DIR,
    USA_SUMMER_DATA_METADATA_FILE_PATH,
    Yolov5sModel,
    YOLOV5S_USA_ALL_SEASONS_V1_MODEL_PATH,
    YOLOV5S_USA_RUS_ALL_SEASONS_V1_MODEL_PATH,
)

EXPECTED_SURVIVOR_IMAGE = Path(os.path.abspath(__file__)).parent / "reallabel_survivor_shared_data" / "expected_survivor_output.png"

CACHE_DIR = Path(os.path.abspath(__file__)).parent / ".tscache"


@pytest.fixture(scope="session")
def survivor_test_stage_args() -> dict[str, Any]:
    """Default arguments for RealLabelTestStage."""
    yolov5s_all_v1_dev_model: od.Model = Yolov5sModel(
        model_path=str(YOLOV5S_USA_ALL_SEASONS_V1_MODEL_PATH),
        transforms=None,
        device="cpu",
    )
    yolov5s_all_v2_dev_model: od.Model = Yolov5sModel(
        model_path=str(YOLOV5S_USA_RUS_ALL_SEASONS_V1_MODEL_PATH),
        transforms=None,
        device="cpu",
    )
    model_dict = {
        "yolov5s_all_v1_dev_model": yolov5s_all_v1_dev_model,
        "yolov5s_all_v2_dev_model": yolov5s_all_v2_dev_model,
    }
    detection_dataset: od.Dataset = FMOWDetectionDataset(
        USA_SUMMER_DATA_IMAGERY_DIR, USA_SUMMER_DATA_METADATA_FILE_PATH,
    )
    map_metric: od.Metric = create_maite_wrapped_metric("mAP_50")

    config = SurvivorConfig(
        unique_identifier_columns=["image_id"],
        metric_column="map_50",
        otb_threshold=0.9,
        difficulty_threshold=0.5,
        conversion_type=ScoreConversionType.ROUNDED,
        conversion_args={"decimals_to_round": 2},
    )

    return {
        "config": config,
        "dataset": detection_dataset,
        "metric": map_metric,
        "models": model_dict,
    }


@pytest.fixture
def test_stage(survivor_test_stage_args: dict) -> SurvivorTestStage:
    # Ensure cache doesn't exist
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(CACHE_DIR)

    # Create and configure SurvivorTestStage
    test_stage = SurvivorTestStage(config=survivor_test_stage_args["config"])
    test_stage.load_models(models=survivor_test_stage_args["models"])
    test_stage.load_dataset(dataset=survivor_test_stage_args["dataset"], dataset_id="test-dataset")
    test_stage.load_metric(
        metric=survivor_test_stage_args["metric"],
        metric_id=survivor_test_stage_args["config"].metric_column
    )

    test_stage.cache_base_path = CACHE_DIR

    yield test_stage

    # Cleanup the cache once test is finished running
    if test_stage.cache_base_path.exists():
        shutil.rmtree(test_stage.cache_base_path)


def test_survivor_test_stage_run_caches(test_stage: SurvivorTestStage) -> None:
    """Test RealLabelTestStage generates a cache object that can be read correctly."""

    # Arrange
    expected_cache_location = Path(test_stage.cache_base_path) / test_stage.cache_id
    expected_results_df_path = expected_cache_location / _SURVIVOR_CACHE_CSV_PATH
    expected_results_png_path = expected_cache_location / _SURVIVOR_CACHE_IMAGE_PATH
    expected_results_config_path = expected_cache_location / _SURVIVOR_CACHE_CONFIGURATION_PATH

    survivor_cache = SurvivorCache()

    # test cache should not exist but wipe it if it does so we have a clean slate
    if Path(test_stage.cache_base_path).exists():
        shutil.rmtree(test_stage.cache_base_path)

    # Act - Build the cache
    test_stage.run(use_cache=True)

    actual_cached_results_df, actual_cached_image = survivor_cache.read_cache(cache_path=str(expected_cache_location))

    # Assert

    # Compare the read-from-cache dataframe against the actual dataframe returned from `run()`. Minor issues in type
    # conversion but can't really be helped :/
    assert expected_results_df_path.exists()
    actual_returned_results_df = test_stage.outputs[0].withColumn("timestamp", sf.col("timestamp").cast("timestamp"))
    assert_spark_dataframes_equal(actual_returned_results_df, actual_cached_results_df.toPandas())

    # Compare the read-from-cache image against the actual image returned from `run()`
    assert expected_results_png_path.exists()
    compare_images(
        str(test_stage.outputs[1]), str(actual_cached_image), 0.001
    )
    # Further compare the image against what we expect the image to look like from a snapshot
    compare_images(
        str(test_stage.outputs[1]), str(EXPECTED_SURVIVOR_IMAGE), 0.001
    )

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
    expected_cache_id = "survivor_cache_4cf267d67853e467cc87e816cccc1acb5e02ddfef62eef7e6ae9c0e145181b81"

    # Act
    actual_cache_id = test_stage.cache_id

    # Assert
    assert actual_cache_id == expected_cache_id


def test_survivor_collect_report_consumables(
    test_stage: SurvivorTestStage,
) -> None:
    """Test collect_report_consumables."""
    # Arrange
    expected_deck = "object_detection_dataset_evaluation"
    expected_layout_name = "TwoImageTextNoHeader"
    expected_content_left = (
        '{"fontsize": 22}'
        "**Types of Data**\n"
        "* Easy: Models score the same and perform well.\n"
        "* Hard: Models score the same and perform poorly.\n"
        "* On the Bubble: Models score differently.\n\n"
        "* Ideally, a dataset would be primarily On the Bubble, so all data is helping distinguish between model "
        "performance.\n\n"
        "* This dataset had 16.7% Easy, 50.0% Hard, and "
        "33.3% On the Bubble data."
    )
    expected_content_right = f"{test_stage.cache_base_path}/{test_stage.cache_id}/{_SURVIVOR_CACHE_IMAGE_PATH}"
    expected_title = "Survivor Dataset Breakdown"

    # Run test stage once to ensure cache is present
    test_stage.run(use_cache=True)

    # Run again to use cache
    test_stage.run(use_cache=True)

    # Act
    output_consumables = test_stage.collect_report_consumables()[0]

    # Assert
    assert output_consumables["deck"] == expected_deck
    assert output_consumables["layout_name"] == expected_layout_name
    assert output_consumables["layout_arguments"]["title"] == expected_title
    assert output_consumables["layout_arguments"]["content_left"] == expected_content_left
    assert output_consumables["layout_arguments"]["content_right"].as_posix() == expected_content_right


def test_survivor_test_stage_collect_report_consumables_error(
    test_stage: SurvivorTestStage,
) -> None:
    """Test collect_report_consumables error when run not called."""
    # No Arrange

    # Act and Assert
    with pytest.raises(RuntimeError):
        test_stage.collect_report_consumables()


def test_survivor_test_stage_collect_metrics(
    test_stage: SurvivorTestStage,
) -> None:
    """Test collect_metrics."""
    # Arrange
    expected_output = {"Low_Val_Data": 1.0 - (1 / 3)}

    test_stage.run()

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
    with pytest.raises(RuntimeError):
        test_stage.collect_metrics()


def test_survivor_test_stage_run_errors(survivor_test_stage_args: dict):
    """Test run() errors."""
    # Arrange
    test_stage_1 = SurvivorTestStage(survivor_test_stage_args["config"])
    test_stage_2 = SurvivorTestStage(survivor_test_stage_args["config"])
    test_stage_3 = SurvivorTestStage(survivor_test_stage_args["config"])

    test_stage_2.load_metric(survivor_test_stage_args["metric"], "test-metric")

    test_stage_3.load_metric(survivor_test_stage_args["metric"], "test-metric")
    test_stage_3.load_dataset(survivor_test_stage_args["dataset"], "test-id")

    # Act and Assert
    with pytest.raises(RuntimeError, match=r"Metric not set!.*"):
        test_stage_1.run(use_cache=False)

    with pytest.raises(RuntimeError, match="Dataset not set!.*"):
        test_stage_2.run(use_cache=False)

    with pytest.raises(RuntimeError, match="Models not set!.*"):
        test_stage_3.run(use_cache=False)
