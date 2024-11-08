# COPYRIGHTS AND PERMISSIONS:
# Copyright 2024 MORSECORP, Inc. All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Test RealLabelTestStage."""
import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pyspark.sql.functions as sf
import pytest
from maite.protocols import object_detection as od
from reallabel import ColumnNameConfig
from matplotlib.testing.compare import compare_images

from jatic_ri.object_detection.test_stages.impls.reallabel_test_stage import (
    Config,
    RealLabelTestStage,
    RealLabelCache,
    _REALLABEL_CACHE_CONFIGURATION_PATH,
    _REALLABEL_CACHE_CSV_PATH,
    _REALLABEL_CACHE_IMAGE_PATH,
)

from tests.testing_utilities.testing_utilities import assert_spark_dataframes_equal
from tests.testing_utilities.example_maite_objects import (  # noqa: E501
    FMOWDetectionDataset,
    USA_SUMMER_DATA_IMAGERY_DIR,
    USA_SUMMER_DATA_METADATA_FILE_PATH,
    Yolov5sModel,
    YOLOV5S_USA_ALL_SEASONS_V1_MODEL_PATH,
)

EXPECTED_REALLABEL_IMAGE = Path(os.path.abspath(__file__)).parent / "reallabel_survivor_shared_data" / "expected_reallabel_output.png"

CACHE_DIR = Path(os.path.abspath(__file__)).parent / ".tscache"


@pytest.fixture(scope="session")
def reallabel_test_stage_args() -> dict[str, Any]:
    """Default arguments for RealLabelTestStage."""
    yolov5s_all_v1_dev_model: od.Model = Yolov5sModel(
        model_path=str(YOLOV5S_USA_ALL_SEASONS_V1_MODEL_PATH),
        transforms=None,
        device="cpu",
    )
    model_dict = {
        "yolov5s_all_v1_dev_model": yolov5s_all_v1_dev_model,
    }
    detection_dataset: od.Dataset = FMOWDetectionDataset(
        USA_SUMMER_DATA_IMAGERY_DIR,
        USA_SUMMER_DATA_METADATA_FILE_PATH,
    )

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
        "config": config,
        "dataset": detection_dataset,
        "models": model_dict,
        "dict_config": dict_config,
    }


@pytest.fixture
def test_stage(reallabel_test_stage_args: dict, request: pytest.FixtureRequest) -> RealLabelTestStage:
    """Default fully initialized test_stage object"""
    # Create and yield test_stage
    test_stage = RealLabelTestStage(config=reallabel_test_stage_args[getattr(request, "param", "config")])
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
    ["config", "dict_config"],
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
    actual_returned_results_df = test_stage.outputs[0].withColumn("classification", sf.col("classification").cast("integer"))
    assert_spark_dataframes_equal(actual_returned_results_df, actual_cached_results_df.toPandas())

    # Compare the read-from-cache image against the actual image returned from `run()`
    assert expected_results_png_path.exists()
    compare_images(
        str(test_stage.outputs[1]), str(actual_cached_image), 0.001
    )

    # Further compare the image against what we expect the image to look like from a snapshot
    compare_images(
        str(test_stage.outputs[1]), str(EXPECTED_REALLABEL_IMAGE), 0.001
    )

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
    expected_cache_id = "reallabel_cache_51d92b0b198aae1e49ea669370b70d3caf05e8ac3d7eb1242ee35f8ad341661a"

    # Act
    actual_cache_id = test_stage.cache_id

    # Assert
    assert actual_cache_id == expected_cache_id


def test_reallabel_test_stage_collect_report_consumables(
    test_stage: RealLabelTestStage,
) -> None:
    """Test collect_report_consumables with cached data enabled."""
    # Arrange
    expected_deck = "object_detection_dataset_evaluation"
    expected_layout_name = "TwoImageTextNoHeader"
    expected_content_left = (
        '{"fontsize": 22}'
        "**Description**\n"
        "* RealLabel aids re-labeling efforts by using model ensembling to determine if a label is a:\n"
        "* True Positive Label: probably correct label.\n"
        "* False Positive Label: potentially incorrect label.\n"
        "* False Negative Label: potentially missing label.\n"
        "* In an example subset of the data, RealLabel has found 4 True Positive, "
        "2 False Positive, and 2 False Negative labels.\n"
        "Displayed is an example of a True Positive label."
    )
    expected_content_right = f"{test_stage.cache_base_path}/{test_stage.cache_id}/{_REALLABEL_CACHE_IMAGE_PATH}"
    expected_title = "RealLabel Label Breakdown"

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
    assert (
        output_consumables["layout_arguments"]["content_left"] == expected_content_left
    )
    assert (
        output_consumables["layout_arguments"]["content_right"].as_posix()
        == expected_content_right
    )


def test_reallabel_test_stage_collect_report_consumables_error(
    test_stage: RealLabelTestStage,
) -> None:
    """Test collect_report_consumables error when run not called."""
    # No Arrange

    # Act and Assert
    with pytest.raises(RuntimeError):
        test_stage.collect_report_consumables()


def test_reallabel_test_stage_collect_metrics_cached_data(
    test_stage: RealLabelTestStage,
) -> None:
    """Test collect_metrics."""
    # Arrange
    expected_output = {"NUM_Re-Label": 4}

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
    with pytest.raises(RuntimeError):
        test_stage.collect_metrics()


def test_reallabel_test_stage_run_errors(reallabel_test_stage_args: dict):
    """Test run() errors."""
    # Arrange
    test_stage_1 = RealLabelTestStage(reallabel_test_stage_args["config"])
    test_stage_2 = RealLabelTestStage(reallabel_test_stage_args["config"])
    test_stage_2.load_dataset(reallabel_test_stage_args["dataset"], "test-id")

    # Act and Assert
    with pytest.raises(RuntimeError, match=r"Dataset not set!.*"):
        test_stage_1.run(use_cache=False)

    # Act and Assert
    with pytest.raises(RuntimeError, match=r"Models not set!.*"):
        test_stage_2.run(use_cache=False)
