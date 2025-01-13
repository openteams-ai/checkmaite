"""Test Object Detection Survivor test stage."""

import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from gradient import Text
from maite.protocols import object_detection as od
from matplotlib.testing.compare import compare_images
from gradient.templates_and_layouts.create_deck import create_deck
from survivor.config import ScoreConversionType

from jatic_ri._common.test_stages.impls.survivor_test_stage_cache import (
    _SURVIVOR_CACHE_CONFIGURATION_PATH,
    SurvivorCache,
)
from jatic_ri._common.test_stages.interfaces.test_stage import RIValidationError
from jatic_ri.object_detection.test_stages.impls.survivor_test_stage import (
    SurvivorConfig,
    SurvivorTestStage,
)
from tests.fake_od_classes import FakeODDataset, FakeODModel
from tests.testing_utilities.testing_utilities import (
    assert_spark_dataframes_equal
)

# This file is the expected output of Survivor if using all the information found in the survivor_test_stage_args
# fixture, and if any of the data, model, metric, or SurvivorConfig information used by that fixture changes,
# then this file will need to be updated
EXPECTED_SURVIVOR_IMAGE = (
    Path(os.path.abspath(__file__)).parent / "reallabel_survivor_shared_data" / "expected_survivor_output.png"
)

CACHE_DIR = Path(os.path.abspath(__file__)).parent / ".tscache"
_DICT_CONFIG = "dict_config"
_SURVIVOR_CONFIG = "config"


def survivor_metric_factory(dataset_length: int, total_models: int) -> od.Metric:
    """
    Returns a MAITE-compliant metric that computes fake metric results relevant for
    testing survivor.

    Fake metric results are divided between easy, hard and 'on-the-bubble' data:
        - easy data means a majority of models agree and have high confidence
        - hard data means a majority of models agree but have low confidence
        - on-the-bubble data means that there is not widespread agreement between the models

    Example - arrays of metric values for 3 models and a dataset of 6 images
               H     H      OTB  OTB     E      E
    model1: [0.001, 0.001, 0.33, 0.33, 0.999, 0.999]
    model2: [0.001, 0.001, 0.66, 0.66, 0.999, 0.999]
    model3: [0.001, 0.001, 1.0 , 1.0 , 0.999, 0.999]

    To produce the fake metric results, the metric assumes that it will be passed one image
    at a time, and that datasets will be iterated over before models. In the example above,
    this means that the metric will be first passed the six images from model 1 (one-by-one),
    then it will be passed the six images from model 2, and finally the six images from model 3.
    """

    class FakeSurvivorMetric:
        def __init__(self) -> None:
            # helper flags to prevent .compute, .reset or .update being
            # called multiple times by mistake.
            # update -> compute -> reset -> update -> ...
            self._can_compute = False
            self._needs_reset = False

            # counter for determining how many images have already been evaluated
            # see function docstring for further details
            self._counter = 0

        def compute(self) -> dict[str, Any]:
            if not self._can_compute:
                raise ValueError("FakeSurvivorMetric requires exactly one image. "
                "Please call .update before computing again.")

            self._can_compute = False
            self._needs_reset = True

            # image_idx and model_idx are used to keep track of the index
            # of the current image and the current model - see function docstring
            # for further details
            image_idx = self._counter % dataset_length
            model_idx = self._counter // dataset_length
            self._counter += 1

            # hard data - all models give the same score and its very low
            if image_idx < dataset_length/3:
                return {"fake_survivor_metric":torch.tensor([0.001])}

            # easy data - all models give the same score and its very high
            if image_idx >= 2*dataset_length/3:
                return {"fake_survivor_metric":torch.tensor([0.999])}

            # otb_data - there is no agreement between model scores.
            # the pattern adopted here is that each model is more confident
            # than the previous model by an increment of 1/total_models
            return {"fake_survivor_metric":torch.tensor([(model_idx + 1) / total_models])}

        def update(self, preds, targets) -> None:
            if self._can_compute:
                raise ValueError("FakeSurvivorMetric requires exactly one image. "
                "Please call .compute before updating again.")
            if self._needs_reset:
                raise ValueError("FakeSurvivorMetric requires exactly one image. "
                "Please call .reset before updating again.")

            self._can_compute = True

        def reset(self) -> None:
            if self._can_compute:
                raise ValueError("FakeSurvivorMetric requires exactly one image. "
                "Please call .compute before resetting again.")

            self._needs_reset = False

    return FakeSurvivorMetric()


@pytest.fixture(scope="session")
def survivor_test_stage_args(fake_od_dataset_default: FakeODDataset, fake_od_model_default: FakeODModel) -> dict[str, Any]:
    """
    Default arguments for RealLabelTestStage.
    
    The fake metric is the most important test object here. The test dataset and model
    are mostly ignored - the only pieces of information used by the fake metric are
    the number of images in the dataset and the number of models.
    """

    # choice of 6 arbitrary, but allows simple divide between easy, hard and otb images (2, 2 and 2)
    survivor_test_stage_dataset: od.Dataset = fake_od_dataset_default
    # at least 2 models are required for model disagreement to make sense
    survivor_test_stage_models: dict[str, od.Model] = {"model_1": fake_od_model_default, "model_2": fake_od_model_default}
    
    fake_survivor_metric = survivor_metric_factory(dataset_length=len(survivor_test_stage_dataset),
    total_models=len(survivor_test_stage_models))

    config = SurvivorConfig(
        unique_identifier_columns=["image_id"],
        metric_column="fake_survivor_metric",
        otb_threshold=0.9,
        difficulty_threshold=0.5,
        conversion_type=ScoreConversionType.ROUNDED,
        conversion_args={"decimals_to_round": 2},
    )

    dict_config = {
        "unique_identifier_columns": ["image_id"],
        "metric_column": "fake_survivor_metric",
        "otb_threshold": 0.9,
        "difficulty_threshold": 0.5,
        "conversion_type": ScoreConversionType.ROUNDED,
        "conversion_args": {"decimals_to_round": 2},
    }

    return {
        _SURVIVOR_CONFIG: config,
        "dataset": survivor_test_stage_dataset,
        "metric": fake_survivor_metric,
        "models": survivor_test_stage_models,
        _DICT_CONFIG: dict_config,
    }


@pytest.fixture(name="test_stage")
def create_test_stage(
    survivor_test_stage_args: dict, request: pytest.FixtureRequest
) -> SurvivorTestStage:
    """Create a SurvivorTestStage object and load in all required args.

    Can load in both the `dict_config` and `config` configurations in `survivor_test_stage_args` depending on the
    string input to `request.param` (set through indirect parametrization of `test_stage`).
    """
    # Ensure cache doesn't exist
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(CACHE_DIR)

    # Create and configure SurvivorTestStage
    test_stage = SurvivorTestStage(
        config=survivor_test_stage_args[getattr(request, "param", _SURVIVOR_CONFIG)]
    )
    test_stage.load_models(models=survivor_test_stage_args["models"])
    test_stage.load_dataset(dataset=survivor_test_stage_args["dataset"], dataset_id="test-dataset")
    test_stage.load_metric(
        metric=survivor_test_stage_args["metric"], metric_id=survivor_test_stage_args["config"].metric_column
    )

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
    """Test RealLabelTestStage generates a cache object that can be read correctly."""
    # Arrange
    expected_cache_location = Path(test_stage.cache_base_path) / test_stage.cache_id
    expected_results_df_path = expected_cache_location / Path("survivor_standard_results.csv")
    expected_results_png_path = expected_cache_location / Path("survivor_result_visualization.png")
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
    actual_returned_results_df = test_stage.outputs[
        0
    ]  # .withColumn("timestamp", sf.col("timestamp").cast("timestamp"))
    assert_spark_dataframes_equal(actual_returned_results_df, actual_cached_results_df.toPandas())

    # Compare the read-from-cache image against the actual image returned from `run()`
    assert expected_results_png_path.exists()
    compare_images(str(test_stage.outputs[1]), str(actual_cached_image), 0.001)
    # Further compare the image against what we expect the image to look like from a snapshot
    compare_images(str(test_stage.outputs[1]), str(EXPECTED_SURVIVOR_IMAGE), 0.001)

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
    expected_cache_id = "survivor_od_cache_1fc5132b703f7ae1d6000e8c70eeccdff731f6f07bc133366640b6f0c7db170d"

    # Act
    actual_cache_id = test_stage.cache_id

    # Assert
    assert actual_cache_id == expected_cache_id


def test_survivor_collect_report_consumables(
    test_stage: SurvivorTestStage,
    artifact_dir,
) -> None:
    """Test collect_report_consumables."""
    # Arrange
    expected_deck = "object_detection_survivor"
    expected_layout_name = "TwoImageTextNoHeader"
    expected_content_left = Text(
        content="**Types of Data**\n"
        "• Easy: Models score the same and perform well.\n"
        "• Hard: Models score the same and perform poorly.\n"
        "• On the Bubble: Models score differently.\n\n"
        "• Ideally, a dataset would be primarily On the Bubble, so all data is helping distinguish between model "
        "performance.\n\n"
        "• This dataset had 33.3% Easy, 33.3% Hard, and "
        "33.3% On the Bubble data.",
        fontsize=22,
    )
    expected_content_right = f"{test_stage.cache_base_path}/{test_stage.cache_id}/survivor_result_visualization.png"
    expected_title = "Survivor Dataset Breakdown"

    # Run test stage once to ensure cache is present
    test_stage.run(use_cache=True)

    # Run again to use cache
    test_stage.run(use_cache=True)

    # Act
    slide_content = test_stage.collect_report_consumables()
    output_consumables = slide_content[0]

    # Assert
    assert output_consumables["deck"] == expected_deck
    assert output_consumables["layout_name"] == expected_layout_name
    assert output_consumables["layout_arguments"]["title"] == expected_title
    assert output_consumables["layout_arguments"]["content_left"].content == expected_content_left.content
    assert output_consumables["layout_arguments"]["content_right"].as_posix() == expected_content_right

    filename = create_deck(slide_content, artifact_dir, 'survivor')
    assert filename.exists()

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
    expected_output = {"Low_Val_Data": 1.0 - (1 / 3)}

    test_stage.run()

    # Act
    actual_output = test_stage.collect_metrics()

    # Assert
    assert actual_output == expected_output


def test_survivor_test_stage_collect_metrics_error(test_stage: SurvivorTestStage) -> None:
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
        test_stage_1.run(use_cache=False)

    with pytest.raises(RIValidationError, match=r"'dataset' not set! Please use `load_dataset\(\)` function"):
        test_stage_2.run(use_cache=False)

    with pytest.raises(RIValidationError, match=r"'metric' not set! Please use `load_metric\(\)` function"):
        test_stage_3.run(use_cache=False)


def test_missing_cache_image_error(tmp_path: Path) -> None:
    """Test error is raised when there is data in the cache but no image."""
    # Arrange
    test_df = pd.DataFrame({"image_id": []})
    test_df.to_csv(tmp_path / "survivor_standard_results.csv")
    cache = SurvivorCache()

    # Act and Assert
    with pytest.warns(
        UserWarning, match=f"Survivor cache path {tmp_path} doesn't contain a cached result visualization!"
    ):
        cache.read_cache(str(tmp_path))


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
