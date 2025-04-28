"""Test RealLabelTestStage."""

import json
import os
from pathlib import Path
from typing import Any

import maite.protocols.object_detection as od
import pandas as pd
import pytest
from gradient.templates_and_layouts.create_deck import create_deck
from reallabel import ColumnNameConfig

from jatic_ri._common.test_stages.interfaces.test_stage import RIValidationError
from jatic_ri.object_detection.test_stages.impls.reallabel_test_stage import (
    _REALLABEL_CACHE_CONFIGURATION_PATH,
    _REALLABEL_CACHE_IMAGE_PATH,
    _REALLABEL_CACHE_JSON_PATH,
    RealLabelCache,
    RealLabelConfig,
    RealLabelTestStage,
    RealLabelTestStageResults,
)
from tests.fake_od_classes import FakeODDataset, FakeODModel

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
def reallabel_test_stage_args(
    fake_od_dataset_reallabel_only: FakeODDataset, fake_od_model_default: FakeODModel
) -> dict[str, Any]:
    """Default arguments for RealLabelTestStage."""

    # a single image with groundtruth along with two models with identical predictions
    reallabel_test_stage_dataset: od.Dataset = fake_od_dataset_reallabel_only
    reallabel_test_stage_models: dict[str, od.Model] = {
        "model_1": fake_od_model_default,
        "model_2": fake_od_model_default,
    }

    config = RealLabelConfig(
        deduplication_algorithm="wbf",
        column_names=ColumnNameConfig(
            unique_identifier_columns=["id"],
            calibrated_confidence_column="score",
        ),
        run_confidence_calibration=False,
        keep_likely_corrects=True,
        deduplication_iou_threshold=0.5,
        minimum_confidence_threshold=0.1,
        threshold_max_aggregated_confidence_fp=0.01,
    )

    dict_config = {
        "deduplication_algorithm": "wbf",
        "column_names": ColumnNameConfig(
            unique_identifier_columns=["id"],
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
def create_test_stage(
    default_eval_tool_no_cache, reallabel_test_stage_args: dict, request: pytest.FixtureRequest
) -> RealLabelTestStage:
    """Create a RealLabelTestStage object and load in all required args.

    Can load in both the `dict_config` and `config` configurations in `reallabel_test_stage_args` depending on the
    string input to `request.param` (set through indirect parametrization of `test_stage`).
    """

    # Create and yield test_stage
    test_stage = RealLabelTestStage(config=reallabel_test_stage_args[getattr(request, "param", _REALLABEL_CONFIG)])
    test_stage.load_models(models=reallabel_test_stage_args["models"])
    test_stage.load_dataset(
        dataset=reallabel_test_stage_args["dataset"],
        dataset_id="test-dataset",
    )
    test_stage.load_eval_tool(default_eval_tool_no_cache)

    return test_stage


@pytest.mark.parametrize(
    "test_stage",
    [_REALLABEL_CONFIG, _DICT_CONFIG],
    ids=["Using RealLabelConfig", "Using dict config"],
    indirect=True,
)
def test_reallabel_test_stage_run_caches(test_stage: RealLabelTestStage, tmp_cache_path) -> None:
    """Test RealLabelTestStage generates a cache object correctly."""
    # Arrange
    expected_cache_location = tmp_cache_path / test_stage.cache_id

    expected_json_results_path = expected_cache_location / _REALLABEL_CACHE_JSON_PATH
    expected_results_config_path = expected_cache_location / _REALLABEL_CACHE_CONFIGURATION_PATH

    reallabel_cache = RealLabelCache()

    # Act
    test_stage.run(use_stage_cache=True)
    actual_returned_results = test_stage.outputs

    actual_cached_results = reallabel_cache.read_cache(cache_path=str(expected_cache_location))

    for field in RealLabelTestStageResults.model_fields:
        if isinstance(getattr(actual_returned_results, field), pd.DataFrame):
            returned_df = getattr(actual_returned_results, field)
            cached_df = getattr(actual_cached_results, field)
            # Pre-existing type conversion issues, so we need to cast the dataframes to the same types for the test
            assert cached_df.equals(returned_df.astype(cached_df.dtypes.to_dict()))
        elif isinstance(getattr(actual_returned_results, field), Path):
            # load the files and compare contents
            returned_path = getattr(actual_returned_results, field)
            cached_path = getattr(actual_cached_results, field)
            assert returned_path.exists()
            assert cached_path.exists()
            # Check that the file sizes are the same
            assert returned_path.stat().st_size == cached_path.stat().st_size
            with returned_path.open("rb") as returned_file, cached_path.open("rb") as cached_file:
                assert returned_file.read() == cached_file.read()
        else:
            assert getattr(actual_returned_results, field) == getattr(actual_cached_results, field)

    assert expected_json_results_path.exists()

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
    expected_cache_id = "reallabel_od_cache_b22bc35e63da8a8f663bc27440c18dc9ec6df61c2064ccd8f1de6e8d9eb4ab16"

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
    expected_content_right = str(artifact_dir / test_stage.cache_path / _REALLABEL_CACHE_IMAGE_PATH)
    expected_title = "RealLabel Label Breakdown"

    # Run test stage once to ensure cache is present
    test_stage.run(use_stage_cache=True)

    # Run again to use cache
    test_stage.run(use_stage_cache=True)

    # Act
    slides = test_stage.collect_report_consumables()
    output_consumables = slides[0]
    combined_lc_text = "".join(
        [subtext.content for subtext in output_consumables["layout_arguments"]["content_left"].content]
    )

    # Assert
    assert output_consumables["deck"] == expected_deck
    assert output_consumables["layout_name"] == expected_layout_name
    assert output_consumables["layout_arguments"]["title"] == expected_title
    assert all(
        expected in combined_lc_text for expected in ["True Positive: 1", "False Positive: 2", "False Negative: 1"]
    )
    assert output_consumables["layout_arguments"]["content_right"].as_posix() == expected_content_right

    filename = create_deck(slides, artifact_dir, "reallabel")
    assert filename.exists()


def test_reallabel_test_stage_collect_report_consumables_error(
    test_stage: RealLabelTestStage,
) -> None:
    """Test collect_report_consumables error when run not called."""
    # No Arrange

    # Act and Assert
    with pytest.raises(RuntimeError, match="TestStage must be run before accessing outputs"):
        test_stage.collect_report_consumables()


def test_reallabel_test_stage_run_errors(reallabel_test_stage_args: dict):
    """Test run() errors."""
    # Arrange
    test_stage_1 = RealLabelTestStage(reallabel_test_stage_args["config"])
    test_stage_2 = RealLabelTestStage(reallabel_test_stage_args["config"])
    test_stage_2.load_models(reallabel_test_stage_args["models"])

    # Act and Assert
    with pytest.raises(RIValidationError, match=r"'models' not set! Please use `load_models\(\)` function"):
        test_stage_1.run(use_stage_cache=False)

    with pytest.raises(RIValidationError, match=r"'dataset' not set! Please use `load_dataset\(\)` function"):
        test_stage_2.run(use_stage_cache=False)


def test_cache_miss_dir_resets(test_stage: RealLabelTestStage, tmp_cache_path) -> None:
    """Test cache miss dir is deleted and resets if it already exists."""
    # Arrange
    output = tmp_cache_path / "reallabel_cache_miss_outputs"
    output.mkdir()
    file = output / "test_file.txt"
    file.touch()

    # Act
    test_stage._run()

    # Assert
    assert output.exists()
    assert not file.exists()
