"""Test RealLabelTestStage."""

import dataclasses
from typing import Any

import pytest

# Skip all tests in this file if reallabel isn't available
pytest.importorskip("reallabel")

# Module-level imports after importorskip to prevent collection errors
import maite.protocols.object_detection as od  # noqa: E402
import pandas as pd  # noqa: E402
from gradient.templates_and_layouts.create_deck import create_deck  # noqa: E402
from reallabel import ColumnNameConfig, RealLabelResults  # noqa: E402

from jatic_ri._common.test_stages.interfaces.test_stage import RIValidationError  # noqa: E402
from jatic_ri.object_detection.test_stages import RealLabelConfig, RealLabelOutputs, RealLabelTestStage  # noqa: E402
from tests.fake_od_classes import FakeODDataset, FakeODModel  # noqa: E402

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
def create_test_stage(reallabel_test_stage_args: dict, request: pytest.FixtureRequest) -> RealLabelTestStage:
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

    return test_stage


@pytest.mark.parametrize(
    "test_stage",
    [_REALLABEL_CONFIG, _DICT_CONFIG],
    ids=["Using RealLabelConfig", "Using dict config"],
    indirect=True,
)
def test_reallabel_test_stage_run_caches(mocker, test_stage: RealLabelTestStage, tmp_cache_path) -> None:
    """Test RealLabelTestStage generates a cache object correctly."""
    run = test_stage.run(use_stage_cache=True)

    mocker.patch.object(test_stage, "_run", side_effect=AssertionError("_run() called while cache hit was expected"))
    cached_run = test_stage.run(use_stage_cache=True)
    assert cached_run is not run

    outputs = run.outputs
    cached_outputs = cached_run.outputs

    def optional_assert_frame_equal(actual, expected):
        if isinstance(actual, pd.DataFrame) and isinstance(expected, pd.DataFrame):
            return pd.testing.assert_frame_equal(actual, expected)

        return actual == expected

    optional_assert_frame_equal(outputs.results, cached_outputs.results)
    optional_assert_frame_equal(outputs.classification_disagreements_df, cached_outputs.classification_disagreements_df)
    optional_assert_frame_equal(outputs.verbose_df, cached_outputs.verbose_df)
    optional_assert_frame_equal(outputs.sequence_priority_score_df, cached_outputs.sequence_priority_score_df)
    optional_assert_frame_equal(
        outputs.sequence_priority_score_balanced_df, cached_outputs.sequence_priority_score_balanced_df
    )
    optional_assert_frame_equal(outputs.wanrs_df, cached_outputs.wanrs_df)
    optional_assert_frame_equal(outputs.aggregated_confidence_df, cached_outputs.aggregated_confidence_df)
    assert outputs.example_image.id == cached_outputs.example_image.id


def test_reallabel_test_stage_collect_report_consumables(
    test_stage: RealLabelTestStage,
    artifact_dir,
) -> None:
    """Test collect_report_consumables with cached data enabled."""
    # Arrange
    expected_deck = "object_detection_reallabel"
    expected_layout_name = "TwoItem"
    expected_title = "RealLabel Label Breakdown"

    # Run test stage once to ensure cache is present
    test_stage.run(use_stage_cache=True)

    # Run again to use cache
    test_stage.run(use_stage_cache=True)

    # Act
    slides = test_stage.collect_report_consumables()
    output_consumables = slides[0]
    combined_lc_text = "".join(
        [subtext.content for subtext in output_consumables["layout_arguments"]["left_item"].content]
    )

    # Assert
    assert output_consumables["deck"] == expected_deck
    assert output_consumables["layout_name"] == expected_layout_name
    assert output_consumables["layout_arguments"]["title"] == expected_title
    assert all(
        expected in combined_lc_text for expected in ["True Positive: 1", "False Positive: 2", "False Negative: 1"]
    )
    assert output_consumables["layout_arguments"]["right_item"].is_file()

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


def test_for_reallabel_output_changes():
    """When updating RealLabel, make sure we catch changes to the results dataclass and determine whether to expose them to the user."""
    available = {f.name for f in dataclasses.fields(RealLabelResults) if not f.name.startswith("_")}
    exposed = set(RealLabelOutputs.model_fields.keys())
    assert exposed.issuperset(available)
