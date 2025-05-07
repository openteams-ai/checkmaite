"""Test Object Detection Survivor test stage."""

from typing import Any

import pandas as pd
import pytest
import torch
from gradient import Text
from gradient.templates_and_layouts.create_deck import create_deck
from maite.protocols import object_detection as od
from survivor.config import SurvivorConfig
from survivor.enums import ScoreConversionType

from jatic_ri._common.test_stages.interfaces.test_stage import RIValidationError
from jatic_ri.object_detection.test_stages.impls.survivor_test_stage import SurvivorTestStage
from tests.fake_od_classes import FakeODDataset, FakeODModel

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
                raise ValueError(
                    "FakeSurvivorMetric requires exactly one image. " "Please call .update before computing again."
                )

            self._can_compute = False
            self._needs_reset = True

            # image_idx and model_idx are used to keep track of the index
            # of the current image and the current model - see function docstring
            # for further details
            image_idx = self._counter % dataset_length
            model_idx = self._counter // dataset_length
            self._counter += 1

            # hard data - all models give the same score and its very low
            if image_idx < dataset_length / 3:
                return {"fake_survivor_metric": torch.tensor([0.001])}

            # easy data - all models give the same score and its very high
            if image_idx >= 2 * dataset_length / 3:
                return {"fake_survivor_metric": torch.tensor([0.999])}

            # otb_data - there is no agreement between model scores.
            # the pattern adopted here is that each model is more confident
            # than the previous model by an increment of 1/total_models
            return {"fake_survivor_metric": torch.tensor([(model_idx + 1) / total_models])}

        def update(self, preds, targets) -> None:
            if self._can_compute:
                raise ValueError(
                    "FakeSurvivorMetric requires exactly one image. " "Please call .compute before updating again."
                )
            if self._needs_reset:
                raise ValueError(
                    "FakeSurvivorMetric requires exactly one image. " "Please call .reset before updating again."
                )

            self._can_compute = True

        def reset(self) -> None:
            if self._can_compute:
                raise ValueError(
                    "FakeSurvivorMetric requires exactly one image. " "Please call .compute before resetting again."
                )

            self._needs_reset = False

    return FakeSurvivorMetric()


@pytest.fixture(scope="session")
def survivor_test_stage_args(
    fake_od_dataset_default: FakeODDataset, fake_od_model_default: FakeODModel
) -> dict[str, Any]:
    """
    Default arguments for RealLabelTestStage.

    The fake metric is the most important test object here. The test dataset and model
    are mostly ignored - the only pieces of information used by the fake metric are
    the number of images in the dataset and the number of models.
    """

    # choice of 6 arbitrary, but allows simple divide between easy, hard and otb images (2, 2 and 2)
    survivor_test_stage_dataset: od.Dataset = fake_od_dataset_default
    # at least 2 models are required for model disagreement to make sense
    survivor_test_stage_models: dict[str, od.Model] = {
        "model_1": fake_od_model_default,
        "model_2": fake_od_model_default,
    }

    fake_survivor_metric = survivor_metric_factory(
        dataset_length=len(survivor_test_stage_dataset), total_models=len(survivor_test_stage_models)
    )

    config = SurvivorConfig(
        metric_column="fake_survivor_metric",
        otb_threshold=0.9,
        easy_hard_threshold=0.5,
        conversion_type=ScoreConversionType.ROUNDED,
        conversion_args={"decimals_to_round": 2},
    )

    dict_config = {
        "metric_column": "fake_survivor_metric",
        "otb_threshold": 0.9,
        "easy_hard_threshold": 0.5,
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
    survivor_test_stage_args: dict, request: pytest.FixtureRequest, default_eval_tool_no_cache
) -> SurvivorTestStage:
    """Create a SurvivorTestStage object and load in all required args.

    Can load in both the `dict_config` and `config` configurations in `survivor_test_stage_args` depending on the
    string input to `request.param` (set through indirect parametrization of `test_stage`).
    """
    # Create and configure SurvivorTestStage
    test_stage = SurvivorTestStage(config=survivor_test_stage_args[getattr(request, "param", _SURVIVOR_CONFIG)])
    test_stage.load_models(models=survivor_test_stage_args["models"])
    test_stage.load_dataset(dataset=survivor_test_stage_args["dataset"], dataset_id="test-dataset")
    test_stage.load_metric(
        metric=survivor_test_stage_args["metric"], metric_id=survivor_test_stage_args["config"].metric_column
    )
    test_stage.load_eval_tool(default_eval_tool_no_cache)

    return test_stage


@pytest.mark.parametrize(
    "test_stage",
    [_SURVIVOR_CONFIG, _DICT_CONFIG],
    ids=["Using SurvivorConfig", "Using dict config"],
    indirect=True,
)
def test_survivor_test_stage_run_caches(mocker, test_stage: SurvivorTestStage, tmp_cache_path) -> None:
    """Test RealLabelTestStage generates a cache object that can be read correctly."""
    run = test_stage.run(use_stage_cache=True)

    mocker.patch.object(test_stage, "_run", side_effect=AssertionError("_run() called while cache hit was expected"))
    cached_run = test_stage.run(use_stage_cache=True)
    assert cached_run is not run

    df, _ = run.outputs
    cached_df, _ = cached_run.outputs

    pd.testing.assert_frame_equal(cached_df, df)


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
    expected_title = "Survivor Dataset Breakdown"

    # Run test stage once to ensure cache is present
    test_stage.run(use_stage_cache=True)

    # Run again to use cache
    test_stage.run(use_stage_cache=True)

    # Act
    slide_content = test_stage.collect_report_consumables()
    output_consumables = slide_content[0]

    # Assert
    assert output_consumables["deck"] == expected_deck
    assert output_consumables["layout_name"] == expected_layout_name
    assert output_consumables["layout_arguments"]["title"] == expected_title
    assert output_consumables["layout_arguments"]["content_left"].content == expected_content_left.content
    assert output_consumables["layout_arguments"]["content_right"].is_file()

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
