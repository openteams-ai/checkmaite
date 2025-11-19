"""Test Object Detection Survivor test stage."""

from copy import deepcopy
from typing import Any

import pytest

# Skip all tests in this file if survivor isn't available
pytest.importorskip("survivor")

# Module-level imports after importorskip to prevent collection errors
import pandas as pd  # noqa: E402
import PIL  # noqa: E402
import torch  # noqa: E402
from gradient import SubText, Text  # noqa: E402
from gradient.templates_and_layouts.create_deck import create_deck  # noqa: E402
from maite.protocols import MetricMetadata  # noqa: E402
from maite.protocols import object_detection as od  # noqa: E402
from survivor.enums import ScoreConversionType  # noqa: E402

from jatic_ri._common.test_stages.impls.survivor_test_stage import SurvivorOutputs  # noqa: E402
from jatic_ri.object_detection.test_stages import SurvivorConfig, SurvivorTestStage  # noqa: E402
from tests.fake_od_classes import FakeODDataset, FakeODModel  # noqa: E402


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
        metadata = MetricMetadata(id="fake-id")

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
                    "FakeSurvivorMetric requires exactly one image. Please call .update before computing again."
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
                    "FakeSurvivorMetric requires exactly one image. Please call .compute before updating again."
                )
            if self._needs_reset:
                raise ValueError(
                    "FakeSurvivorMetric requires exactly one image. Please call .reset before updating again."
                )

            self._can_compute = True

        def reset(self) -> None:
            if self._can_compute:
                raise ValueError(
                    "FakeSurvivorMetric requires exactly one image. Please call .compute before resetting again."
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

    return {
        "config": config,
        "dataset": survivor_test_stage_dataset,
        "metric": fake_survivor_metric,
        "models": survivor_test_stage_models,
    }


def test_survivor_test_stage_run_caches(mocker, survivor_test_stage_args: dict) -> None:
    """Test RealLabelTestStage generates a cache object that can be read correctly."""

    test_stage = SurvivorTestStage()
    config = survivor_test_stage_args["config"]
    # CRUCIAL: we need to have unique model ids, which with the current design means unique model objects
    models = [deepcopy(m) for m in survivor_test_stage_args["models"].values()]
    for idx, m in enumerate(models):
        m.metadata = {"id": f"model_{idx+1}"}

    datasets = [survivor_test_stage_args["dataset"]]
    for d in datasets:
        d.metadata = {"id": "test-dataset"}
    metrics = [survivor_test_stage_args["metric"]]
    for m in metrics:
        m.metadata = {"id": "fake_survivor_metric"}

    run = test_stage.run(config=config, models=models, datasets=datasets, metrics=metrics, use_stage_cache=True)
    mocker.patch.object(test_stage, "_run", side_effect=AssertionError("_run() called while cache hit was expected"))
    cached_run = test_stage.run(config=config, models=models, datasets=datasets, metrics=metrics, use_stage_cache=True)
    assert cached_run is not run

    def assert_item_equal(obj1: Any, obj2: Any, field_name: str) -> None:
        """Compare two objects and assert they are equal."""
        if isinstance(obj1, list):
            for item1, item2 in zip(obj1, obj2, strict=True):
                assert_item_equal(item1, item2, field_name)
        elif isinstance(obj1, pd.DataFrame):
            pd.testing.assert_frame_equal(obj1, obj2)
        elif isinstance(obj1, PIL.Image.Image):
            # images are a visualization of the data, so comparing dataframe is sufficient
            return
        else:
            assert obj1 == obj2, f'Field "{field_name}" does not match between cached and run outputs'

    for field in SurvivorOutputs.model_fields:
        cached_attr = getattr(cached_run.outputs, field)
        run_attr = getattr(run.outputs, field)
        assert_item_equal(run_attr, cached_attr, field)


def test_survivor_collect_report_consumables(
    survivor_test_stage_args: dict,
    artifact_dir,
) -> None:
    """Test collect_report_consumables."""

    test_stage = SurvivorTestStage()

    # Arrange
    expected_deck = "jatic_ri.object_detection.test_stages._impls.survivor_test_stage.SurvivorTestStage"
    expected_layout_name = "TwoItem"
    expected_content_left = Text(
        content=[
            SubText("Types of Data\n", bold=True),
            "• Easy: Models score the same and perform well.\n"
            "• Hard: Models score the same and perform poorly.\n"
            "• On the Bubble: Models score differently.\n\n"
            "• Ideally, a dataset would be primarily On the Bubble, so all data is helping distinguish between model "
            "performance.\n\n"
            "• This dataset had 33.3% Easy, 33.3% Hard, and "
            "33.3% On the Bubble data.",
        ],
        fontsize=22,
    )
    expected_title = "Survivor Dataset Breakdown"

    config = survivor_test_stage_args["config"]
    # CRUCIAL: we need to have unique model ids, which with the current design means unique model objects
    models = [deepcopy(m) for m in survivor_test_stage_args["models"].values()]
    for idx, m in enumerate(models):
        m.metadata = {"id": f"model_{idx+1}"}
    datasets = [survivor_test_stage_args["dataset"]]
    for d in datasets:
        d.metadata = {"id": "test-dataset"}
    metrics = [survivor_test_stage_args["metric"]]
    for m in metrics:
        m.metadata = {"id": "fake_survivor_metric"}

    # Run test stage once to ensure cache is present
    run = test_stage.run(config=config, models=models, datasets=datasets, metrics=metrics, use_stage_cache=True)

    # Run again to use cache
    _ = test_stage.run(config=config, models=models, datasets=datasets, metrics=metrics, use_stage_cache=True)
    # Act
    slide_content = run.collect_report_consumables(threshold=0.5)
    output_consumables = slide_content[0]

    # Assert
    assert output_consumables["deck"] == expected_deck
    assert output_consumables["layout_name"] == expected_layout_name
    assert output_consumables["layout_arguments"]["title"] == expected_title
    assert output_consumables["layout_arguments"]["left_item"].content == expected_content_left.content
    assert output_consumables["layout_arguments"]["right_item"].is_file()

    filename = create_deck(slide_content, artifact_dir, "survivor")
    assert filename.exists()
