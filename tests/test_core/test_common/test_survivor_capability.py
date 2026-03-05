from copy import deepcopy

import pytest

pytest.importorskip("survivor")

import torch  # noqa: E402

from checkmaite.core._common.survivor_capability import SurvivorBase, SurvivorConfig  # noqa: E402


def survivor_metric_factory(dataset_length, total_models):
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
        metadata = {"id": "fake-id"}

        def __init__(self):
            # helper flags to prevent .compute, .reset or .update being
            # called multiple times by mistake.
            # update -> compute -> reset -> update -> ...
            self._can_compute = False
            self._needs_reset = False

            # counter for determining how many images have already been evaluated
            # see function docstring for further details
            self._counter = 0

        def compute(self):
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

        def update(self, preds, targets, metadata):
            if self._can_compute:
                raise ValueError(
                    "FakeSurvivorMetric requires exactly one image. Please call .compute before updating again."
                )
            if self._needs_reset:
                raise ValueError(
                    "FakeSurvivorMetric requires exactly one image. Please call .reset before updating again."
                )

            self._can_compute = True

        def reset(self):
            if self._can_compute:
                raise ValueError(
                    "FakeSurvivorMetric requires exactly one image. Please call .compute before resetting again."
                )

            self._needs_reset = False

    return FakeSurvivorMetric()


@pytest.fixture
def survivor_od_capability_args(fake_od_dataset_default, fake_od_model_default):
    """
    Default arguments for RealLabelTestStage.

    The fake metric is the most important test object here. The test dataset and model
    are mostly ignored - the only pieces of information used by the fake metric are
    the number of images in the dataset and the number of models.
    """

    # choice of 6 arbitrary, but allows simple divide between easy, hard and otb images (2, 2 and 2)
    survivor_test_stage_dataset = fake_od_dataset_default
    # at least 2 models are required for model disagreement to make sense
    survivor_test_stage_models = {
        "model_1": fake_od_model_default,
        "model_2": fake_od_model_default,
    }

    fake_survivor_metric = survivor_metric_factory(
        dataset_length=len(survivor_test_stage_dataset), total_models=len(survivor_test_stage_models)
    )

    return {
        "dataset": survivor_test_stage_dataset,
        "metric": fake_survivor_metric,
        "models": survivor_test_stage_models,
    }


@pytest.fixture
def survivor_ic_capability_args(fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default):
    fake_model = fake_ic_model_default
    model_dict = {
        "fake_model": fake_model,
    }
    detection_dataset = fake_ic_dataset_default
    map_metric = fake_ic_metric_default

    return {
        "dataset": detection_dataset,
        "metric": map_metric,
        "models": model_dict,
    }


@pytest.fixture
def capability_args(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "capability_args",
    [
        "survivor_od_capability_args",
        "survivor_ic_capability_args",
    ],
    indirect=True,
)
@pytest.mark.unsupported
def test_run_and_collect(capability_args):
    capability = SurvivorBase()

    # CRUCIAL: we need to have unique model ids, which with the current design means unique model objects
    models = [deepcopy(m) for m in capability_args["models"].values()]
    for idx, m in enumerate(models):
        m.metadata = {"id": f"model_{idx+1}"}
    datasets = [capability_args["dataset"]]
    for d in datasets:
        d.metadata = {"id": "test-dataset"}
    metrics = [capability_args["metric"]]
    for m in metrics:
        m.metadata = {"id": "fake_survivor_metric"}

    if hasattr(metrics[0], "calculated_metrics"):
        config = SurvivorConfig(metric_column=metrics[0].calculated_metrics.keys().__iter__().__next__())
    else:
        # od test fixtures do not have calculated_metrics property in current design ...
        config = SurvivorConfig(metric_column="fake_survivor_metric")

    output = capability.run(models=models, datasets=datasets, metrics=metrics, config=config, use_cache=False)

    assert output.model_dump()  # smoke test

    assert output.collect_report_consumables(threshold=0.5)  # smoke test

    assert output.collect_md_report(threshold=0.5)  # smoke test
