"""Survivor Object Detection Test Stage Implementation"""

from jatic_ri._common.test_stages.impls.survivor_test_stage import SurvivorTestStageBase


class SurvivorTestStage(SurvivorTestStageBase):
    """Survivor Test Stage Object.

    Survivor uses an ensemble of models and metrics based on model inference results, to provide insight into
    how difficult a set of image may be for models. Generally speaking, "Easy" images are those that most models
    perform well on, "Hard" images are those that most models perform poorly on, and "On the Bubble" images are those
    that models have a wide variety of performance ranges on.

    For more info, see our docs! https://jatic.pages.jatic.net/morse/survivor/

    This test stage also uses MAITE-wrapped models, datasets, and metrics, and MAITE itself, to produce the model
    metric results needed if they are not present in the cache before running Survivor itself.

    Attributes
    ----------
    outputs
        A tuple-like object of Survivor results with the layout:
        [0]: The SurvivorResults.raw_output_df dataframe.
        [1]: Histogram of the number of images per Survivor category: Easy, Hard, and On the Bubble.
    metric
        The MAITE-wrapped metric object that should be fed the model inference results
        for metric calculation.
    dataset
        The MAITE-wrapped dataset object on which the models should run inference and
        produce results.
    models
        The dictionary of model names to their MAITE-wrapped model objects
        whose inference should be used when running Survivor.
    """

    _deck: str = "object_detection_survivor"
    _task: str = "od"
