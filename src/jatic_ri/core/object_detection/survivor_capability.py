import maite.protocols.object_detection as od

from jatic_ri.core._common.survivor_capability import SurvivorBase


class Survivor(SurvivorBase[od.Dataset, od.Model, od.Metric]):
    """Survivor capability.

    Survivor uses an ensemble of models and metrics based on model inference results, to provide insight into
    how difficult a set of image may be for models. Generally speaking, "Easy" images are those that most models
    perform well on, "Hard" images are those that most models perform poorly on, and "On the Bubble" images are those
    that models have a wide variety of performance ranges on.

    For more info, see our docs! https://jatic.pages.jatic.net/morse/survivor/

    This capability also uses MAITE-wrapped models, datasets, and metrics, and MAITE itself, to produce the model
    metric results needed if they are not present in the cache before running Survivor itself.
    """
