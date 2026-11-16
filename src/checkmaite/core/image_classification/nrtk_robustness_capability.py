import maite.protocols.image_classification as ic
from nrtk.interop import MAITEImageClassificationAugmentation

from checkmaite.core._common.nrtk_robustness_capability import (
    NrtkRobustnessBase,
    NrtkRobustnessConfig,
    NrtkRobustnessOutputs,
)

__all__ = ["NrtkRobustness", "NrtkRobustnessConfig", "NrtkRobustnessOutputs"]


class NrtkRobustness(NrtkRobustnessBase[ic.Dataset, ic.Model, ic.Metric]):
    """
    Apply realistic image perturbations and measure how a model's performance changes.

    Evaluates the model on the unperturbed dataset (the baseline), then at each level of an NRTK perturbation
    sweep, and reports each level's score next to the baseline.
    """

    task = "image_classification"
    _augmentation_type = MAITEImageClassificationAugmentation
