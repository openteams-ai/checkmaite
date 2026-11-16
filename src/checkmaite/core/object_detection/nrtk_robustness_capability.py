import maite.protocols.object_detection as od
from nrtk.interop import MAITEObjectDetectionAugmentation

from checkmaite.core._common.nrtk_robustness_capability import (
    NrtkRobustnessBase,
    NrtkRobustnessConfig,
    NrtkRobustnessOutputs,
)

__all__ = ["NrtkRobustness", "NrtkRobustnessConfig", "NrtkRobustnessOutputs"]


class NrtkRobustness(NrtkRobustnessBase[od.Dataset, od.Model, od.Metric]):
    """
    Apply realistic image perturbations and measure how a model's performance changes.

    Evaluates the model on the unperturbed dataset (the baseline), then at each level of an NRTK perturbation
    sweep, and reports each level's score next to the baseline.
    """

    task = "object_detection"
    _augmentation_type = MAITEObjectDetectionAugmentation
