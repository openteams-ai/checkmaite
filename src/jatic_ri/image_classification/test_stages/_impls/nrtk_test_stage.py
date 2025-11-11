"""NRTKTestStage implementation"""

import maite.protocols.image_classification as ic

from jatic_ri._common.test_stages.impls.nrtk_test_stage import NRTKTestStageBase


class NRTKTestStage(NRTKTestStageBase[ic.Dataset, ic.Model, ic.Metric]):
    """
    NRTK test stage that applies realistic image perturbations, evaluates a configured metric, and reports
    performance deltas.

    Iterates over NRTK perturbations that mimic real-world conditions, applies them to the dataset, runs the metric
    on each perturbed variant, and generates a report summarizing changes in model performance.
    """

    _task: str = "image_classification"
