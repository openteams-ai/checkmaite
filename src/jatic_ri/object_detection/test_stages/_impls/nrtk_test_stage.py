"""NRTKTestStage implementation"""

import maite.protocols.object_detection as od

from jatic_ri._common.test_stages.impls.nrtk_test_stage import NRTKTestStageBase


class NRTKTestStage(NRTKTestStageBase[od.Dataset, od.Model, od.Metric]):
    """
    NRTK test stage that applies realistic image perturbations, evaluates a configured metric, and reports
    performance deltas.

    Iterates over NRTK perturbations that mimic real-world conditions, applies them to the dataset, runs the metric
    on each perturbed variant, and generates a report summarizing changes in model performance.
    """

    _deck: str = "object_detection_dataset_evaluation"
    _task: str = "object_detection"
