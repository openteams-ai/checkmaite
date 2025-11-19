"""Dataset Shift Object Detection Test Stage Implementation"""

import maite.protocols.object_detection as od

from jatic_ri._common.test_stages.impls.dataeval_shift_test_stage import DatasetShiftTestStageBase


class DatasetShiftTestStage(DatasetShiftTestStageBase[od.Dataset, od.Model, od.Metric]):
    """Detects dataset shift between two datasets using various methods"""
