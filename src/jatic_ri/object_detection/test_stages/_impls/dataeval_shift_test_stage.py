"""Dataset Shift Object Detection Test Stage Implementation"""

import maite.protocols.object_detection as od

from jatic_ri._common.test_stages.impls.dataeval_shift_test_stage import DatasetShiftTestStageBase


class DatasetShiftTestStage(DatasetShiftTestStageBase[od.Dataset]):
    """Detects dataset shift between two datasets using various methods"""

    _deck: str = "object_detection_dataset_evaluation"
    _task: str = "od"
