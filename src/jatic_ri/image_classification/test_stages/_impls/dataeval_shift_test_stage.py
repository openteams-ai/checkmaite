"""Dataset Shift Image Classification Test Stage Implementation"""

import maite.protocols.image_classification as ic

from jatic_ri._common.test_stages.impls.dataeval_shift_test_stage import DatasetShiftTestStageBase


class DatasetShiftTestStage(DatasetShiftTestStageBase[ic.Dataset, ic.Model, ic.Metric]):
    """Detects dataset shift between two image classification datasets using various methods"""

    _task: str = "ic"
