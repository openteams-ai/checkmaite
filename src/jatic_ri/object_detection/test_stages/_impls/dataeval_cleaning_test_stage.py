"""DataEval Object Detection Cleaning Test Stage"""

import maite.protocols.object_detection as od

from jatic_ri._common.test_stages.impls.dataeval_cleaning_test_stage import (
    DatasetCleaningTestStageBase,
)


class DatasetCleaningTestStage(DatasetCleaningTestStageBase[od.Dataset, od.Model, od.Metric]):
    "Object detection cleaning test stage"

    _deck: str = "object_detection_cleaning_evaluation"
    _task: str = "od"
