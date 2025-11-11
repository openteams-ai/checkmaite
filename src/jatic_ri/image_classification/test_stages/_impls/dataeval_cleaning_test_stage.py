"""DataEval Image Classification Cleaning Test Stage"""

import maite.protocols.image_classification as ic

from jatic_ri._common.test_stages.impls.dataeval_cleaning_test_stage import DatasetCleaningTestStageBase


class DatasetCleaningTestStage(DatasetCleaningTestStageBase[ic.Dataset, ic.Model, ic.Metric]):
    "Image classification cleaning test stage"

    _task: str = "ic"
