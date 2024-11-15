"""Dataset Shift Image Classification Test Stage Implementation"""

import maite.protocols.image_classification as ic

from jatic_ri._common.test_stages.impls.dataeval_shift_test_stage import DatasetShiftTestStageBase


class DatasetShiftTestStage(DatasetShiftTestStageBase[ic.Dataset]):
    """Detects dataset shift between two image classification datasets using various methods

    Performs three drift detection and two out of distribution tests
    against dataset 2 using dataset 1 as the reference
    - Drift: Maximum mean discrepency, Cramer-von Mises, and Kolmogorov-Smirnov
    - OOD: AE, VAEGMM

    Attributes
    ----------
    outputs : Optional[dict[str, Any]], default None
        Dictionary where key is the metric category and values are method OutputClass results as dicts
    cache : Optional[Cache[dict[str, Any]]], default JSONCache(encoder=NumpyEncoder)
        Cache object that can load in pre-run results into self.outputs
    device : Literal["cpu"], default "cpu"
        The device to run preprocessing models on
    deck_name : Literal["image_classification_dataset_evaluation"]
        Title slide of the gradient PowerPoint
    """

    _deck: str = "image_classification_dataset_evaluation"
    _task: str = "ic"
