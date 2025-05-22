"""Dataset Shift Object Detection Test Stage Implementation"""

import maite.protocols.object_detection as od

from jatic_ri._common.test_stages.impls.dataeval_shift_test_stage import DatasetShiftTestStageBase


class DatasetShiftTestStage(DatasetShiftTestStageBase[od.Dataset]):
    """Detects dataset shift between two datasets using various methods

    Performs three drift detection and two out of distribution tests
    against dataset 2 using dataset 1 as the reference
    - Drift: Maximum mean discrepency, Cramer-von Mises, and Kolmogorov-Smirnov
    - OOD: AE, VAEGMM

    Attributes
    ----------
    outputs : dict[str, Any] | None, default None
        Dictionary where key is the metric category and values are method OutputClass results as dicts
    cache : Cache[dict[str, Any]] | None, default JSONCache(encoder=NumpyEncoder)
        Cache object that can load in pre-run results into self.outputs
    device : Literal["cpu"], default "cpu"
        The device to run preprocessing models on
    deck_name : Literal["object_detection_dataset_evaluation"]
        Title slide of the gradient PowerPoint
    """

    _deck: str = "object_detection_dataset_evaluation"
    _task: str = "od"
