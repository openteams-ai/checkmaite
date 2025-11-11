"""DataEval Object Detection Bias Test Stage"""

import maite.protocols.object_detection as od

from jatic_ri._common.test_stages.impls.dataeval_bias_test_stage import DatasetBiasTestStageBase


class DatasetBiasTestStage(DatasetBiasTestStageBase[od.Dataset, od.Model, od.Metric]):
    """
    Measures four aspects of bias in a single dataset and programmatically generates a Gradient report
    with the measurements of bias, potential risks, and any actions required to reduce bias if found

    Bias is measured using four metrics: balance, coverage, diversity, parity.

    Balance, diversity, and parity calculate different aspects of correlation
    between metadata factors and class labels, while coverage is calculated using only the images
    """

    _task: str = "od"
    # Coco dataset loader includes these by default as metadata, excluded here as not useful to analysis
    _metadata_to_exclude: list[str] = ["id", "file_name"]
