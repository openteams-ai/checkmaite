import maite.protocols.image_classification as ic

from jatic_ri.core._common.dataeval_bias_capability import DataevalBiasBase


class DataevalBias(DataevalBiasBase[ic.Dataset, ic.Model, ic.Metric]):
    """
    Measures four aspects of bias in a single dataset and programmatically generates a Gradient report
    with the measurements of bias, potential risks, and any actions required to reduce bias if found

    Bias is measured using four metrics: balance, coverage, diversity, parity.

    Balance, diversity, and parity calculate different aspects of correlation
    between metadata factors and class labels, while coverage is calculated using only the images
    """
