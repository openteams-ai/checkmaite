import maite.protocols.image_classification as ic

from jatic_ri.core._common.dataeval_shift_capability import DataevalShiftBase


class DataevalShift(DataevalShiftBase[ic.Dataset, ic.Model, ic.Metric]):
    """Detects dataset shift between two image classification datasets using various methods"""
