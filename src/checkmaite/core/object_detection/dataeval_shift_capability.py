import maite.protocols.object_detection as od

from checkmaite.core._common.dataeval_shift_capability import DataevalShiftBase


class DataevalShift(DataevalShiftBase[od.Dataset, od.Model, od.Metric]):
    """Detects dataset shift between two datasets using various methods"""
