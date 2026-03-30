from checkmaite.core.analytics_store._schema import BaseRecord


class DataevalFeasibilityRecord(BaseRecord, table_name="dataeval_feasibility"):
    """Record for DataevalFeasibility capability results.

    Stores dataset feasibility summary metrics including Bayes Error Rate
    bounds. OD-specific fields (num_instances, num_classes, health stats)
    are None for image classification runs.
    """

    dataset_id: str

    # BER bounds (always present).
    # IC outputs call the upper bound ``ber``; OD outputs call it ``ber_upper``.
    # Both map to ``ber_upper`` here for a unified schema.
    ber_upper: float
    ber_lower: float

    # OD-specific fields (None for IC runs)
    num_instances: int | None = None
    num_classes: int | None = None
    small_object_ratio: float | None = None
    truncated_bbox_ratio: float | None = None
    overlap_image_ratio: float | None = None
    health_warning_count: int | None = None
