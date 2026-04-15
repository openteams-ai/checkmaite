from checkmaite.core.analytics_store._schema import BaseRecord
from checkmaite.core.capability_core import (
    Capability,
    Number,
    TConfig,
    TDataset,
    TMetric,
    TModel,
    TOutputs,
)


class XaitkExplainableBase(Capability[TOutputs, TDataset, TModel, TMetric, TConfig]):
    """Xai capability that takes in the necessary arguements to demo saliency map generation."""

    @property
    def supports_datasets(self) -> Number:
        """Number of datasets this capability supports.
        Returns
        -------
        Number
            An enumeration value indicating dataset support.
        """
        return Number.ONE

    @property
    def supports_models(self) -> Number:
        """Number of models this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating model support.
        """
        return Number.ONE

    @property
    def supports_metrics(self) -> Number:
        """Number of metrics this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating metric support.
        """
        return Number.ZERO

    @property
    def name(self) -> str:
        return self.__class__.__name__


class XaitkExplainableRecord(BaseRecord, table_name="xaitk_explainable"):
    """Record for XaitkExplainable capability results.

    One record is emitted per saliency map. For IC, this means one record
    per (image, class) pair. For OD, one record per detection. OD-specific
    fields are None for IC runs.

    Attributes
    ----------
    dataset_id : str
        Dataset identifier (cross-capability JOIN key).
    model_id : str
        Model identifier.
    saliency_generator_type : str
        Short class name of the saliency generator (e.g. ``"RISEStack"``).
    image_index : int
        Position of the image in the dataset (0-based).
    gt_label : str | None
        Ground truth label (IC only).
    image_id : str | None
        Image ID from dataset metadata (OD only).
    detection_index : int | None
        Which detection within the image (OD only, 0-based).
    predicted_label : str | None
        Predicted class label (OD only).
    confidence : float | None
        Detection confidence score (OD only).
    mean_saliency : float
        Mean pixel value across the saliency map.
    max_saliency : float
        Maximum pixel value in the saliency map.
    std_saliency : float
        Standard deviation of pixel values.
    positive_saliency_ratio : float
        Fraction of pixels with value > 0.
    """

    # Cross-capability JOIN key
    dataset_id: str

    # Entity identifiers
    model_id: str

    # Saliency configuration
    saliency_generator_type: str  # e.g. "RISEStack", "DRISEStack"

    # Image identification
    image_index: int  # position in dataset (0-based)

    # IC-specific fields (None for OD runs)
    gt_label: str | None = None

    # OD-specific fields (None for IC runs)
    image_id: str | None = None
    detection_index: int | None = None
    predicted_label: str | None = None
    confidence: float | None = None

    # Saliency map summary statistics
    mean_saliency: float
    max_saliency: float
    std_saliency: float
    positive_saliency_ratio: float  # fraction of pixels > 0
