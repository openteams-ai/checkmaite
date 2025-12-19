from smqtk_core.configuration import from_config_dict as from_config_dict

from jatic_ri.core.capability_core import (
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
