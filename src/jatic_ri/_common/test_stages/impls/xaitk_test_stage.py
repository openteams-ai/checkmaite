"""XAITKTestStage implementation"""

from smqtk_core.configuration import from_config_dict as from_config_dict

from jatic_ri._common.test_stages.interfaces.test_stage import Number, TDataset, TestStage, TMetric, TModel, TOutputs


class XAITKTestStageBase(TestStage[TOutputs, TDataset, TModel, TMetric]):
    """XAITK Test Stage that takes in the necessary arguements to demo saliency map generation."""

    @property
    def supports_datasets(self) -> Number:
        """Number of datasets this test stage supports.

        Returns
        -------
        Number
            An enumeration value indicating dataset support.
        """
        return Number.ONE

    @property
    def supports_models(self) -> Number:
        """Number of models this test stage supports.

        Returns
        -------
        Number
            An enumeration value indicating model support.
        """
        return Number.ONE

    @property
    def supports_metrics(self) -> Number:
        """Number of metrics this test stage supports.

        Returns
        -------
        Number
            An enumeration value indicating metric support.
        """
        return Number.ZERO

    @property
    def name(self) -> str:
        """Returns classname as a string"""
        return self.__class__.__name__
