"""XAITKTestStage implementation"""

# Python generic imports
from __future__ import annotations

from typing import Generic, TypeVar

from smqtk_core.configuration import from_config_dict as from_config_dict

# Local imports
from jatic_ri._common.test_stages.interfaces.plugins import (
    SingleDatasetPlugin,
    SingleModelPlugin,
    TDataset,
    TModel,
)
from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, OutputsBase, TestStage

TOutputs = TypeVar("TOutputs", bound=OutputsBase)
TConfig = TypeVar("TConfig", bound=ConfigBase)


class XAITKTestStageBase(
    TestStage[TOutputs],
    SingleModelPlugin[TModel],
    SingleDatasetPlugin[TDataset],
    Generic[TConfig, TOutputs, TModel, TDataset],
):
    """
    XAITK Test Stage that takes in the necessary arguements to demo saliency map generation.

    Attributes:
        config: The configuration model, specified in the subclass for the sub-problem (e.g. OD or IC).
    """

    config: TConfig

    def _create_config(self) -> TConfig:
        return self.config

    @property
    def name(self) -> str:
        """Returns classname as a string"""
        return self.__class__.__name__
