from collections.abc import Sequence
from typing import Any

import maite.protocols.object_detection as od
import numpy as np
import pandas as pd
import pydantic
from dataeval.metrics.estimators import uap
from gradient.slide_deck.shapes import Text
from gradient.templates_and_layouts.generic_layouts.section_by_item import SectionByItem

from jatic_ri.core._types import Device
from jatic_ri.core._utils import set_device
from jatic_ri.core.cached_tasks import predict
from jatic_ri.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
)


class DataevalFeasibilityConfig(CapabilityConfigBase):
    """Configuration for the Feasibility Capability"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    device: Device = pydantic.Field(default_factory=lambda: set_device("cpu"))


class DataevalFeasibilityOutputs(CapabilityOutputsBase):
    """Container for the Feasibility outputs"""

    uap: np.ndarray


class DataevalFeasibilityRun(CapabilityRunBase[DataevalFeasibilityConfig, DataevalFeasibilityOutputs]):
    """Container for the Feasibility outputs and configuration"""

    config: DataevalFeasibilityConfig
    outputs: DataevalFeasibilityOutputs

    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:
        """Create slides for Gradient report

        Parameters
        ----------
        threshold
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
            A list of dictionaries, where each dictionary represents a slide
            for the Gradient report.
        """

        raise ValueError("Feasibility capability for Object Detection is not possible until MAITE>=0.8.0 is supported.")

        # the following code is not supported until MAITE>=0.8.0 is supported
        # it requires that target.scores be a 2D array, not a 1D array
        # NOTE: code will need to be updated to fail gracefully if 1D array is encountered
        # When this gets updated, also update the markdown report method below
        uap = self.outputs["uap"]
        threshold = self.threshold
        is_feasible = uap >= threshold

        feasibility_df = pd.DataFrame(
            {
                "Feasible": [str(is_feasible)],
                "Upperbound Average Precision": [round(uap, 3)],
                "Performance Goal": [round(threshold, 3)],
            },
        )

        title = f"Dataset: {self.dataset_id} | Category: Feasibility"
        heading = "Metric: UAP"
        text = [
            "**Result:**",
            f"* Performance goal of {threshold} {'is' if is_feasible else 'is NOT'} feasible.",
            "**Tests for:**",
            "* Achievability of performance goal",
            "**Risks:**",
            "* Performance goal cannot be achieved by any model",
            "* Models that report performance above the bound are overfit and will not generalize to real-world problems",  # noqa: E501
            "**Action:**",
            f"* {'No action required' if is_feasible else 'Reduce difficulty of the problem statement'}",
        ]
        content = [Text(t, fontsize=16) for t in text]

        feasibility_slide_args = {
            "deck": "object_detection_dataset_evaluation",
            "layout_name": "SectionByItem",
            "layout_arguments": {
                SectionByItem.ArgKeys.TITLE.value: title,
                SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
                SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
                SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: feasibility_df,
            },
        }

        return [feasibility_slide_args]

    def collect_md_report(self, threshold: float) -> str:  # noqa: ARG002
        """Create Markdown report for feasibility analysis.

        Parameters
        ----------
        threshold : float
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
        str
            Markdown-formatted report content.
        """
        raise ValueError("Feasibility test for Object Detection is not possible until MAITE>=0.8.0 is supported.")


class DataevalFeasibility(
    Capability[
        DataevalFeasibilityOutputs,
        od.Dataset,
        od.Model,
        od.Metric,
        DataevalFeasibilityConfig,
    ]
):
    """
    Measures whether the available data (both quantity and quality) can be used to
    satisfy the necessary performance characteristics of the machine learning model
    and programatically generates a Gradient report with the results.
    """

    _RUN_TYPE = DataevalFeasibilityRun

    @classmethod
    def _create_config(cls) -> DataevalFeasibilityConfig:
        return DataevalFeasibilityConfig()

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

    def _run(
        self,
        models: list[od.Model],  # noqa: ARG002
        datasets: list[od.Dataset],  # noqa: ARG002
        metrics: list[od.Metric],  # noqa: ARG002
        config: DataevalFeasibilityConfig,  # noqa: ARG002
        use_prediction_and_evaluation_cache: bool,
    ) -> DataevalFeasibilityOutputs:
        """Run the feasibility capability"""

        raise ValueError("Feasibility capability for Object Detection is not possible until MAITE>=0.8.0 is supported.")

        # the following code is not supported until MAITE>=0.8.0 is supported
        # it requires that target.scores be a 2D array, not a 1D array
        # NOTE: code will need to be updated to fail gracefully if 1D array is encountered

        predictions, _ = predict(
            model=self.model,
            dataset=self.dataset,
            dataset_id=self.dataset_id,
            batch_size=32,
            use_cache=use_prediction_and_evaluation_cache,
        )

        targets: Sequence[od.TargetType] = []
        for batch in predictions:
            targets.extend(batch)

        labels = np.array([target.labels for target in targets]).flatten()

        unique_labels = set(labels)
        scores = [target.scores for target in targets]
        scores = np.array(scores).reshape(-1, len(unique_labels))

        return DataevalFeasibilityOutputs(uap=uap(labels=labels, scores=scores).uap)
