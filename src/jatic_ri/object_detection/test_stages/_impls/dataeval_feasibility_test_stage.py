"""DataEval Feasibility Test Stage"""

from collections.abc import Sequence
from typing import Any

import maite.protocols.object_detection as od
import numpy as np
import pandas as pd
import pydantic
from dataeval.metrics.estimators import uap
from gradient.slide_deck.shapes import Text
from gradient.templates_and_layouts.generic_layouts.section_by_item import SectionByItem

from jatic_ri._common.models import set_device
from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, Number, OutputsBase, RunBase, TestStage
from jatic_ri.cached_tasks import predict
from jatic_ri.util._types import Device


class DatasetObjectDetectionFeasibilityConfig(ConfigBase):
    """Configuration for the Object Detection Dataset Feasibility Test Stage"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    device: Device = pydantic.Field(default_factory=lambda: set_device("cpu"))


class DatasetObjectDetectionFeasibilityOutputs(OutputsBase):
    """Container for the Object Detection Feasibility Test Stage outputs"""

    uap: np.ndarray


class DatasetObjectDetectionFeasibilityRun(RunBase):
    """Container for the Object Detection Feasibility Test Stage outputs and configuration"""

    config: DatasetObjectDetectionFeasibilityConfig
    outputs: DatasetObjectDetectionFeasibilityOutputs

    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:
        """Create slides for Gradient report

        Parameters
        ----------
        threshold : float
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, where each dictionary represents a slide
            for the Gradient report.
        """

        raise ValueError("Feasibility test for Object Detection is not possible until MAITE>=0.8.0 is supported.")

        # the following code is not supported until MAITE>=0.8.0 is supported
        # it requires that target.scores be a 2D array, not a 1D array
        # NOTE: code will need to be updated to fail gracefully if 1D array is encountered
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


class DatasetObjectDetectionFeasibilityTestStage(
    TestStage[DatasetObjectDetectionFeasibilityOutputs, od.Dataset, od.Model, od.Metric]
):
    """
    Measures whether the available data (both quantity and quality) can be used to
    satisfy the necessary performance characteristics of the machine learning model
    and programatically generates a Gradient report with the results.
    """

    _task: str = "od"

    _RUN_TYPE = DatasetObjectDetectionFeasibilityRun

    def __init__(
        self,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = set_device(device)

    def _create_config(self) -> ConfigBase:
        return DatasetObjectDetectionFeasibilityConfig(
            device=self.device,
        )

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

    def _run(
        self,
        models: list[od.Model],  # noqa: ARG002
        datasets: list[od.Dataset],  # noqa: ARG002
        metrics: list[od.Metric],  # noqa: ARG002
    ) -> DatasetObjectDetectionFeasibilityOutputs:
        """Run the feasibility test"""

        raise ValueError("Feasibility test for Object Detection is not possible until MAITE>=0.8.0 is supported.")

        # the following code is not supported until MAITE>=0.8.0 is supported
        # it requires that target.scores be a 2D array, not a 1D array
        # NOTE: code will need to be updated to fail gracefully if 1D array is encountered

        predictions, _ = predict(
            model=self.model,
            dataset=self.dataset,
            dataset_id=self.dataset_id,
            batch_size=32,
        )

        targets: Sequence[od.TargetType] = []
        for batch in predictions:
            targets.extend(batch)

        labels = np.array([target.labels for target in targets]).flatten()

        unique_labels = set(labels)
        scores = [target.scores for target in targets]
        scores = np.array(scores).reshape(-1, len(unique_labels))

        return DatasetObjectDetectionFeasibilityOutputs(uap=uap(labels=labels, scores=scores).uap)
