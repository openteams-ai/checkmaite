from typing import Any, Literal

import maite.protocols.image_classification as ic
import numpy as np
import pandas as pd
import pydantic
from dataeval.data import Embeddings, Metadata
from dataeval.metrics.estimators import BEROutput, ber
from gradient.slide_deck.shapes import SubText, Text
from gradient.templates_and_layouts.generic_layouts.section_by_item import SectionByItem
from pydantic import Field

from jatic_ri.core._types import Device
from jatic_ri.core._utils import get_resnet18, set_device
from jatic_ri.core.capability_core import CapabilityConfigBase, CapabilityRunBase, CapabilityRunner, Number
from jatic_ri.core.report._markdown import MarkdownOutput


class DataevalFeasibilityConfig(CapabilityConfigBase):
    """Configuration for the Image Classification Dataset Feasibility capability"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    batch_size: int = Field(default=1, description="Batch size to use when encoding images.")

    device: Device = pydantic.Field(default_factory=lambda: set_device("cpu"))
    ber_method: Literal["KNN", "MST"] = pydantic.Field(
        default="KNN", description="The method to use for the Bayes Error Rate"
    )
    knn_n_neighbors: int = pydantic.Field(default=1, description="The number of neighbors to use for the KNN method")
    precision: int = pydantic.Field(default=3, description="The number of decimal places to round the results to")


class DataevalFeasibilityOutputs(pydantic.BaseModel):
    """Container for the Feasibility capability outputs"""

    ber: float
    ber_lower: float


class DataevalFeasibilityRun(CapabilityRunBase[DataevalFeasibilityConfig, DataevalFeasibilityOutputs]):
    """Container for the Feasibility capability outputs and configuration"""

    config: DataevalFeasibilityConfig
    outputs: DataevalFeasibilityOutputs

    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:
        results = self.outputs

        dataset_id = self.dataset_metadata[0]["id"]

        is_feasible = results.ber > threshold
        feasibility_dict = {
            "Feasible": [str(is_feasible)],
            "Bayes Error Rate": [np.round(results.ber, self.config.precision)],
            "Lower Bayes Error Rate": [np.round(results.ber_lower, self.config.precision)],
            "Performance Goal": [threshold],
        }
        feasibility_df = pd.DataFrame.from_dict(feasibility_dict)

        title = f"Dataset: {dataset_id} | Category: Feasibility"
        heading = "Metric: Bayes Error Rate"
        content = [
            Text(t)
            for t in (
                [SubText("Result:", bold=True)],
                f"Performance goal of {threshold} {'is' if is_feasible else 'is NOT'} feasible.",
                [SubText("Tests for:", bold=True)],
                " * Achievability of performance goal",
                [SubText("Risk(s):", bold=True)],
                " * Performance goal cannot be achieved by any model (problem too hard)",
                " * Models that report performance above the goal are overfit and \
                will not generalize to real-world problems",
                [SubText("Action:", bold=True)],
                f"* {'No action required' if is_feasible else 'Reduce difficulty of the problem statement'}",
            )
        ]

        return [
            {
                "deck": self.capability_id,
                "layout_name": "SectionByItem",
                "layout_arguments": {
                    SectionByItem.ArgKeys.TITLE.value: title,
                    SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
                    SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
                    SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: feasibility_df,
                },
            },
        ]

    def collect_md_report(self, threshold: float) -> str:
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
        results = self.outputs
        dataset_id = self.dataset_metadata[0]["id"]
        is_feasible = results.ber > threshold

        md = MarkdownOutput("Dataset Feasibility Analysis")

        md.add_text(f"**Dataset**: {dataset_id}")
        md.add_text("**Category**: Feasibility")

        md.add_section(heading="Bayes Error Rate")
        md.add_text(f"**Result:** Performance goal of {threshold} {'is' if is_feasible else 'is NOT'} feasible.")
        md.add_blank_line()
        md.add_text("**Tests for:**")
        md.add_bulleted_list(["Achievability of performance goal"])
        md.add_text("**Risk(s):**")
        md.add_bulleted_list(
            [
                "Performance goal cannot be achieved by any model (problem too hard)",
                "Models that report performance above the goal are overfit and will not generalize to "
                "real-world problems",
            ]
        )
        md.add_text("**Action:**")
        action = "No action required" if is_feasible else "Reduce difficulty of the problem statement"
        md.add_bulleted_list([action])

        md.add_subsection(heading="Results")
        md.add_table(
            headers=["Metric", "Value"],
            rows=[
                ["Feasible", str(is_feasible)],
                ["Bayes Error Rate", str(np.round(results.ber, self.config.precision))],
                [
                    "Lower Bayes Error Rate",
                    str(np.round(results.ber_lower, self.config.precision)),
                ],
                ["Performance Goal", str(threshold)],
            ],
        )

        return md.render()


class DataevalFeasibility(
    CapabilityRunner[
        DataevalFeasibilityOutputs,
        ic.Dataset,
        ic.Model,
        ic.Metric,
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
        return Number.ZERO

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
        models: list[ic.Model],  # noqa: ARG002
        datasets: list[ic.Dataset],
        metrics: list[ic.Metric],  # noqa: ARG002
        config: DataevalFeasibilityConfig,
        use_prediction_and_evaluation_cache: bool,  # noqa: ARG002
    ) -> DataevalFeasibilityOutputs:
        dataset = datasets[0]

        model, transform = get_resnet18()
        embeddings = Embeddings(dataset, config.batch_size, transform, model, device=config.device)
        metadata = Metadata(dataset)

        b: BEROutput = ber(
            embeddings=embeddings.to_numpy(),
            labels=metadata.class_labels,
            method=config.ber_method,
            k=config.knn_n_neighbors,
        )

        b_data = b.data()
        return DataevalFeasibilityOutputs.model_validate({"ber": b_data["ber"], "ber_lower": b_data["ber_lower"]})
