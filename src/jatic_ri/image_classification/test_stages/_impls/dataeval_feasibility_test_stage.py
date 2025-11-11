"""DataEval Feasibility Test Stage"""

from typing import Any, Literal

import maite.protocols.image_classification as ic
import numpy as np
import pandas as pd
import pydantic
from dataeval.data import Embeddings, Metadata
from dataeval.metrics.estimators import BEROutput, ber
from gradient.slide_deck.shapes import SubText, Text
from gradient.templates_and_layouts.generic_layouts.section_by_item import SectionByItem

from jatic_ri._common.models import set_device
from jatic_ri._common.test_stages.impls._dataeval_utils import get_resnet18
from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, Number, RunBase, TestStage
from jatic_ri.util._types import Device


class DatasetImageClassificationFeasibilityConfig(ConfigBase):
    """Configuration for the Image Classification Dataset Feasibility Test Stage"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    device: Device = pydantic.Field(default_factory=lambda: set_device("cpu"))
    ber_method: Literal["KNN", "MST"] = pydantic.Field(
        default="KNN", description="The method to use for the Bayes Error Rate"
    )
    knn_n_neighbors: int = pydantic.Field(default=1, description="The number of neighbors to use for the KNN method")
    precision: int = pydantic.Field(default=3, description="The number of decimal places to round the results to")


class DatasetImageClassificationFeasibilityOutputs(pydantic.BaseModel):
    """Container for the Image Classification Feasibility Test Stage outputs"""

    ber: float
    ber_lower: float


class DatasetImageClassificationFeasibilityRun(RunBase):
    """Container for the Image Classification Feasibility Test Stage outputs and configuration"""

    config: DatasetImageClassificationFeasibilityConfig
    outputs: DatasetImageClassificationFeasibilityOutputs

    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:
        """Create slides for Gradient report"""

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
                "deck": self.test_stage_id,
                "layout_name": "SectionByItem",
                "layout_arguments": {
                    SectionByItem.ArgKeys.TITLE.value: title,
                    SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
                    SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
                    SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: feasibility_df,
                },
            },
        ]


class DatasetImageClassificationFeasibilityTestStage(
    TestStage[DatasetImageClassificationFeasibilityOutputs, ic.Dataset, ic.Model, ic.Metric]
):
    """
    Measures whether the available data (both quantity and quality) can be used to
    satisfy the necessary performance characteristics of the machine learning model
    and programatically generates a Gradient report with the results.
    """

    _task: str = "ic"

    _RUN_TYPE = DatasetImageClassificationFeasibilityRun

    def __init__(
        self,
        device: str = "cpu",
        ber_method: Literal["KNN", "MST"] = "KNN",
        knn_n_neighbors: int = 1,
        precision: int = 3,
    ) -> None:
        super().__init__()
        self.device = set_device(device)
        self.ber_method: Literal["KNN", "MST"] = ber_method
        self.knn_n_neighbors = knn_n_neighbors
        self.precision = precision

    def _create_config(self) -> ConfigBase:
        return DatasetImageClassificationFeasibilityConfig(
            device=self.device,
            ber_method=self.ber_method,
            knn_n_neighbors=self.knn_n_neighbors,
            precision=self.precision,
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
        return Number.ZERO

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
        models: list[ic.Model],  # noqa: ARG002
        datasets: list[ic.Dataset],
        metrics: list[ic.Metric],  # noqa: ARG002
    ) -> DatasetImageClassificationFeasibilityOutputs:
        """Run the feasibility test"""

        dataset = datasets[0]

        model, transform = get_resnet18()
        embeddings = Embeddings(dataset, self._batch_size, transform, model, device=self.device)
        metadata = Metadata(dataset)

        b: BEROutput = ber(
            embeddings=embeddings.to_numpy(),
            labels=metadata.class_labels,
            method=self.ber_method,
            k=self.knn_n_neighbors,
        )

        b_data = b.data()
        return DatasetImageClassificationFeasibilityOutputs.model_validate(
            {"ber": b_data["ber"], "ber_lower": b_data["ber_lower"]}
        )
