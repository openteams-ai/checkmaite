import logging
from typing import Any

import maite.protocols.image_classification as ic
import numpy as np
import pandas as pd
import pydantic
from dataeval import Embeddings, Metadata
from dataeval.extractors import TorchExtractor
from pydantic import Field

from checkmaite.core._common._knn import compute_ber_knn
from checkmaite.core._common.dataeval_feasibility_record import DataevalFeasibilityRecord
from checkmaite.core._common.feature_extractor import load_feature_extractor, pca_projector, to_unit_interval_01
from checkmaite.core._types import Device, ModelSpec, TorchvisionModelSpec
from checkmaite.core._utils import deprecated, requires_optional_dependency, set_device
from checkmaite.core.capability_core import Capability, CapabilityConfigBase, CapabilityRunBase, Number
from checkmaite.core.report import _gradient as gd
from checkmaite.core.report._markdown import MarkdownOutput

logger = logging.getLogger(__name__)


class DataevalFeasibilityConfig(CapabilityConfigBase):
    """Configuration for the Image Classification Dataset Feasibility capability"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    batch_size: int = Field(default=1, description="Batch size to use when encoding images.")

    device: Device = pydantic.Field(default_factory=lambda: set_device("cpu"))
    knn_n_neighbors: int = pydantic.Field(
        default=1, description="The number of neighbors to use for kNN BER estimation."
    )
    precision: int = pydantic.Field(default=3, description="The number of decimal places to round the results to")
    feature_extractor_spec: ModelSpec = Field(
        default_factory=TorchvisionModelSpec,
        description=(
            "Spec for model used to turn each image into a numeric feature vector (embeddings). "
            "This is not the model-under-test; it's just for representation."
        ),
    )
    target_embedding_dim: int = Field(
        default=256,
        ge=1,
        description=(
            "Requested embedding size after any optional PCA step. "
            "If the extractor outputs more than this, we reduce with PCA; "
            "if it outputs <= this, we keep the native size."
        ),
    )


class DataevalFeasibilityOutputs(pydantic.BaseModel):
    """Container for the Feasibility capability outputs"""

    ber: float
    ber_lower: float


class DataevalFeasibilityRun(CapabilityRunBase[DataevalFeasibilityConfig, DataevalFeasibilityOutputs]):
    """Container for the Feasibility capability outputs and configuration"""

    config: DataevalFeasibilityConfig
    outputs: DataevalFeasibilityOutputs

    def extract(self) -> list[DataevalFeasibilityRecord]:
        """Extract summary metrics from this DataevalFeasibility run.

        Returns a single record per dataset with BER bounds.
        OD-specific fields are left as None for IC runs.
        """
        return [
            DataevalFeasibilityRecord(
                run_uid=self.run_uid,
                dataset_id=self.dataset_metadata[0]["id"],
                # IC outputs call the upper bound ``ber``; record uses ``ber_upper``
                # for a unified schema with OD (which uses ``ber_upper`` natively).
                ber_upper=self.outputs.ber,
                ber_lower=self.outputs.ber_lower,
            )
        ]

    # The order is important
    @requires_optional_dependency("gradient", install_hint="pip install '.[unsupported]'")
    @deprecated(replacement="collect_md_report")
    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:
        results = self.outputs

        dataset_id = self.dataset_metadata[0]["id"]

        is_feasible = (1.0 - results.ber) >= threshold
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
            gd.Text(t)
            for t in (
                [gd.SubText("Result:", bold=True)],
                f"Performance goal of {threshold} {'is' if is_feasible else 'is NOT'} feasible.",
                [gd.SubText("Tests for:", bold=True)],
                " * Achievability of performance goal",
                [gd.SubText("Risk(s):", bold=True)],
                " * Performance goal cannot be achieved by any model (problem too hard)",
                " * Models that report performance above the goal are overfit and \
                will not generalize to real-world problems",
                [gd.SubText("Action:", bold=True)],
                f"* {'No action required' if is_feasible else 'Reduce difficulty of the problem statement'}",
            )
        ]

        return [
            {
                "deck": self.capability_id,
                "layout_name": "SectionByItem",
                "layout_arguments": {
                    gd.SectionByItem.ArgKeys.TITLE.value: title,
                    gd.SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
                    gd.SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
                    gd.SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: feasibility_df,
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
        is_feasible = (1.0 - results.ber) >= threshold

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
    Capability[
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

        # Turn each image into feature vector using pre-trained feature extractor + matching preprocessing.
        # If extractor outputs more dimensions than needed, fit PCA on dataset and project down to
        # target dimension. Downstream BER code assumes values live on a [0,1] so rescale.
        fe = load_feature_extractor(device=config.device, model_spec=config.feature_extractor_spec)
        extractor = TorchExtractor(fe.model, transforms=fe.transforms, device=config.device)
        embeddings = Embeddings(
            dataset,
            extractor=extractor,
            batch_size=config.batch_size,
        )[:]

        n, d = embeddings.shape

        if config.target_embedding_dim < d:
            k_max = min(n, d)
            k = min(config.target_embedding_dim, k_max)

            if k != config.target_embedding_dim:
                logger.warning(
                    f"Requested target_embedding_dim={config.target_embedding_dim}, but PCA is limited to "
                    f"min(N, D)={k_max} (N={n}, D={d}). Using {k} components.",
                )

            if k < d:
                proj = pca_projector(embeddings, out_dim=k)
                embeddings = proj.transform(embeddings)
        else:
            logger.warning(
                f"Cannot reduce embeddings to target_embedding_dim={config.target_embedding_dim}: feature "
                f"extractor already outputs {d}-D embeddings. Returning {d}-D embeddings without PCA.",
            )

        embeddings_01 = to_unit_interval_01(embeddings)
        # ber expects 2D array, shape (N, P), P embedding dimensions
        if embeddings_01.ndim == 1:
            embeddings_01 = embeddings_01[None, :]

        metadata = Metadata(dataset)

        ber_upper, ber_lower = compute_ber_knn(
            embeddings=embeddings_01,
            labels=metadata.class_labels,
            k=config.knn_n_neighbors,
        )

        return DataevalFeasibilityOutputs.model_validate({"ber": ber_upper, "ber_lower": ber_lower})
