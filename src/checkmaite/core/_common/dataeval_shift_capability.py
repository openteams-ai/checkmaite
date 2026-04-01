import logging
from typing import Any

import numpy as np
import pandas as pd
import pydantic
import torch
from dataeval import Embeddings
from dataeval.extractors import TorchExtractor
from dataeval.shift import DriftMMD, DriftOutput, DriftUnivariate, OODKNeighbors
from pydantic import Field

from checkmaite.core._common.feature_extractor import load_feature_extractor, pca_projector
from checkmaite.core._types import Device, ModelSpec, TorchvisionModelSpec
from checkmaite.core._utils import deprecated, requires_optional_dependency
from checkmaite.core.analytics_store._schema import BaseRecord
from checkmaite.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
    TDataset,
    TMetric,
    TModel,
)
from checkmaite.core.report import _gradient as gd
from checkmaite.core.report._markdown import MarkdownOutput

logger = logging.getLogger(__name__)


class DataevalShiftConfig(CapabilityConfigBase):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    device: Device = torch.device("cpu")
    batch_size: int = Field(default=1, description="Batch size to use when encoding images.")
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


class DataevalShiftDriftOutputs(CapabilityOutputsBase):
    # We can not put DriftOutput[DriftMMD.Stats] type specifications
    # due dataeval using TypedDict from typing and
    # Pydantic Error: pydantic.errors.PydanticUserError: Please use `typing_extensions.TypedDict`
    # instead of `typing.TypedDict` on Python < 3.12.
    mmd: DriftOutput  #  [DriftMMD.Stats]
    cvm: DriftOutput  #  [DriftUnivariate.Stats]
    ks: DriftOutput  #  [DriftUnivariate.Stats]


class DataevalShiftOODKNNOutput(CapabilityOutputsBase):
    is_ood: np.ndarray
    instance_score: np.ndarray
    feature_score: np.ndarray | None


class DataevalShiftOODOutputs(CapabilityOutputsBase):
    ood_knn: DataevalShiftOODKNNOutput


class DataevalShiftOutputs(pydantic.BaseModel):
    drift: DataevalShiftDriftOutputs
    ood: DataevalShiftOODOutputs


class DataevalShiftRecord(BaseRecord, table_name="dataeval_shift"):
    """Record for DataevalShift capability results.

    Stores drift detection and out-of-distribution (OOD) summary metrics.
    Shift is a two-dataset capability (reference vs evaluation), so it uses
    ``reference_dataset_id`` and ``evaluation_dataset_id`` instead of the
    single ``dataset_id`` convention used by single-dataset capabilities.
    """

    reference_dataset_id: str
    evaluation_dataset_id: str

    # Drift: MMD (Maximum Mean Discrepancy)
    mmd_drifted: bool
    mmd_distance: float
    mmd_p_val: float
    mmd_threshold: float

    # Drift: CVM (Cramer-von Mises)
    cvm_drifted: bool
    cvm_distance: float
    cvm_p_val: float
    cvm_threshold: float
    cvm_feature_drift_count: int  # number of individually drifted features

    # Drift: KS (Kolmogorov-Smirnov)
    ks_drifted: bool
    ks_distance: float
    ks_p_val: float
    ks_threshold: float
    ks_feature_drift_count: int  # number of individually drifted features

    # OOD: KNN summary (aggregated from per-sample arrays)
    ood_count: int
    ood_total: int
    ood_ratio: float
    ood_mean_instance_score: float
    ood_std_instance_score: float
    ood_max_instance_score: float


class DataevalShiftRun(CapabilityRunBase[DataevalShiftConfig, DataevalShiftOutputs]):
    config: DataevalShiftConfig
    outputs: DataevalShiftOutputs

    def extract(self) -> list[DataevalShiftRecord]:
        """Extract summary metrics from this DataevalShift run.

        Returns a single record with drift test results (MMD, CVM, KS)
        and aggregated OOD statistics from per-sample arrays.
        """
        drift = self.outputs.drift
        ood_knn = self.outputs.ood.ood_knn

        is_ood = ood_knn.is_ood
        ood_count = int(np.sum(is_ood))
        ood_total = len(is_ood)

        return [
            DataevalShiftRecord(
                run_uid=self.run_uid,
                reference_dataset_id=self.dataset_metadata[0]["id"],
                evaluation_dataset_id=self.dataset_metadata[1]["id"],
                # MMD
                mmd_drifted=drift.mmd.drifted,
                mmd_distance=drift.mmd.distance,
                mmd_p_val=drift.mmd.details["p_val"],
                mmd_threshold=drift.mmd.threshold,
                # CVM
                cvm_drifted=drift.cvm.drifted,
                cvm_distance=drift.cvm.distance,
                cvm_p_val=drift.cvm.details["p_val"],
                cvm_threshold=drift.cvm.threshold,
                cvm_feature_drift_count=int(np.sum(drift.cvm.details["feature_drift"])),
                # KS
                ks_drifted=drift.ks.drifted,
                ks_distance=drift.ks.distance,
                ks_p_val=drift.ks.details["p_val"],
                ks_threshold=drift.ks.threshold,
                ks_feature_drift_count=int(np.sum(drift.ks.details["feature_drift"])),
                # OOD aggregates
                ood_count=ood_count,
                ood_total=ood_total,
                ood_ratio=ood_count / ood_total if ood_total > 0 else 0.0,
                ood_mean_instance_score=float(np.mean(ood_knn.instance_score)) if ood_total > 0 else 0.0,
                ood_std_instance_score=float(np.std(ood_knn.instance_score)) if ood_total > 0 else 0.0,
                ood_max_instance_score=float(np.max(ood_knn.instance_score)) if ood_total > 0 else 0.0,
            )
        ]

    @requires_optional_dependency("gradient", install_hint="pip install '.[unsupported]'")
    @deprecated(replacement="collect_md_report")
    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:  # noqa: ARG002 # pragma: no cover
        """Convert results from drift and OOD detection into a Gradient-consumable list of slides.

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

        outputs = self.outputs

        deck = self.capability_id

        report_consumables = []

        dataset_ids = [d["id"] for d in self.dataset_metadata]

        drift_slide = collect_drift(deck, outputs.drift, dataset_ids=dataset_ids)
        report_consumables.append(drift_slide)

        ood_slide = collect_ood(deck, outputs.ood, dataset_ids=dataset_ids)
        report_consumables.append(ood_slide)

        return report_consumables

    def collect_md_report(self, threshold: float) -> str:  # noqa: ARG002
        """Convert results from drift and OOD detection into Markdown format.

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
        outputs = self.outputs
        dataset_ids = [d["id"] for d in self.dataset_metadata]

        md = MarkdownOutput("Dataset Shift Analysis")

        collect_drift_md(md, outputs.drift, dataset_ids=dataset_ids)

        md.add_section_divider()

        collect_ood_md(md, outputs.ood, dataset_ids=dataset_ids)

        return md.render()


class DataevalShiftBase(Capability[DataevalShiftOutputs, TDataset, TModel, TMetric, DataevalShiftConfig]):
    """Detects dataset shift between two datasets using various methods.

    Performs three drift detection and two out-of-distribution tests
    against dataset 2 using dataset 1 as the reference.
    - Drift: Maximum mean discrepancy, Cramer-von Mises, and Kolmogorov-Smirnov
    - OOD: Autoencoder (AE)

    Attributes
    ----------
    _RUN_TYPE
        The type of the run object associated with this capability.
    """

    _RUN_TYPE = DataevalShiftRun

    @classmethod
    def _create_config(cls) -> DataevalShiftConfig:
        return DataevalShiftConfig()

    @property
    def supports_datasets(self) -> Number:
        """Number of datasets this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating dataset support.
        """
        return Number.TWO

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
        models: list[TModel],  # noqa: ARG002
        datasets: list[TDataset],
        metrics: list[TMetric],  # noqa: ARG002
        config: DataevalShiftConfig,
        use_prediction_and_evaluation_cache: bool,  # noqa: ARG002
    ) -> DataevalShiftOutputs:
        """Run methods for drift and OOD detectors.

        Returns
        -------
            An object containing the combined outputs from drift and OOD
            detection.
        """
        dataset_1 = datasets[0]
        dataset_2 = datasets[1]
        return DataevalShiftOutputs.model_validate(
            {
                "drift": self._run_drift(dataset_1=dataset_1, dataset_2=dataset_2, config=config),
                "ood": self._run_ood(dataset_1=dataset_1, dataset_2=dataset_2, config=config),
            }
        )

    def _run_drift(
        self, dataset_1: TDataset, dataset_2: TDataset, config: DataevalShiftConfig
    ) -> DataevalShiftDriftOutputs:
        """Run MMD, CVM, and KS drift detection methods.

        Compares embeddings of`dataset_2` against`dataset_1`.

        Returns
        -------
        An object containing the outputs from MMD, CVM, and KS drift
        detectors.
        """

        # Drift feature extraction logic:
        # Use the same feature extractor + preprocessing on both datasets so “differences” reflect data shift.
        # Optionally fit PCA on reference embeddings (dataset_1) and apply same projection to both sides to
        # compare in lower-dimensional space without leaking information from dataset_2 into projection.
        fe = load_feature_extractor(device=config.device, model_spec=config.feature_extractor_spec)

        if len(dataset_1) < 2 or len(dataset_2) < 2:
            raise ValueError(
                "Dataset drift detection is computed using unbiased statistics which require "
                f"at least 2 datapoints, {min(len(dataset_1), len(dataset_2))} "
                "datapoints were provided. Please provide more images as datapoints."
            )
        extractor = TorchExtractor(fe.model, transforms=fe.transforms, device=config.device)
        emb_1 = Embeddings(
            dataset_1,
            extractor=extractor,
            batch_size=config.batch_size,
        )[:]

        emb_2 = Embeddings(
            dataset_2,
            extractor=extractor,
            batch_size=config.batch_size,
        )[:]

        n1, d1 = emb_1.shape

        if config.target_embedding_dim < d1:
            k_max = min(n1, d1)
            k = min(config.target_embedding_dim, k_max)

            if k != config.target_embedding_dim:
                logger.warning(
                    f"Requested target_embedding_dim={config.target_embedding_dim}, but PCA is limited to "
                    f"min(N_ref, D)={k_max} (N_ref={n1}, D={d1}). Using {k} components.",
                )

            if k < d1:
                proj = pca_projector(emb_1, out_dim=k)
                emb_1 = proj.transform(emb_1)
                emb_2 = proj.transform(emb_2)
        else:
            logger.warning(
                f"Cannot reduce embeddings to target_embedding_dim={config.target_embedding_dim}: feature "
                f"extractor already outputs {d1}-D embeddings. Returning {d1}-D embeddings without PCA.",
            )

        detectors = {
            "mmd": DriftMMD(device=config.device),
            "cvm": DriftUnivariate(method="cvm"),
            "ks": DriftUnivariate(method="ks"),
        }
        return DataevalShiftDriftOutputs.model_validate(
            {name: detector.fit(emb_1).predict(emb_2).data() for name, detector in detectors.items()}
        )

    def _run_ood(
        self, dataset_1: TDataset, dataset_2: TDataset, config: DataevalShiftConfig
    ) -> DataevalShiftOODOutputs:
        """Run Autoencoder (AE) based Out-of-Distribution (OOD) detection.

         Trains an AE on`dataset_1` and then detects OOD samples in
        `dataset_2`.

        Returns
        -------
        An object containing the outputs from the OOD AE detector.
        """
        # OOD feature extraction logic:
        # Embed both datasets with the same feature extractor so OOD scoring compares like-with-like.
        # consistent space. Optionally compress embeddings with PCA fitted on reference set (dataset_1).
        fe = load_feature_extractor(device=config.device, model_spec=config.feature_extractor_spec)
        extractor = TorchExtractor(fe.model, transforms=fe.transforms, device=config.device)
        emb_1 = Embeddings(
            dataset_1,
            extractor=extractor,
            batch_size=config.batch_size,
        )[:]

        emb_2 = Embeddings(
            dataset_2,
            extractor=extractor,
            batch_size=config.batch_size,
        )[:]

        n1, d = emb_1.shape

        if config.target_embedding_dim < d:
            k_max = min(n1, d)
            k = min(config.target_embedding_dim, k_max)

            if k != config.target_embedding_dim:
                logger.warning(
                    f"Requested target_embedding_dim={config.target_embedding_dim}, but PCA is limited to "
                    f"min(N_ref, D)={k_max} (N_ref={n1}, D={d}). Using {k} components.",
                )

            if k < d:
                proj = pca_projector(emb_1, out_dim=k)
                emb_1 = proj.transform(emb_1)
                emb_2 = proj.transform(emb_2)
        else:
            logger.warning(
                f"Cannot reduce embeddings to target_embedding_dim={config.target_embedding_dim}: feature "
                f"extractor already outputs {d}-D embeddings. Returning {d}-D embeddings without PCA.",
            )

        detectors = {"ood_knn": OODKNeighbors(k=min(10, n1 - 1), threshold_perc=99, distance_metric="cosine")}

        for detector in detectors.values():
            detector.fit(emb_1)

        return DataevalShiftOODOutputs.model_validate(
            {name: detector.predict(emb_2).data() for name, detector in detectors.items()}
        )


def collect_drift(
    deck: str, drift_outputs: DataevalShiftDriftOutputs, *, dataset_ids: list[str]
) -> dict[str, Any]:  # pragma: no cover
    """Generate Gradient compliant kwargs for drift detection results.

    Uses outputs from MMD, KS, and CVM methods.

    Parameters
    ----------
    drift_outputs
        Outputs from the drift detection methods (MMD, KS, CVM).
    dataset_ids
        A list containing the IDs of the two datasets being compared.

    Returns
    -------
    A dictionary representing a`SectionByItem` slide, containing
    drift analysis text and a corresponding DataFrame.
    """
    drift_fields = {
        field_name: {"drifted": field_value.drifted, "distance": field_value.distance, "p_val": field_value.p_val}
        for field_name, field_value in drift_outputs
    }

    any_drift = any(d["drifted"] for d in drift_fields.values())

    drift_df = pd.DataFrame(
        {
            "Method": [drift_output[0] for drift_output in drift_outputs],
            "Has drifted?": ["Yes" if d["drifted"] else "No" for d in drift_fields.values()],
            "Test statistic": [d["distance"] for d in drift_fields.values()],
            "P-value": [d["p_val"] for d in drift_fields.values()],
        },
    )

    title = f"Dataset 1: {dataset_ids[0]} - Dataset 2: {dataset_ids[1]} | Category: Dataset Shift"
    heading = "Metric: Drift"
    text = [
        [gd.SubText("Result:", bold=True)],
        f"• {dataset_ids[1]} has{' ' if any_drift else ' not '}drifted from {dataset_ids[0]}",
        [gd.SubText("Tests for:", bold=True)],
        "• Covariate shift",
        [gd.SubText("Risks:", bold=True)],
        "• Degradation of model performance",
        "• Real-world performance no longer meets performance requirements",
        [gd.SubText("Action:", bold=True)],
        f"• {'Retrain model (augmentation, transfer learning)' if any_drift else 'No action required'}",
    ]
    content = [gd.Text(t, fontsize=16) for t in text]

    return {
        "deck": deck,
        "layout_name": "SectionByItem",
        "layout_arguments": {
            gd.SectionByItem.ArgKeys.TITLE.value: title,
            gd.SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
            gd.SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
            gd.SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: drift_df,
        },
    }


def collect_ood(
    deck: str, ood_outputs: DataevalShiftOODOutputs, *, dataset_ids: list[str]
) -> dict[str, Any]:  # pragma: no cover
    """Parse OOD results into a report-consumable slide.

    Creates a table showing the number of OOD samples and the threshold
    used for their calculation.
    Note: Image path access and index retrieval from the dataset are not
    currently implemented.

    Parameters
    ----------
    ood_outputs
        Outputs from the OOD detection methods, containing`is_ood`,
        `instance_scores`, and`feature_scores`.
    dataset_ids
        A list containing the IDs of the two datasets.

    Returns
    -------
    A dictionary representing a`SectionByItem` slide, containing
    OOD analysis text and a corresponding DataFrame.
    """
    ood_fields = {
        field_name: {"is_ood": field_value.is_ood, "instance_score": field_value.instance_score}
        for field_name, field_value in ood_outputs
        if hasattr(field_value, "is_ood") and hasattr(field_value, "instance_score")
    }

    percents = np.array([round(np.sum(x["is_ood"]) * 100 / len(x["is_ood"]), 1) for x in ood_fields.values()])
    counts = np.array([np.sum(x["is_ood"], dtype=int) for x in ood_fields.values()])

    # NOTE: Not the true threshold. Currently not available in OODOutput so smallest score found in outliers used
    # Gets minimum of outlier-only scores or takes max of all scores if no outliers
    thresholds = [
        np.min(x["instance_score"][x["is_ood"]]) if sum(x["is_ood"]) else np.max(x["instance_score"])
        for x in ood_fields.values()
    ]

    ood_df = pd.DataFrame(
        {
            "Method": [ood_output[0] for ood_output in ood_outputs],
            "OOD Count": counts,
            "OOD Percent": percents,
            "Threshold": thresholds,
        },
    )

    title = f"Dataset 1: {dataset_ids[0]} - Dataset 2: {dataset_ids[1]} | Category: Dataset Shift"
    heading = "Metric: Out-of-distribution (OOD)"
    text = [
        [gd.SubText("Result:", bold=True)],
        f"• {max(percents)}% OOD images were found in {dataset_ids[1]}",
        [gd.SubText("Tests for:", bold=True)],
        f"• {dataset_ids[1]} data that is OOD from {dataset_ids[0]}",
        [gd.SubText("Risks:", bold=True)],
        "• Degradation of model performance",
        "• Real-world performance no longer meets requirements",
        [gd.SubText("Action:", bold=True)],
        f"• {'Retrain model (augmentation, transfer learning)' if sum(percents) else 'No action required'}",
        f"{'• Examine OOD samples to learn source of covariate shift' if sum(percents) else ''}",
    ]

    content = [gd.Text(t, fontsize=16) for t in text]

    return {
        "deck": deck,
        "layout_name": "SectionByItem",
        "layout_arguments": {
            gd.SectionByItem.ArgKeys.TITLE.value: title,
            gd.SectionByItem.ArgKeys.LINE_SECTION_HALF.value: True,
            gd.SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
            gd.SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
            gd.SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: ood_df,
        },
    }


# ============================================================================
# Markdown Report Generation Functions
# ============================================================================


def collect_drift_md(
    md: MarkdownOutput,
    drift_outputs: DataevalShiftDriftOutputs,
    *,
    dataset_ids: list[str],
) -> None:
    """Generate Markdown content for drift detection results.

    Uses outputs from MMD, KS, and CVM methods.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    drift_outputs : DataevalShiftDriftOutputs
        Outputs from the drift detection methods (MMD, KS, CVM).
    dataset_ids : list[str]
        A list containing the IDs of the two datasets being compared.
    """
    drift_fields = {
        field_name: {
            "drifted": field_value.drifted,
            "distance": field_value.distance,
            "p_val": field_value.details["p_val"],
            "feature_drift": field_value.details.get("feature_drift", None),
        }
        for field_name, field_value in drift_outputs
    }

    any_drift = any(d["drifted"] for d in drift_fields.values())

    md.add_text(f"**Dataset 1**: {dataset_ids[0]}")
    md.add_text(f"**Dataset 2**: {dataset_ids[1]}")
    md.add_text("**Category**: Dataset Shift")

    md.add_section(heading="Drift Detection")
    md.add_text(f"**Result:** {dataset_ids[1]} has{' ' if any_drift else ' not '}drifted from {dataset_ids[0]}")
    md.add_blank_line()
    md.add_text("**Tests for:**")
    md.add_bulleted_list(["Covariate shift"])
    md.add_text("**Risks:**")
    md.add_bulleted_list(
        [
            "Degradation of model performance",
            "Real-world performance no longer meets performance requirements",
        ]
    )
    md.add_text("**Action:**")
    action = "Retrain model (augmentation, transfer learning)" if any_drift else "No action required"
    md.add_bulleted_list([action])

    md.add_subsection(heading="Drift Test Results")
    rows = [
        [
            field_name,
            "Yes" if d["drifted"] else "No",
            f"{d['distance']:.6f}",
            f"{d['p_val']:.6f}",
            str(int(np.sum(d["feature_drift"]))) if d["feature_drift"] is not None else "N/A",
        ]
        for field_name, d in drift_fields.items()
    ]
    md.add_table(
        headers=["Method", "Has drifted?", "Test statistic", "P-value", "Features drifted"],
        rows=rows,
    )


def collect_ood_md(
    md: MarkdownOutput,
    ood_outputs: DataevalShiftOODOutputs,
    *,
    dataset_ids: list[str],
) -> None:
    """Generate Markdown content for OOD detection results.

    Creates a table showing the number of OOD samples and the threshold
    used for their calculation.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    ood_outputs : DataevalShiftOODOutputs
        Outputs from the OOD detection methods, containing `is_ood`,
        `instance_scores`, and `feature_scores`.
    dataset_ids : list[str]
        A list containing the IDs of the two datasets.
    """
    ood_fields = {
        field_name: {
            "is_ood": field_value.is_ood,
            "instance_score": field_value.instance_score,
        }
        for field_name, field_value in ood_outputs
        if hasattr(field_value, "is_ood") and hasattr(field_value, "instance_score")
    }

    percents = np.array([round(np.sum(x["is_ood"]) * 100 / len(x["is_ood"]), 1) for x in ood_fields.values()])
    counts = np.array([np.sum(x["is_ood"], dtype=int) for x in ood_fields.values()])

    # NOTE: Not the true threshold. Currently not available in OODOutput so smallest score found in outliers used
    # Gets minimum of outlier-only scores or takes max of all scores if no outliers
    thresholds = [
        (np.min(x["instance_score"][x["is_ood"]]) if sum(x["is_ood"]) else np.max(x["instance_score"]))
        for x in ood_fields.values()
    ]

    md.add_section(heading="Out-of-Distribution (OOD) Detection")
    md.add_text(f"**Result:** {max(percents)}% OOD images were found in {dataset_ids[1]}")
    md.add_blank_line()
    md.add_text("**Tests for:**")
    md.add_bulleted_list([f"{dataset_ids[1]} data that is OOD from {dataset_ids[0]}"])
    md.add_text("**Risks:**")
    md.add_bulleted_list(
        [
            "Degradation of model performance",
            "Real-world performance no longer meets requirements",
        ]
    )
    md.add_text("**Action:**")
    actions = []
    if sum(percents):
        actions.append("Retrain model (augmentation, transfer learning)")
        actions.append("Examine OOD samples to learn source of covariate shift")
    else:
        actions.append("No action required")
    md.add_bulleted_list(actions)

    md.add_subsection(heading="OOD Test Results")
    rows = [
        [
            ood_output[0],
            str(counts[idx]),
            f"{percents[idx]:.1f}%",
            f"{thresholds[idx]:.6f}",
        ]
        for idx, ood_output in enumerate(ood_outputs)
    ]
    md.add_table(
        headers=["Method", "OOD Count", "OOD Percent", "Threshold"],
        rows=rows,
    )
