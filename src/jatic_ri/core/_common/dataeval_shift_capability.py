from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pydantic
import torch
from dataeval.data import Embeddings
from dataeval.detectors.drift import DriftCVM, DriftKS, DriftMMD
from dataeval.detectors.ood import OOD_AE
from dataeval.outputs import DriftMMDOutput
from dataeval.utils.torch.models import AE
from gradient import SubText
from gradient.slide_deck.shapes import Text
from gradient.templates_and_layouts.generic_layouts.section_by_item import SectionByItem
from pydantic import Field

from jatic_ri.core._types import Device
from jatic_ri.core._utils import get_resnet18
from jatic_ri.core.capability_core import (
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    CapabilityRunner,
    Number,
    TDataset,
    TMetric,
    TModel,
)
from jatic_ri.core.report._markdown import MarkdownOutput


class DataevalShiftConfig(CapabilityConfigBase):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    device: Device = torch.device("cpu")
    dim: int = 128
    batch_size: int = Field(default=1, description="Batch size to use when encoding images.")


class DataevalShiftUnivariateOutput(CapabilityOutputsBase):
    drifted: bool
    threshold: float
    p_val: float
    distance: float
    feature_drift: np.ndarray
    feature_threshold: float
    p_vals: np.ndarray
    distances: np.ndarray


class DataevalShiftDriftOutputs(CapabilityOutputsBase):
    mmd: DriftMMDOutput
    cvm: DataevalShiftUnivariateOutput
    ks: DataevalShiftUnivariateOutput


class DataevalShiftOODAEOutput(CapabilityOutputsBase):
    is_ood: np.ndarray
    instance_score: np.ndarray
    feature_score: np.ndarray | None


class DataevalShiftOODOutputs(CapabilityOutputsBase):
    ood_ae: DataevalShiftOODAEOutput


class DataevalShiftOutputs(pydantic.BaseModel):
    drift: DataevalShiftDriftOutputs
    ood: DataevalShiftOODOutputs


class DataevalShiftRun(CapabilityRunBase[DataevalShiftConfig, DataevalShiftOutputs]):
    config: DataevalShiftConfig
    outputs: DataevalShiftOutputs

    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:  # noqa: ARG002
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


class DataevalShiftBase(CapabilityRunner[DataevalShiftOutputs, TDataset, TModel, TMetric, DataevalShiftConfig]):
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
        model, transform = get_resnet18(config.dim)
        emb_1 = Embeddings(dataset_1, config.batch_size, transform, model, device=config.device)
        emb_2 = Embeddings(dataset_2, config.batch_size, transform, model, device=config.device)

        if len(dataset_1) < 2 or len(dataset_2) < 2:
            raise ValueError(
                "Dataset drift detection is computed using unbiased statistics which require "
                f"at least 2 datapoints, {min(len(dataset_1), len(dataset_2))} "
                "datapoints were provided. Please provide more images as datapoints."
            )

        detectors = {
            "mmd": partial(DriftMMD, device=config.device),
            "cvm": DriftCVM,
            "ks": DriftKS,
        }

        return DataevalShiftDriftOutputs.model_validate(
            {name: detector(data=emb_1).predict(emb_2).data() for name, detector in detectors.items()}
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
        _, transform = get_resnet18(config.dim)
        emb_1 = Embeddings(
            dataset_1, config.batch_size, transform, torch.nn.Identity(), device=config.device
        ).to_numpy()
        emb_2 = Embeddings(
            dataset_2, config.batch_size, transform, torch.nn.Identity(), device=config.device
        ).to_numpy()

        detectors = {"ood_ae": OOD_AE(AE(emb_1[0].shape))}

        for detector in detectors.values():
            detector.fit(emb_1, threshold_perc=99, epochs=20, verbose=False)

        return DataevalShiftOODOutputs.model_validate(
            {name: detector.predict(emb_2).data() for name, detector in detectors.items()}
        )


def collect_drift(deck: str, drift_outputs: DataevalShiftDriftOutputs, *, dataset_ids: list[str]) -> dict[str, Any]:
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
        [SubText("Result:", bold=True)],
        f"• {dataset_ids[1]} has{' ' if any_drift else ' not '}drifted from {dataset_ids[0]}",
        [SubText("Tests for:", bold=True)],
        "• Covariate shift",
        [SubText("Risks:", bold=True)],
        "• Degradation of model performance",
        "• Real-world performance no longer meets performance requirements",
        [SubText("Action:", bold=True)],
        f"• {'Retrain model (augmentation, transfer learning)' if any_drift else 'No action required'}",
    ]
    content = [Text(t, fontsize=16) for t in text]

    return {
        "deck": deck,
        "layout_name": "SectionByItem",
        "layout_arguments": {
            SectionByItem.ArgKeys.TITLE.value: title,
            SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
            SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
            SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: drift_df,
        },
    }


def collect_ood(deck: str, ood_outputs: DataevalShiftOODOutputs, *, dataset_ids: list[str]) -> dict[str, Any]:
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
        [SubText("Result:", bold=True)],
        f"• {max(percents)}% OOD images were found in {dataset_ids[1]}",
        [SubText("Tests for:", bold=True)],
        f"• {dataset_ids[1]} data that is OOD from {dataset_ids[0]}",
        [SubText("Risks:", bold=True)],
        "• Degradation of model performance",
        "• Real-world performance no longer meets requirements",
        [SubText("Action:", bold=True)],
        f"• {'Retrain model (augmentation, transfer learning)' if sum(percents) else 'No action required'}",
        f"{'• Examine OOD samples to learn source of covariate shift' if sum(percents) else ''}",
    ]

    content = [Text(t, fontsize=16) for t in text]

    return {
        "deck": deck,
        "layout_name": "SectionByItem",
        "layout_arguments": {
            SectionByItem.ArgKeys.TITLE.value: title,
            SectionByItem.ArgKeys.LINE_SECTION_HALF.value: True,
            SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
            SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
            SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: ood_df,
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
            "p_val": field_value.p_val,
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
        ]
        for field_name, d in drift_fields.items()
    ]
    md.add_table(
        headers=["Method", "Has drifted?", "Test statistic", "P-value"],
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
