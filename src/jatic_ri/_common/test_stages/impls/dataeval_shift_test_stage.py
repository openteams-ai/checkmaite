"""DataEval classes and methods between image classification and object detection test stages"""

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch

# _run imports
from dataeval.data import Embeddings
from dataeval.detectors.drift import DriftCVM, DriftKS, DriftMMD
from dataeval.detectors.ood import OOD_AE
from dataeval.utils.torch.models import AE

# report_consumable imports
from gradient import SubText
from gradient.slide_deck.shapes import Text
from gradient.templates_and_layouts.generic_layouts.section_by_item import SectionByItem

from jatic_ri._common.test_stages.impls._dataeval_utils import get_resnet18
from jatic_ri._common.test_stages.interfaces.plugins import TDataset, TwoDatasetPlugin
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.util.cache import JSONCache, NumpyEncoder


class DatasetShiftTestStageBase(TestStage[dict[str, Any]], TwoDatasetPlugin[TDataset]):
    """Detects dataset shift between two datasets using various methods

    Performs three drift detection and two out of distribution tests
    against dataset 2 using dataset 1 as the reference
    - Drift: Maximum mean discrepency, Cramer-von Mises, and Kolmogorov-Smirnov
    - OOD: AE, VAEGMM

    Attributes
    ----------
    outputs : dict[str, Any] | None, default None
        Dictionary where key is the metric category and values are method OutputClass results as dicts
    cache : Cache[dict[str, Any]] | None, default JSONCache(encoder=NumpyEncoder)
        Cache object that can load in pre-run results into self.outputs
    device : Literal["cpu"], default "cpu"
        The device to run preprocessing models on
    _deck : str
        Deck name used for Gradient's Title slide. Should be overwritten in subclasses
    """

    cache: Cache[dict[str, Any]] | None = JSONCache(encoder=NumpyEncoder)
    device = "cpu"
    _deck: str
    _task: str

    # TODO: move to config
    _dim: int = 128

    def __init__(self) -> None:
        super().__init__()

    @property
    def cache_id(self) -> str:
        """Unique identifier for cached results"""
        return f"shift_{self._task}_{self.dataset_1_id}_{self.dataset_2_id}.json"

    def _run(self) -> dict[str, Any]:
        """Run methods for drift and ood detectors"""

        return {
            **self._run_drift(),
            **self._run_ood(),
        }

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Converts results from drift and ood into a Gradient consumable list of slides"""

        report_consumables = []

        # Adds drift slide
        drift_slide = self._collect_drift(self.outputs["drift"])
        report_consumables.append(drift_slide)

        # Adds ood slide
        ood_slide = self._collect_ood(self.outputs["ood"])
        report_consumables.append(ood_slide)

        return report_consumables

    def _run_drift(self) -> dict[str, Any]:
        """Runs MMD, CVM, and KS methods against images"""

        model, transform = get_resnet18(self._dim)
        emb_1 = Embeddings(self.dataset_1, self._batch_size, transform, model, self.device)
        emb_2 = Embeddings(self.dataset_2, self._batch_size, transform, model, self.device)

        kwargs = {"data": emb_1}
        detectors = {
            "Maximum Mean Discrepency": partial(DriftMMD, device=self.device),
            "Cramér-von Mises": DriftCVM,
            "Kolmogorov-Smirnov": DriftKS,
        }

        outputs = {name: detector(**kwargs).predict(emb_2).data() for name, detector in detectors.items()}
        return {"drift": outputs}

    def _run_ood(self) -> dict[str, Any]:
        """Runs AE and VAEGMM ood methods against images"""

        ood_kwargs = {
            "threshold_perc": 99,
            "epochs": 20,
            "verbose": False,
        }

        _, transform = get_resnet18(self._dim)
        emb_1 = Embeddings(self.dataset_1, self._batch_size, transform, torch.nn.Identity(), self.device).to_numpy()
        emb_2 = Embeddings(self.dataset_2, self._batch_size, transform, torch.nn.Identity(), self.device).to_numpy()

        # Additional detectors may be added in the future
        detectors = {"OOD_AE": OOD_AE(AE(emb_1[0].shape))}

        for detector in detectors.values():
            detector.fit(emb_1, **ood_kwargs)

        outputs = {name: detector.predict(emb_2).data() for name, detector in detectors.items()}
        return {"ood": outputs}

    def _collect_drift(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Generates gradient compliant kwargs using the drift outputs of MMD, KS, and CVM methods

        Parameters
        ----------
        outputs:
            Drift method output classes of MMD, KS, and CVM as dictionaries

        Returns
        -------
        dict
            Single `SectionByItem` slide containing drift text and corresponding dataframe
        """
        any_drift = any(d["drifted"] for d in outputs.values())

        drift_df = pd.DataFrame(
            {
                "Method": list(outputs),
                "Has drifted?": ["Yes" if d["drifted"] else "No" for d in outputs.values()],
                "Test statistic": [d["distance"] for d in outputs.values()],
                "P-value": [d["p_val"] for d in outputs.values()],
            },
        )

        # Gradient slide kwargs
        title = f"Dataset 1: {self.dataset_1_id} - Dataset 2: {self.dataset_2_id} | Category: Dataset Shift"
        heading = "Metric: Drift"
        text = [
            [SubText("Result:", bold=True)],
            f"• {self.dataset_2_id} has{' ' if any_drift else ' not '}drifted from {self.dataset_1_id}",
            [SubText("Tests for:", bold=True)],
            "• Covariate shift",
            [SubText("Risks:", bold=True)],
            "• Degradation of model performance",
            "• Real-world performance no longer meets performance requirements",
            [SubText("Action:", bold=True)],
            f"• {'Retrain model (augmentation, transfer learning)' if any_drift else 'No action required'}",
        ]
        content = [Text(t, fontsize=16) for t in text]

        # Set up Gradient slide
        return {
            "deck": self._deck,
            "layout_name": "SectionByItem",
            "layout_arguments": {
                SectionByItem.ArgKeys.TITLE.value: title,
                SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
                SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
                SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: drift_df,
            },
        }

    def _collect_ood(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """
        Parses ood results into a report consumable slide

        This function creates a table of the number of ood samples and the threshold used to calculate them.
        At this moment, image path access as well as index retrieval from the dataset are unimplemented

        Parameters
        ----------
        outputs:
            The out of distribution specific results containing "is_ood", "instance_scores", and "feature_scores"

        Returns
        -------
        dict
            Single `SectionByItem` slide containing OOD text and corresponding dataframe
        """

        percents = np.array([round(np.sum(x["is_ood"]) * 100 / len(x["is_ood"]), 1) for x in outputs.values()])
        counts = np.array([np.sum(x["is_ood"], dtype=int) for x in outputs.values()])

        # NOTE: Not the true threshold. Currently not available in OODOutput so smallest score found in outliers used
        # Gets minimum of outlier-only scores or takes max of all scores if no outliers
        thresholds = [
            np.min(x["instance_score"][x["is_ood"]]) if sum(x["is_ood"]) else np.max(x["instance_score"])
            for x in outputs.values()
        ]

        ood_df = pd.DataFrame(
            {
                "Method": list(outputs),
                "OOD Count": counts,
                "OOD Percent": percents,
                "Threshold": thresholds,
            },
        )

        title = f"Dataset 1: {self.dataset_1_id} - Dataset 2: {self.dataset_2_id} | Category: Dataset Shift"
        heading = "Metric: Out-of-distribution (OOD)"
        text = [
            [SubText("Result:", bold=True)],
            f"• {max(percents)}% OOD images were found in {self.dataset_2_id}",
            [SubText("Tests for:", bold=True)],
            f"• {self.dataset_2_id} data that is OOD from {self.dataset_1_id}",
            [SubText("Risks:", bold=True)],
            "• Degradation of model performance",
            "• Real-world performance no longer meets requirements",
            [SubText("Action:", bold=True)],
            f"• {'Retrain model (augmentation, transfer learning)' if sum(percents) else 'No action required'}",
            f"{'• Examine OOD samples to learn source of covariate shift' if sum(percents) else ''}",
        ]

        content = [Text(t, fontsize=16) for t in text]

        return {
            "deck": self._deck,
            "layout_name": "SectionByItem",
            "layout_arguments": {
                SectionByItem.ArgKeys.TITLE.value: title,
                SectionByItem.ArgKeys.LINE_SECTION_HALF.value: True,
                SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
                SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
                SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: ood_df,
            },
        }
