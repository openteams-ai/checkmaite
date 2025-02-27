"""DataEval classes and methods between image classification and object detection test stages"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd

# _run imports
from dataeval.detectors.drift import DriftCVM, DriftKS, DriftMMD
from dataeval.detectors.ood import OOD_AE
from dataeval.utils.torch.models import AE

# report_consumable imports
from gradient.slide_deck.shapes import Text
from gradient.templates_and_layouts.generic_layouts.text_data import TextData
from numpy.typing import NDArray

from jatic_ri._common.test_stages.interfaces.plugins import TDataset, TwoDatasetPlugin
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.util.cache import JSONCache, NumpyEncoder

from ._dataeval_utils import EmbeddingNet, extract_embeddings


class DatasetShiftTestStageBase(TestStage[dict[str, Any]], TwoDatasetPlugin[TDataset]):
    """Detects dataset shift between two datasets using various methods

    Performs three drift detection and two out of distribution tests
    against dataset 2 using dataset 1 as the reference
    - Drift: Maximum mean discrepency, Cramer-von Mises, and Kolmogorov-Smirnov
    - OOD: AE, VAEGMM

    Attributes
    ----------
    outputs : Optional[dict[str, Any]], default None
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

    def __init__(self) -> None:
        super().__init__()
        self._embedding_net = EmbeddingNet().to(self.device)

    @property
    def cache_id(self) -> str:
        """Unique identifier for cached results"""
        return f"shift_{self._task}_{self.dataset_1_id}_{self.dataset_2_id}.json"

    def _run(self) -> dict[str, Any]:
        """Run methods for drift and ood detectors"""

        images_1 = [image for image, *_ in self.dataset_1]
        images_2 = [image for image, *_ in self.dataset_2]

        return {
            **self._run_drift(images_1=images_1, images_2=images_2),
            **self._run_ood(images_1=np.array(images_1), images_2=np.array(images_2)),
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

    def _run_drift(self, images_1: list[NDArray], images_2: list[NDArray]) -> dict[str, Any]:
        """Runs MMD, CVM, and KS methods against images"""

        embeddings_1 = extract_embeddings(images_1, embedding_net=self._embedding_net, device=self.device).cpu()
        embeddings_2 = extract_embeddings(images_2, embedding_net=self._embedding_net, device=self.device).cpu()

        kwargs = {"x_ref": embeddings_1}
        detectors = {
            "Maximum Mean Discrepency": partial(DriftMMD, device=self.device),
            "Cramér-von Mises": DriftCVM,
            "Kolmogorov-Smirnov": DriftKS,
        }

        outputs = {name: detector(**kwargs).predict(embeddings_2).dict() for name, detector in detectors.items()}
        return {"drift": outputs}

    def _run_ood(self, images_1: NDArray[Any], images_2: NDArray[Any]) -> dict[str, Any]:
        """Runs AE and VAEGMM ood methods against images"""

        input_shape = images_1[0].shape

        ood_kwargs = {
            "threshold_perc": 99,
            "epochs": 20,
            "verbose": False,
        }

        # Additional detectors may be added in the future
        detectors = {"OOD_AE": OOD_AE(AE(input_shape))}

        for detector in detectors.values():
            detector.fit(images_1, **ood_kwargs)

        outputs = {name: detector.predict(images_2).dict() for name, detector in detectors.items()}
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
            Single `TextData` slide containing drift text and corresponding dataframe
        """
        any_drift = any(d["is_drift"] for d in outputs.values())

        drift_df = pd.DataFrame(
            {
                "Method": list(outputs),
                "Has drifted?": ["Yes" if d["is_drift"] else "No" for d in outputs.values()],
                "Test statistic": [d["distance"] for d in outputs.values()],
                "P-value": [d["p_val"] for d in outputs.values()],
            },
        )

        # Gradient slide kwargs
        title = f"Dataset 1: {self.dataset_1_id} - Dataset 2: {self.dataset_2_id} | Category: Dataset Shift"
        heading = "Metric: Drift"
        text = [
            "**Result:**",
            f"* {self.dataset_2_id} has{' ' if any_drift else ' not '}drifted from {self.dataset_1_id}",
            "**Tests for:**",
            "* Covariate shift",
            "**Risks:**",
            "* Degradation of model performance",
            "* Real-world performance no longer meets performance requirements",
            "**Action:**",
            f"* {'Retrain model (augmentation, transfer learning)' if any_drift else 'No action required'}",
        ]
        content = [Text(t, fontsize=16) for t in text]

        # Set up Gradient slide
        return {
            "deck": self._deck,
            "layout_name": "TextData",
            "layout_arguments": {
                TextData.ArgKeys.TITLE.value: title,
                TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                TextData.ArgKeys.DATA_COLUMN_TABLE.value: drift_df,
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
            Single `TextData` slide containing OOD text and corresponding dataframe
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
            "**Result:**",
            f"* {max(percents)}% OOD images were found in {self.dataset_2_id}",
            "**Tests for:**",
            f"* {self.dataset_2_id} data that is OOD from {self.dataset_1_id}",
            "**Risks:**",
            "* Degradation of model performance",
            "* Real-world performance no longer meets requirements",
            "**Action:**",
            f"* {'Retrain model (augmentation, transfer learning)' if sum(percents) else 'No action required'}",
            f"{'* Examine OOD samples to learn source of covariate shift' if sum(percents) else ''}",
        ]

        content = [Text(t, fontsize=16) for t in text]

        return {
            "deck": self._deck,
            "layout_name": "TextData",
            "layout_arguments": {
                TextData.ArgKeys.TITLE.value: title,
                TextData.ArgKeys.TEXT_COLUMN_HALF.value: True,
                TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                TextData.ArgKeys.DATA_COLUMN_TABLE.value: ood_df,
            },
        }
