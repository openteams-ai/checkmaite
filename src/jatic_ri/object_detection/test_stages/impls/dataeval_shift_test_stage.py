"""Dataset Shift Test Stage Implementation"""

from functools import partial
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

# _run imports
from dataeval.detectors.drift import DriftCVM, DriftKS, DriftMMD
from dataeval.detectors.ood import OOD_AE, OOD_VAEGMM
from dataeval.utils.tensorflow.models import AE, VAEGMM, create_model
from dataeval.utils.torch import read_dataset

# report_consumable imports
from gradient.templates_and_layouts.generic_layouts.text_data import Text, TextData

from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import TwoDatasetPlugin
from jatic_ri.util.cache import JSONCache, NumpyEncoder


class DatasetShiftTestStage(TestStage[dict[str, Any]], TwoDatasetPlugin):
    """Detects dataset shift between two datasets using various methods

    Performs three drift detection and two out of distribution tests
    against dataset 2 using dataset 1 as the reference
    - Drift: Maximum mean discrepency, Cramer-von Mises, and Kolmogorov-Smirnov
    - OOD: AE, VAEGMM

    Attributes
    ----------
    outputs: Optional[dict[str, Any]], default None
        Dictionary where key is the metric category and values are method OutputClass results as dicts
    cache: Optional[Cache[dict[str, Any]]], default JSONCache(encoder=NumpyEncoder)
        Cache object that can load in pre-run results into self.outputs
    device: Literal["cpu"], default "cpu"
        The device to run preprocessing models on
    """

    cache: Optional[Cache[dict[str, Any]]] = JSONCache(encoder=NumpyEncoder)
    device = "cpu"

    @property
    def cache_id(self) -> str:
        """Unique identifier for cached results"""
        return f"shift_{self.dataset_1_id}_{self.dataset_2_id}.json"

    def _run(self) -> dict[str, Any]:
        """Run methods for drift and ood detectors"""

        images_1 = read_dataset(self.dataset_1)[0]  # type: ignore
        images_2 = read_dataset(self.dataset_2)[0]  # type: ignore

        return {
            **self._run_drift(images_1=images_1, images_2=images_2),
            **self._run_ood(images_1=images_1, images_2=images_2),
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

    def _run_drift(self, images_1: list[Any], images_2: list[Any]) -> dict[str, Any]:
        """Runs MMD, CVM, and KS methods against images"""

        imgs1 = torch.stack(images_1)
        imgs2 = torch.stack(images_2)
        kwargs = {"x_ref": imgs1}
        detectors = {
            "Maximum Mean Discrepency": partial(DriftMMD, device=self.device),
            "Cramér-von Mises": DriftCVM,
            "Kolmogorov-Smirnov": DriftKS,
        }

        outputs = {name: detector(**kwargs).predict(imgs2).dict() for name, detector in detectors.items()}
        return {"drift": outputs}

    def _run_ood(self, images_1: list[Any], images_2: list[Any]) -> dict[str, Any]:
        """Runs AE and VAEGMM ood methods against images"""

        imgs1 = np.asarray(images_1)
        imgs2 = np.asarray(images_2)

        input_shape = imgs1[0].shape

        ood_kwargs = {
            "threshold_perc": 99,
            "epochs": 20,
            "verbose": False,
        }

        detectors = {
            "OOD_AE": OOD_AE(create_model(AE, input_shape)),
            "OOD_VAEGMM": OOD_VAEGMM(create_model(VAEGMM, input_shape)),
        }

        for detector in detectors.values():
            detector.fit(imgs1, **ood_kwargs)

        outputs = {name: detector.predict(imgs2).dict() for name, detector in detectors.items()}
        return {"ood": outputs}

    def _collect_drift(self, outputs: dict[str, Any]) -> dict[str, Any]:
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
            "deck": "object_detection_dataset_evaluation",
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

        This function creates a table of the number of ood samples and the threshold used to calculate them.\n
        Currently `TextData` is used, but the `TextTableData` format would allow example ood instances to be included.
        These would include the image, the label, and the outlier score for 2-3 examples.
        At this moment, image path access as well as index retrieval from the dataset are unimplemented

        Parameters
        ----------
        outputs: dict[str, Any]
            The out of distribution specific results containing "is_ood", "instance_scores", and "feature_scores"

        Returns
        -------
        list[dict[str, Any]]
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
            "deck": "object_detection_dataset_evaluation",
            "layout_name": "TextData",
            "layout_arguments": {
                TextData.ArgKeys.TITLE.value: title,
                TextData.ArgKeys.TEXT_COLUMN_HALF.value: True,
                TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                TextData.ArgKeys.DATA_COLUMN_TABLE.value: ood_df,
            },
        }
