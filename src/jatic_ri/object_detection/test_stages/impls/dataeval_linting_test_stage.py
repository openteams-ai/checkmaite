"""DataEval Linting Object Detection Test Stage"""

from __future__ import annotations

from pathlib import Path
from string import capwords
from typing import Any

import numpy as np
import pandas as pd
from dataeval._internal.interop import as_numpy
from dataeval.detectors.linters import Duplicates, Outliers
from dataeval.metrics.stats import DimensionStatsOutput, PixelStatsOutput, VisualStatsOutput, datasetstats, hashstats
from gradient.slide_deck.shapes import Text
from gradient.templates_and_layouts.generic_layouts.text_table_images import TextTableImages
from matplotlib import patches
from matplotlib import pyplot as plt
from more_itertools import take

from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import SingleDatasetPlugin
from jatic_ri.util.cache import JSONCache, NumpyEncoder

MU = "\u03bc"
IMAGE_ARGKEYS = [
    [
        TextTableImages.ArgKeys.DATA_COLUMN_1_IMAGE_1.value,
        TextTableImages.ArgKeys.DATA_COLUMN_1_IMAGE_2.value,
        TextTableImages.ArgKeys.DATA_COLUMN_1_IMAGE_3.value,
    ],
    [
        TextTableImages.ArgKeys.DATA_COLUMN_2_IMAGE_1.value,
        TextTableImages.ArgKeys.DATA_COLUMN_2_IMAGE_2.value,
        TextTableImages.ArgKeys.DATA_COLUMN_2_IMAGE_3.value,
    ],
    [
        TextTableImages.ArgKeys.DATA_COLUMN_3_IMAGE_1.value,
        TextTableImages.ArgKeys.DATA_COLUMN_3_IMAGE_2.value,
        TextTableImages.ArgKeys.DATA_COLUMN_3_IMAGE_3.value,
    ],
    [
        TextTableImages.ArgKeys.DATA_COLUMN_4_IMAGE_1.value,
        TextTableImages.ArgKeys.DATA_COLUMN_4_IMAGE_2.value,
        TextTableImages.ArgKeys.DATA_COLUMN_4_IMAGE_3.value,
    ],
    [
        TextTableImages.ArgKeys.DATA_COLUMN_5_IMAGE_1.value,
        TextTableImages.ArgKeys.DATA_COLUMN_5_IMAGE_2.value,
        TextTableImages.ArgKeys.DATA_COLUMN_5_IMAGE_3.value,
    ],
    [
        TextTableImages.ArgKeys.DATA_COLUMN_6_IMAGE_1.value,
        TextTableImages.ArgKeys.DATA_COLUMN_6_IMAGE_2.value,
        TextTableImages.ArgKeys.DATA_COLUMN_6_IMAGE_3.value,
    ],
    [
        TextTableImages.ArgKeys.DATA_COLUMN_7_IMAGE_1.value,
        TextTableImages.ArgKeys.DATA_COLUMN_7_IMAGE_2.value,
        TextTableImages.ArgKeys.DATA_COLUMN_7_IMAGE_3.value,
    ],
    [
        TextTableImages.ArgKeys.DATA_COLUMN_8_IMAGE_1.value,
        TextTableImages.ArgKeys.DATA_COLUMN_8_IMAGE_2.value,
        TextTableImages.ArgKeys.DATA_COLUMN_8_IMAGE_3.value,
    ],
]


class DatasetLintingTestStage(TestStage[dict[str, Any]], SingleDatasetPlugin):
    """
    Dataset Linting TestStage implementation.

    Performs dataset linting by identifying duplicates (exact and near) as well as statistical outliers
    using various pixel and image statistics on the image data.
    """

    cache: Cache[dict[str, Any]] | None = JSONCache(encoder=NumpyEncoder, compress=True)

    @property
    def cache_id(self) -> str:
        """Unique cache id for output"""
        return f"linting-{self.dataset_id}.dat"

    @property
    def cache_contents_path(self) -> Path:
        """Cache base folder for image artifacts"""
        return Path(self.cache_base_path, f"linting-{self.dataset_id}")

    def _run(self) -> dict[str, Any]:
        """Run linting"""

        hashes = hashstats((d[0] for d in self.dataset), (d[1].boxes for d in self.dataset))
        dupes = Duplicates().from_stats(hashes)
        stats = datasetstats((d[0] for d in self.dataset), (d[1].boxes for d in self.dataset))
        outliers = Outliers(outlier_method="zscore", outlier_threshold=3).from_stats(stats)

        return {
            "duplicates": dupes.dict(),
            "outliers": {int(k): v for k, v in outliers.issues.items()},
            "stats": stats.dict(),
        }

    def _to_pct_2(self, dividend: float, divisor: float) -> tuple[str, str]:
        """Format a dividend and divisor as tuple of 'percent%' 'dividend/divisor'"""
        return f"{dividend/divisor:.2%}", f"{dividend}/{divisor}"

    def _to_pct_1(self, dividend: float, divisor: float) -> str:
        """Format a dividend and divisor as a single string of 'percent% dividend/divisor'"""
        return " ".join(self._to_pct_2(dividend, divisor))

    def _to_title(self, text: str) -> str:
        """Format a string as a title by replacing '_' with ' ' and capitalizing"""
        return capwords(text.replace("_", " "))

    def _generate_duplicates_report(self) -> dict[str, Any]:
        """
        Generate the duplicates report which consists of a single slide with both
        exact duplicates and near duplicates.

        - Slide Format: `TextTableImages`
        - Text Content: Explanation of test, risks and action item
        - Table Content: Table of exact and near duplicates
        - Image Content: Sample images of identified exact and near duplicates
        """
        dupes: dict[str, list[list[int]]] = self.outputs["duplicates"]
        exact: list[list[int]] = dupes["exact"]
        near: list[list[int]] = dupes["near"]
        source_index = self.outputs["stats"]["source_index"]
        len_ds = len(source_index)

        total_ed = sum(len(d) for d in exact)
        total_nd = sum(len(d) for d in near)
        total = total_ed + total_nd

        duplicates_df = pd.DataFrame(
            {
                "Exact duplicates": [
                    *self._to_pct_2(total_ed, len_ds),
                    f"{len(exact)} unique",
                ],
                "Near duplicates": [
                    *self._to_pct_2(total_nd, len_ds),
                    f"{len(near)} unique",
                ],
            }
        )

        examples = [
            take(3, take(2, sorted(exact, key=lambda x: -len(x)))) if exact else [],
            take(3, take(2, sorted(near, key=lambda x: -len(x)))) if near else [],
        ]

        fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=100)
        self.cache_contents_path.mkdir(exist_ok=True)

        near_dp_kwargs = {}
        for i, nd in enumerate(examples):
            for j, dl in enumerate(nd):
                filepath = Path(self.cache_contents_path, f"dupe_{i}_{dl[0]}_{len(dl)}.png")
                if not filepath.exists():
                    for di, ax in zip(dl, axes):
                        ii, ti, _ = source_index[di]
                        data = self.dataset[ii]
                        image = data[0]
                        label = as_numpy(data[1].labels)[ti or 0]
                        mapping = getattr(self.dataset, "index2label", None)
                        label = label if mapping is None else mapping[label]
                        title = f"{label}"
                        ax.clear()
                        ax.axis("off")
                        ax.set_title(title)
                        ax.imshow(np.moveaxis(as_numpy(image), 0, -1))
                        if ti is not None:
                            box = as_numpy(data[1].boxes)[ti]
                            rect = patches.Rectangle(
                                (box[0], box[1]),
                                box[2] - box[0],
                                box[3] - box[1],
                                linewidth=2,
                                edgecolor="r",
                                facecolor="none",
                            )
                            ax.add_patch(rect)
                    fig.savefig(str(filepath), pad_inches=0.0)
                near_dp_kwargs[IMAGE_ARGKEYS[i][j]] = filepath

        plt.close(fig)

        # Gradient slide kwargs
        title = f"Dataset: {self.dataset_id} | Category: Linting"
        heading = "Duplicates"
        content = [
            Text(t, fontsize=14)
            for t in (
                "**Result:**",
                f"* Total Duplicates: {self._to_pct_1(total, len_ds)}",
                "**Tests for:**",
                "* Uncleaned data",
                "**Risks:**",
                "* Leakage",
                "* Lack of robustness",
                "* Poor real-world performance",
                "* Poor generalization",
                "**Action:**",
                f"* {'No action required' if total == 0 else 'Evaluate duplicates and clean data'}",
            )
        ]

        # Set up Gradient slide
        return {
            "deck": "object_detection_dataset_evaluation",
            "layout_name": "TextTableImages",
            "layout_arguments": {
                TextTableImages.ArgKeys.TITLE.value: title,
                TextTableImages.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                TextTableImages.ArgKeys.TEXT_COLUMN_BODY.value: content,
                TextTableImages.ArgKeys.DATA_COLUMN_TABLE.value: duplicates_df,
                TextTableImages.ArgKeys.IMAGE_WIDTH.value: 2.0,
                **near_dp_kwargs,
            },
        }

    def _generate_outliers_report(self, output_type: type) -> dict[str, Any]:
        """
        Generate the outliers report which consists of a single slide for the
        type of outlier analysis performed.  Outlier output types implemented
        are `DimensionStatsOutput`, `PixelStatsOutput`, and `VisualStatsOutput`.

        - Slide Format: `TextTableImages`
        - Text Content: Explanation of test, risks and action item
        - Table Content: Table of outliers identified for the analysis performed
        - Image Content: Sample images of identified outliers with labels
        """
        outliers: dict[int, dict[str, float]] = self.outputs["outliers"]
        stats: dict[str, Any] = self.outputs["stats"]
        source_index: list[tuple[int, int | None, int | None]] = stats["source_index"]
        total_out = len(outliers)
        len_ds = len(source_index)

        issues: dict[str, list[tuple[int, float]]] = {}
        for i, issue in outliers.items():
            for k in (s for s in output_type.__annotations__ if s in issue):
                issues.setdefault(k, []).append((int(i), issue[k]))
        issues = dict(take(8, sorted(issues.items(), key=lambda x: -len(x[1]))))

        issue_means: dict[str, float] = {}
        for k in (s for s in output_type.__annotations__ if s in issues):
            value = np.mean(stats[k])
            issue_means[k] = value
            issues[k] = sorted(issues[k], key=lambda x: -np.abs(x[1] - value))

        total_category = sum(len(v) for v in issues.values())
        category_name = output_type.__name__.replace("Output", "")

        outliers_df = pd.DataFrame(
            {
                self._to_title(k): [
                    *self._to_pct_2(len(v), len_ds),
                    f"{MU} = {issue_means[k]:.2f}",
                ]
                for k, v in issues.items()
            }
        )

        for i in range(max(2 - len(outliers_df), 0)):
            outliers_df[" " * i] = ["", ""]

        images: dict[str, list[Path]] = {}
        fig = plt.figure(figsize=(2, 2), dpi=100)
        ax = fig.add_axes(111)
        ax.set_aspect("equal")
        self.cache_contents_path.mkdir(exist_ok=True)
        for k, targets in issues.items():
            for si, value in (targets[_] for _ in range(min(len(targets), 3))):
                i, ti, _ = source_index[si]
                filepath = Path(self.cache_contents_path, f"{k}_{i}_{ti}.png")
                images.setdefault(k, []).append(filepath)
                if not filepath.exists():
                    data = self.dataset[i]
                    image = data[0]
                    label = as_numpy(data[1].labels)[ti or 0]
                    mapping = getattr(self.dataset, "index2label", None)
                    label = label if mapping is None else mapping[label]
                    title = f"{label}: {str(round(value, 2))}"
                    ax.clear()
                    ax.axis("off")
                    ax.set_title(title)
                    ax.imshow(np.moveaxis(as_numpy(image), 0, -1))
                    if ti is not None:
                        box = as_numpy(data[1].boxes)[ti]
                        rect = patches.Rectangle(
                            (box[0], box[1]),
                            box[2] - box[0],
                            box[3] - box[1],
                            linewidth=2,
                            edgecolor="r",
                            facecolor="none",
                        )
                        ax.add_patch(rect)
                    fig.savefig(str(filepath), pad_inches=0.0)
        plt.close(fig)

        image_kwargs = {
            IMAGE_ARGKEYS[i][j]: img_path
            for i, img_list in enumerate(images.values())
            for j, img_path in enumerate(img_list)
        }

        # Gradient slide kwargs
        title = f"Dataset: {self.dataset_id} | Category: Linting"
        heading = f"Outliers\n{category_name}"
        content = [
            Text(t, fontsize=14)
            for t in (
                "**Result:**",
                f"* Total Outliers: {self._to_pct_1(total_out, len_ds)}",
                f"* {category_name} Outliers: {self._to_pct_1(total_category, len_ds)}",
                "**Tests for:**",
                "* Uncleaned data",
                "**Risks:**",
                "* Leakage",
                "* Lack of robustness",
                "* Poor real-world performance",
                "* Poor generalization",
                "**Action:**",
                f"* {'No action required' if total_category == 0 else 'Evaluate outliers and clean data'}",
            )
        ]

        # Set up Gradient slide
        return {
            "deck": "object_detection_dataset_evaluation",
            "layout_name": "TextTableImages",
            "layout_arguments": {
                TextTableImages.ArgKeys.TITLE.value: title,
                TextTableImages.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                TextTableImages.ArgKeys.TEXT_COLUMN_BODY.value: content,
                TextTableImages.ArgKeys.DATA_COLUMN_TABLE.value: outliers_df,
                **image_kwargs,
            },
        }

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Generate reports for duplicates and outliers and include if not empty"""

        return [
            slide
            for slide in (
                self._generate_duplicates_report(),
                self._generate_outliers_report(DimensionStatsOutput),
                self._generate_outliers_report(PixelStatsOutput),
                self._generate_outliers_report(VisualStatsOutput),
            )
            if slide
        ]
