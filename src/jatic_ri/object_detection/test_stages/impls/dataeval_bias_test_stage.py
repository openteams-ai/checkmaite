import os  # noqa: D100
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from dataeval.metrics.bias import balance, coverage, diversity, parity
from dataeval.utils.torch import read_dataset
from gradient.slide_deck.shapes import Text
from gradient.templates_and_layouts.generic_layouts.text_data import TextData
from gradient.templates_and_layouts.generic_layouts.text_table_images import TextTableImages
from maite.protocols import ArrayLike
from PIL import Image

from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import SingleDatasetPlugin
from jatic_ri.util.cache import JSONCache, NumpyEncoder


def create_text_data_slide(
    title: str,
    heading: str,
    text: list[str],
    table: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Fills in TextData template with values

    Note: Image path support to be added in the future
    """

    content = [Text(t, fontsize=16) for t in text]

    template = {
        "deck": "object_detection_dataset_evaluation",
        "layout_name": "TextData",
        "layout_arguments": {
            TextData.ArgKeys.TITLE.value: title,
            # Text arguments
            TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
            TextData.ArgKeys.TEXT_COLUMN_HALF.value: True,
            TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
        },
    }

    if table is not None:
        template["layout_arguments"].update({TextData.ArgKeys.DATA_COLUMN_TABLE.value: table})

    return template


class DatasetBiasTestStage(TestStage[dict[str, Any]], SingleDatasetPlugin):
    """
    Measures four aspects of bias in a single dataset and programmatically generates a Gradient report
    with the measurements of bias, potential risks, and any actions required to reduce bias if found

    Bias is measured using four metrics: balance, coverage, diversity, parity.

    Balance, diversity, and parity calculate different aspects of correlation
    between metadata factors and class labels, while coverage is calculated using only the images
    """

    cache: Optional[Cache[dict[str, Any]]] = JSONCache(encoder=NumpyEncoder)
    COVERAGE_KEY = "coverage"
    BALANCE_KEY = "balance"
    DIVERSITY_KEY = "diversity"
    PARITY_KEY = "parity"

    @property
    def cache_id(self) -> str:
        """Bias Test Stage cache identifier"""
        return f"bias_{self.dataset_id}.json"

    @property
    def image_folder(self) -> Path:
        """Image folder path for storing temporary image examples"""

        # Remove .json from cache file, create cache_id folder, save images
        # Used by all report functions. Run ensures self.cache_path exists
        folder = Path(os.path.splitext(self.cache_path)[0])
        if not os.path.exists(folder):  # pragma: no cover
            os.mkdir(folder)
        return folder

    def save_image(self, image: ArrayLike, path: str) -> Path:
        """Saves an image at image_folder / path as RGB image"""

        full_path = self.image_folder / Path(path)
        image_pil = Image.fromarray(image, mode="RGB")  # type: ignore
        image_pil.save(full_path)

        return full_path

    def _run(self) -> dict[str, Any]:
        """Run bias analysis using coverage and parity"""

        # Separate data into individual lists
        images, targets, metadata = read_dataset(self.dataset)  # type: ignore

        # Aggregate each metadata factor into lists:
        # dict[factor] = List[factor_values]
        factor_lists = defaultdict(list)
        labels = []

        # Flattens all targets into one array and
        # copies metadata for an image into all of its targets
        for target, mdata in zip(targets, metadata):
            # Generates flat list of all labels
            tlabels = np.array(target.labels)
            labels.extend(tlabels.tolist())

            # Aggregates list for each metadata factor
            for k, v in mdata.items():
                if not isinstance(v, Sequence):
                    v = [v] * len(tlabels)
                factor_lists[k].extend(v)

        # Convert all lists into ArrayLike
        metadata_arrs: dict[str, ArrayLike] = {k: np.array(v) for k, v in factor_lists.items()}

        result_dict = {
            self.BALANCE_KEY: balance(labels, metadata_arrs).dict(),
            self.DIVERSITY_KEY: diversity(labels, metadata_arrs).dict(),
            self.PARITY_KEY: parity(labels, data_factors=metadata_arrs).dict(),
        }
        try:  # In the case where images are non-homogenous, skip running coverage
            images = np.array(images)
            result_dict.update({self.COVERAGE_KEY: coverage(images, k=5).dict()})
        except ValueError:
            pass

        return result_dict

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect consumables"""

        method_map = {
            self.BALANCE_KEY: self._report_balance,
            self.COVERAGE_KEY: self._report_coverage,
            self.DIVERSITY_KEY: self._report_diversity,
            self.PARITY_KEY: self._report_parity,
        }

        # Iterate over all bias methods and append to slide list
        return [
            report_func(self.outputs[method_key])
            for method_key, report_func in method_map.items()
            if method_key in self.outputs
        ]

    def _report_coverage(self, outputs: dict[str, Any]) -> dict[str, Any]:
        # Gradient specific kwargs
        title = f"Dataset: {self.dataset_id} | Category Bias"
        heading = "Metric: Coverage"
        image_caption = "Examples"

        # Calculate results from coverage outputs
        indices = outputs["indices"]
        total_length = len(self.dataset)
        uncovered_count = len(indices)
        uncovered_percent = round(uncovered_count / total_length, 3)

        text = [
            "**Result:**",
            f"{uncovered_percent*100}% uncovered images in dataset",
            "**Tests for:**",
            "* Adequate sampling of data",
            "**Risks**:",
            "* Poor real-world performance",
            "* Lack of robustness",
            "* Poor generalization",
            "**Actions:**",
            "* Increase respresentation of rare but relevant samples in areas of poor coverage",
        ]

        content = [Text(t, fontsize=16) for t in text]

        cov_df = pd.DataFrame(
            {
                "Poor Coverage": [f"{uncovered_count} of {total_length} ({uncovered_percent*100}%)"],
                "Threshold": [round(outputs["critical_value"], 3)],
            },
        )

        # Collect uncovered examples
        top_k = 3
        highest_uncovered_indices = np.argsort(indices)[:top_k]
        image_paths = []

        # Save top_k images to cache for gradient deck to reference
        for i, index in enumerate(highest_uncovered_indices):
            path = self.save_image(np.array(self.dataset[index][0]), path=f"coverage_example_{i}.png")
            image_paths.append(path)

        # Create gradient template argument dictionary
        layout_args = {
            TextTableImages.ArgKeys.TITLE.value: title,
            # Text arguments
            TextTableImages.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
            TextTableImages.ArgKeys.TEXT_COLUMN_BODY.value: content,
            TextTableImages.ArgKeys.TEXT_COLUMN_HALF.value: True,
            TextTableImages.ArgKeys.IMAGE_CAPTION.value: image_caption,
            # DataFrame arguments
            TextTableImages.ArgKeys.DATA_COLUMN_TABLE.value: cov_df,
        }

        # Selectively assign argkeys based on the number of image paths at a specific column
        # Can be removed when template is updated to TextData
        column_args = (
            str(TextTableImages.ArgKeys.DATA_COLUMN_2_IMAGE_1.value),
            str(TextTableImages.ArgKeys.DATA_COLUMN_2_IMAGE_2.value),
            str(TextTableImages.ArgKeys.DATA_COLUMN_2_IMAGE_3.value),
        )

        image_path_args = dict(zip(column_args, image_paths))
        layout_args.update(image_path_args)

        # Gradient slide creation
        return {
            "deck": "object_detection_dataset_evaluation",
            "layout_name": "TextTableImages",
            "layout_arguments": layout_args,
        }

    def _report_balance(self, outputs: dict[str, Any]) -> dict[str, Any]:
        title = f"Dataset: {self.dataset_id} | Category Bias"
        heading = "Metric: Balance"

        factors = outputs["factors"]

        rollup = np.sum(np.array(factors)[0, 1:] > 0.5)

        text = [
            "**Result:**",
            f"Balance: {rollup} factors co-occuring with class label",
            "**Tests for:**",
            "* Spurious correlations",
            "**Risks:**",
            "* Model trained with data are not equitable (not fair)",
            "* Models learn shortcuts",
            "* Poor real-world performance",
            "* Lack of robustness / poor generalization",
            "**Action:**",
            f"* {'Ensure balanced representation of all classes for all metadata' if rollup else 'No action required'}",
        ]

        return create_text_data_slide(
            title=title,
            heading=heading,
            text=text,
        )

    def _report_diversity(self, outputs: dict[str, Any]) -> dict[str, Any]:
        title = f"Dataset: {self.dataset_id} | Category Bias"
        heading = "Metric: Diversity"

        rollup = np.sum(np.array(outputs["diversity_index"]) < 0.5)

        text = [
            "**Result:**",
            f"Diversity: {rollup} factors with low diversity",
            "**Tests for:**",
            "* Evenness of distribution of factors",
            "**Risks:**",
            "* Model trained with data are not equitable (not fair)",
            "* Models learn shortcuts",
            "* Poor real-world performance",
            "* Lack of robustness / poor generalization",
            "**Action:**",
            f"* {'Ensure balanced representation of all classes for all metadata' if rollup else 'No action required'}",
        ]

        return create_text_data_slide(
            title,
            heading,
            text,
        )

    def _report_parity(self, outputs: dict[str, Any]) -> dict[str, Any]:
        title = f"Dataset: {self.dataset_id} | Category Bias"
        heading = "Metric: Parity"

        # metadata_factors = np.array(outputs["score"])
        metadata_chisquares = np.round(np.array(outputs["score"]), 3)
        metadata_pvalues = np.array(outputs["p_value"])
        rollup = np.sum(metadata_pvalues < 0.05)

        metadata_parity_table = pd.DataFrame(
            {
                # "Metadata Factor": metadata_factors, # Currently does not return factor list
                "Chi-Square": metadata_chisquares,
                "P-value": metadata_pvalues,
            },
        )

        text = [
            "**Result:**",
            f"Parity: {rollup} factors correlated with labels",
            "**Tests for:**",
            " * Evenness of distribution of factors",
            "**Risks:**",
            " * Model trained with data are not equitable (not fair)",
            " * Models learn shortcuts",
            " * Poor real-world performance",
            " * Lack of robustness / poor generalization",
            "**Action:**",
            f"* {'Ensure balanced representation of all classes for all metadata' if rollup else 'No action required'}",
        ]

        return create_text_data_slide(title=title, heading=heading, text=text, table=metadata_parity_table)
