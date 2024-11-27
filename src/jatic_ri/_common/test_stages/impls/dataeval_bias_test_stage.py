"""DataEval Bias Common Test Stage"""

import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from dataeval.metrics.bias import balance, coverage, diversity, parity
from numpy.typing import NDArray

from jatic_ri._common.test_stages.interfaces.plugins import SingleDatasetPlugin, TDataset
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.util.cache import JSONCache, NumpyEncoder
from jatic_ri.util.slide_deck import create_text_data_slide, create_text_table_data_slide


class DatasetBiasTestStageBase(TestStage[dict[str, Any]], SingleDatasetPlugin[TDataset]):
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
        return f"bias_{self._task}_{self.dataset_id}.json"

    @property
    def image_folder(self) -> Path:
        """Image folder path for storing temporary image examples"""

        # Remove .json from cache file, create cache_id folder, save images
        # Used by all report functions. Run ensures self.cache_path exists
        folder = Path(os.path.splitext(self.cache_path)[0])
        if not os.path.exists(folder):  # pragma: no cover
            os.mkdir(folder)
        return folder

    @abstractmethod
    def _get_images_labels_factors(self) -> tuple[list[NDArray[Any]], NDArray[np.int_], dict[str, NDArray[Any]]]:
        """Aggregate dataset into images, labels and metadata_factors"""

    def _run(self) -> dict[str, Any]:
        """Run bias analysis using coverage and parity"""

        images, labels, metadata = self._get_images_labels_factors()

        # Getting the output for balance and diversity to plot the results
        bal_out = balance(labels, metadata)
        div_out = diversity(labels, metadata)

        bal_dict = bal_out.dict()
        div_dict = div_out.dict()

        # Saving the figure in the results dict to pull out later
        bal_img_path = str(self.image_folder / "balance_heatmap.png")
        bal_fig = bal_out.plot()
        bal_fig.savefig(bal_img_path, format="png")
        div_img_path = str(self.image_folder / "diversity_heatmap.png")
        div_fig = div_out.plot()
        div_fig.savefig(div_img_path, format="png")

        bal_dict["image"] = bal_img_path
        div_dict["image"] = div_img_path

        result_dict = {
            self.BALANCE_KEY: bal_dict,
            self.DIVERSITY_KEY: div_dict,
            self.PARITY_KEY: parity(labels, metadata).dict(),
        }
        try:  # In the case where images are non-homogenous, skip running coverage
            images = np.array(images)
            cov_out = coverage(images, k=5)
            cov_dict = cov_out.dict()
            cov_img_path = str(self.image_folder / "coverage_plot.png")
            cov_fig = cov_out.plot(images)
            cov_fig.savefig(cov_img_path)
            cov_dict["image"] = cov_img_path
            result_dict.update({self.COVERAGE_KEY: cov_dict})
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
        """Format coverage results for Gradient consumption"""
        # Gradient specific kwargs
        title = f"Dataset: {self.dataset_id} | Category Bias"
        heading = "Metric: Coverage"

        # Calculate results from coverage outputs
        indices = outputs["indices"]
        total_length = len(self.dataset)
        uncovered_count = len(indices)
        uncovered_percent = round(uncovered_count / total_length, 1)

        action_str = (
            "Increase respresentation of rare but relevant samples in areas of poor coverage"
            if uncovered_count > 0
            else "No action required"
        )
        text = [
            "**Result:**",
            f"Coverage: {uncovered_percent*100}% uncovered images in dataset",
            "**Tests for:**",
            "* Adequate sampling of data",
            "**Risks**:",
            "* Poor real-world performance",
            "* Lack of robustness",
            "* Poor generalization",
            "**Actions:**",
            f"* {action_str}",
        ]

        cov_df = pd.DataFrame(
            {
                "Poor Coverage": [f"{uncovered_count} of {total_length} ({uncovered_percent*100}%)"],
                "Threshold": [round(outputs["critical_value"], 2)],
            },
        )
        image_path = Path(outputs["image"])

        # Gradient slide creation
        return create_text_table_data_slide(
            deck=self._deck,
            title=title,
            heading=heading,
            text=text,
            table=cov_df,
            image_path=image_path,
        )

    def _report_balance(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Format balance results for Gradient consumption"""
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

        image_path = Path(outputs["image"])

        return create_text_data_slide(
            deck=self._deck,
            title=title,
            heading=heading,
            text=text,
            image_path=image_path,
        )

    def _report_diversity(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Format diversity results for Gradient consumption"""
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

        image_path = Path(outputs["image"])

        return create_text_data_slide(
            deck=self._deck,
            title=title,
            heading=heading,
            text=text,
            image_path=image_path,
        )

    def _report_parity(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Format parity results for Gradient consumption"""
        title = f"Dataset: {self.dataset_id} | Category Bias"
        heading = "Metric: Parity"

        metadata_factors = np.array(outputs["metadata_names"])
        metadata_chisquares = np.round(np.array(outputs["score"]), 2)
        metadata_pvalues = np.round(np.array(outputs["p_value"]), 3)
        rollup = np.sum(metadata_pvalues < 0.05)

        metadata_parity_table = pd.DataFrame(
            {
                "Metadata Factor": metadata_factors,
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

        return create_text_data_slide(
            deck=self._deck,
            title=title,
            heading=heading,
            text=text,
            table=metadata_parity_table,
        )
