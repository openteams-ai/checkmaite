"""DataEval Bias Common Test Stage"""

from typing import Any, Literal

import numpy as np
import pandas as pd
import pydantic
import torch
from dataeval.data import Embeddings, Images, Metadata
from dataeval.metrics.bias import balance, coverage, diversity, parity
from gradient import SubText
from pydantic import Field

from jatic_ri._common.models import set_device
from jatic_ri._common.test_stages.impls._dataeval_utils import get_resnet18
from jatic_ri._common.test_stages.interfaces.plugins import SingleDatasetPlugin, TDataset
from jatic_ri._common.test_stages.interfaces.test_stage import (
    ConfigBase,
    OutputsBase,
    RunBase,
    TestStage,
)
from jatic_ri.util._types import Device, Image
from jatic_ri.util.slide_deck import create_section_by_item_slide, create_section_by_stacked_items_slide
from jatic_ri.util.utils import temp_image_file


class DataevalBiasConfig(ConfigBase):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    device: Device = Field(default_factory=lambda: set_device(None))
    metadata_to_exclude: list[str] = Field(
        default_factory=lambda: [], description="Dataset metadata to exclude from bias analysis"
    )
    num_neighbors: int = Field(
        default=5, description="Number of neighbors to consider when computing mutual information between factors"
    )
    diversity_method: Literal["simpson", "shannon"] = Field(
        default="simpson",
        description="The methodology used for defining diversity. The method specified "
        "defines diversity as the inverse Simpson diversity index linearly rescaled to the unit interval, "
        "or the normalized form of the Shannon entropy. diversity = 1 implies that samples are evenly "
        "distributed across a particular factor, diversity = 0 implies that all samples belong to one "
        "category/bin.",
    )
    radius_type: Literal["adaptive", "naive"] = Field(
        default="adaptive", description="The function used to determine radius for coverage."
    )
    percent: float = Field(
        default=0.01, description="Percent of observations to be considered uncovered. Only applies to adaptive radius."
    )


class DataevalBiasBalanceOutputs(OutputsBase):
    balance: np.ndarray
    factors: np.ndarray
    classwise: np.ndarray
    factor_names: list[str]
    class_names: list[str]
    image: Image


class DataevalBiasDiversityOutputs(OutputsBase):
    diversity_index: np.ndarray
    classwise: np.ndarray
    factor_names: list[str]
    class_names: list[str]
    image: Image


class DataevalBiasParityOutputs(OutputsBase):
    score: np.ndarray
    p_value: np.ndarray
    factor_names: list[str]


class DataevalBiasCoverageOutputs(OutputsBase):
    total: int
    uncovered_indices: np.ndarray
    critical_value_radii: np.ndarray
    coverage_radius: float
    image: Image | None = None


class DataevalBiasOutputs(pydantic.BaseModel):
    balance: DataevalBiasBalanceOutputs | None = None
    diversity: DataevalBiasDiversityOutputs | None = None
    parity: DataevalBiasParityOutputs | None = None
    coverage: DataevalBiasCoverageOutputs


class DataevalBiasRun(RunBase):
    config: DataevalBiasConfig
    outputs: DataevalBiasOutputs


class DatasetBiasTestStageBase(TestStage[DataevalBiasOutputs], SingleDatasetPlugin[TDataset]):
    """
    Measures four aspects of bias in a single dataset and programmatically generates a Gradient report
    with the measurements of bias, potential risks, and any actions required to reduce bias if found

    Bias is measured using four metrics: balance, coverage, diversity, parity.

    Balance, diversity, and parity calculate different aspects of correlation
    between metadata factors and class labels, while coverage is calculated using only the images
    """

    _RUN_TYPE = DataevalBiasRun

    device: torch.device = set_device(None)

    def __init__(
        self,
        metadata_to_exclude: list[str] | None = None,
        num_neighbors: int = 5,
        diversity_method: Literal["simpson", "shannon"] = "simpson",
        radius_type: Literal["adaptive", "naive"] = "adaptive",
        percent: float = 0.01,
    ) -> None:
        super().__init__()

        if metadata_to_exclude:
            self.metadata_to_exclude = metadata_to_exclude
        else:
            self.metadata_to_exclude = []

        self.num_neighbors = num_neighbors
        self.diversity_method: Literal["simpson", "shannon"] = diversity_method
        self.radius_type: Literal["adaptive", "naive"] = radius_type
        self.percent: float = percent

    def _create_config(self) -> ConfigBase:
        return DataevalBiasConfig(
            device=self.device,
            metadata_to_exclude=self.metadata_to_exclude,
            num_neighbors=self.num_neighbors,
            diversity_method=self.diversity_method,
            radius_type=self.radius_type,
            percent=self.percent,
        )

    def _run(self) -> DataevalBiasOutputs:
        """Run bias analysis using coverage and parity"""

        model, transform = get_resnet18()
        images = Images(self.dataset)
        embeddings = Embeddings(self.dataset, self._batch_size, transform, model, self.device).to_numpy()
        embeddings = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())
        metadata = Metadata(self.dataset, exclude=self.metadata_to_exclude)

        # metadata is not empty and hence valid to run balance, diversity, parity
        if metadata.factor_names:
            bal_out = balance(metadata, num_neighbors=self.num_neighbors)
            bal_dict = bal_out.data()
            bal_dict["image"] = bal_out.plot()

            div_out = diversity(metadata, method=self.diversity_method)
            div_dict = div_out.data()
            div_dict["image"] = div_out.plot()

            par_out = parity(metadata)
            par_dict = par_out.data()
        else:
            bal_dict = None
            div_dict = None
            par_dict = None

        cov_out = coverage(
            embeddings,
            num_observations=min(max(3, int(np.sqrt(len(images)))), 20),
            radius_type=self.radius_type,
            percent=self.percent,
        )
        cov_dict = cov_out.data()
        cov_dict["total"] = len(self.dataset)

        if len({image.shape for image in images}) == 1:
            cov_dict["image"] = cov_out.plot(images)

        return DataevalBiasOutputs.model_validate(
            {"balance": bal_dict, "diversity": div_dict, "parity": par_dict, "coverage": cov_dict}
        )

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect consumables"""
        if self._stored_run is None:
            raise RuntimeError("Can only collect consumables after run() was called")
        dataset_id = self._stored_run.dataset_ids[0]
        outputs: DataevalBiasOutputs = self._stored_run.outputs

        report_list = [
            self._report_coverage(outputs.coverage, dataset_id),
        ]
        if outputs.balance is not None:
            report_list.append(self._report_balance(outputs.balance, dataset_id))
        if outputs.diversity is not None:
            report_list.append(self._report_diversity(outputs.diversity, dataset_id))
        if outputs.parity is not None:
            report_list.append(self._report_parity(outputs.parity, dataset_id))
        return report_list

    def _report_coverage(self, coverage: DataevalBiasCoverageOutputs, dataset_id: str) -> dict[str, Any]:
        """Format coverage results for Gradient consumption"""
        # Gradient specific kwargs
        title = f"Dataset: {dataset_id} | Category Bias"
        heading = "Metric: Coverage"

        # Calculate results from coverage outputs
        uncovered_count = len(coverage.uncovered_indices)
        uncovered_percent = round(uncovered_count / coverage.total, 1)

        action_str = (
            "Increase representation of rare but relevant samples in areas of poor coverage"
            if uncovered_count > 0
            else "No action required"
        )
        text = [
            [SubText("Result:", bold=True)],
            f"Coverage: {uncovered_percent*100}% uncovered images in dataset",
            [SubText("Tests for:", bold=True)],
            "• Adequate sampling of data",
            [SubText("Risks:", bold=True)],
            "• Poor real-world performance",
            "• Lack of robustness",
            "• Poor generalization",
            [SubText("Actions:", bold=True)],
            f"• {action_str}",
        ]

        cov_df = pd.DataFrame(
            {
                "Poor Coverage": [f"{uncovered_count} of {coverage.total} ({uncovered_percent*100}%)"],
                "Threshold": [round(coverage.coverage_radius, 2)],
            },
        )

        # Gradient slide creation
        slide_kwargs = {"deck": self._deck, "title": title, "heading": heading, "text": text, "table": cov_df}
        # Image is only available if the input dataset had homogeneous-sized images
        if coverage.image is not None:
            return create_section_by_stacked_items_slide(
                **slide_kwargs,
                image_path=temp_image_file(coverage.image),
            )
        return create_section_by_item_slide(**slide_kwargs)

    def _report_balance(self, outputs: DataevalBiasBalanceOutputs, dataset_id: str) -> dict[str, Any]:
        """Format balance results for Gradient consumption"""
        title = f"Dataset: {dataset_id} | Category Bias"
        heading = "Metric: Balance"

        rollup = np.sum(outputs.factors[0, 1:] > 0.5)

        text = [
            [SubText("Result:", bold=True)],
            f"Balance: {rollup} factors co-occurring with class label",
            [SubText("Tests for:", bold=True)],
            "• Spurious correlations",
            [SubText("Risks:", bold=True)],
            "• Model trained with data are not equitable (not fair)",
            "• Models learn shortcuts",
            "• Poor real-world performance",
            "• Lack of robustness / poor generalization",
            [SubText("Action:", bold=True)],
            f"• {'Ensure balanced representation of all classes for all metadata' if rollup else 'No action required'}",
        ]

        return create_section_by_item_slide(
            deck=self._deck,
            title=title,
            heading=heading,
            text=text,
            image_path=temp_image_file(outputs.image),
        )

    def _report_diversity(self, outputs: DataevalBiasDiversityOutputs, dataset_id: str) -> dict[str, Any]:
        """Format diversity results for Gradient consumption"""
        title = f"Dataset: {dataset_id} | Category Bias"
        heading = "Metric: Diversity"

        rollup = np.sum(outputs.diversity_index < 0.5)

        text = [
            [SubText("Result:", bold=True)],
            f"Diversity: {rollup} factors with low diversity",
            [SubText("Tests for:", bold=True)],
            "• Evenness of distribution of factors",
            [SubText("Risks:", bold=True)],
            "• Model trained with data are not equitable (not fair)",
            "• Models learn shortcuts",
            "• Poor real-world performance",
            "• Lack of robustness / poor generalization",
            [SubText("Action:", bold=True)],
            f"• {'Ensure balanced representation of all classes for all metadata' if rollup else 'No action required'}",
        ]

        return create_section_by_item_slide(
            deck=self._deck,
            title=title,
            heading=heading,
            text=text,
            image_path=temp_image_file(outputs.image),
        )

    def _report_parity(self, outputs: DataevalBiasParityOutputs, dataset_id: str) -> dict[str, Any]:
        """Format parity results for Gradient consumption"""
        title = f"Dataset: {dataset_id} | Category Bias"
        heading = "Metric: Parity"

        metadata_pvalues = np.round(outputs.p_value, 3)
        rollup = np.sum(metadata_pvalues < 0.05)

        metadata_parity_table = pd.DataFrame(
            {
                "Metadata Factor": outputs.factor_names,
                "Chi-Square": np.round(outputs.score, 2),
                "P-value": metadata_pvalues,
            },
        )

        text = [
            [SubText("Result:", bold=True)],
            f"Parity: {rollup} factors correlated with labels",
            [SubText("Tests for:", bold=True)],
            " • Evenness of distribution of factors",
            [SubText("Risks:", bold=True)],
            " • Model trained with data are not equitable (not fair)",
            " • Models learn shortcuts",
            " • Poor real-world performance",
            " • Lack of robustness / poor generalization",
            [SubText("Action:", bold=True)],
            f"• {'Ensure balanced representation of all classes for all metadata' if rollup else 'No action required'}",
        ]

        return create_section_by_item_slide(
            deck=self._deck,
            title=title,
            heading=heading,
            text=text,
            table=metadata_parity_table,
        )
