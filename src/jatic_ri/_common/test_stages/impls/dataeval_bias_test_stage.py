"""DataEval Bias Common Test Stage.

Measures four aspects of bias in a single dataset: balance, coverage,
diversity, and parity. Programmatically generates a Gradient report with bias
measurements, potential risks, and actions to reduce bias if found.
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pydantic
import torch
from dataeval.data import Embeddings, Images, Metadata
from dataeval.metrics.bias import balance, coverage, diversity
from gradient import SubText
from gradient.slide_deck.shapes import Text
from gradient.slide_deck.shapes.image_shapes import GradientImage
from gradient.templates_and_layouts.generic_layouts import SectionByItem
from pydantic import Field

from jatic_ri import PACKAGE_DIR, cache_path
from jatic_ri._common.models import set_device
from jatic_ri._common.test_stages.impls._dataeval_utils import get_resnet18, plot_blank_or_single_image
from jatic_ri._common.test_stages.interfaces.plugins import SingleDatasetPlugin, TDataset
from jatic_ri._common.test_stages.interfaces.test_stage import (
    ConfigBase,
    OutputsBase,
    RunBase,
    TestStage,
)
from jatic_ri.util._types import Device, Image
from jatic_ri.util.slide_deck import (
    create_section_by_item_slide,
    create_table_text_slide,
    create_two_item_text_slide,
)
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
    image_classwise: Image
    image_metadata: Image


class DataevalBiasDiversityOutputs(OutputsBase):
    diversity_index: np.ndarray
    classwise: np.ndarray
    factor_names: list[str]
    class_names: list[str]
    image: Image


class DataevalBiasCoverageOutputs(OutputsBase):
    total: int
    uncovered_indices: np.ndarray
    critical_value_radii: np.ndarray
    coverage_radius: float
    image: Image | None = None


class DataevalBiasOutputs(pydantic.BaseModel):
    balance: DataevalBiasBalanceOutputs | None = None
    diversity: DataevalBiasDiversityOutputs | None = None
    coverage: DataevalBiasCoverageOutputs


class DataevalBiasRun(RunBase):
    config: DataevalBiasConfig
    outputs: DataevalBiasOutputs


class DatasetBiasTestStageBase(TestStage[DataevalBiasOutputs], SingleDatasetPlugin[TDataset]):
    """Measures bias in a single dataset.

    Generates a Gradient report with bias measurements, potential risks, and
    actions to reduce bias. Bias is measured using four metrics: balance,
    coverage, diversity, and parity. Balance, diversity, and parity assess
    correlations between metadata factors and class labels. Coverage is
    calculated using only the images.

    Parameters
    ----------
    metadata_to_exclude : list[str], optional
        Dataset metadata to exclude from bias analysis. Defaults to None.
    num_neighbors : int, optional
        Number of neighbors for mutual information in balance calculation.
        Defaults to 5.
    diversity_method : Literal["simpson", "shannon"], optional
        Methodology for defining diversity. Defaults to "simpson".
    radius_type : Literal["adaptive", "naive"], optional
        Function to determine radius for coverage. Defaults to "adaptive".
    percent : float, optional
        Percent of observations considered uncovered (adaptive radius only).
        Defaults to 0.01.

    Attributes
    ----------
    _RUN_TYPE : type[DataevalBiasRun]
        The type of the run object associated with this test stage.
    device : torch.device
        The device to use for computations.
    metadata_to_exclude : list[str]
        Dataset metadata to exclude from bias analysis.
    num_neighbors : int
        Number of neighbors for mutual information in balance calculation.
    diversity_method : Literal["simpson", "shannon"]
        Methodology for defining diversity.
    radius_type : Literal["adaptive", "naive"]
        Function to determine radius for coverage.
    percent : float
        Percent of observations considered uncovered (adaptive radius only).
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
        """Run bias analysis.

        Performs bias analysis using coverage, and optionally balance, diversity,
        and parity if metadata is available.

        Returns
        -------
        DataevalBiasOutputs
            The outputs of the bias analysis.
        """
        model, transform = get_resnet18()
        images = Images(self.dataset)

        embeddings = Embeddings(self.dataset, self._batch_size, transform, model, device=self.device).to_numpy()
        embeddings = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())
        # coverage tool expects sequence of vectors
        if len(embeddings.shape) == 1:
            embeddings = np.array([embeddings])

        metadata = Metadata(self.dataset, exclude=self.metadata_to_exclude)

        # metadata is not empty and hence valid to run balance, diversity, parity
        if metadata.factor_names:
            bal_out = balance(metadata, num_neighbors=self.num_neighbors)
            bal_dict = bal_out.data()
            bal_dict["image_metadata"] = bal_out.plot(plot_classwise=False)
            if len(np.unique(metadata.class_labels)) != len(metadata.class_names):
                bal_dict["image_classwise"] = bal_out.plot(
                    plot_classwise=True, row_labels=np.unique(metadata.class_labels)
                )
            else:
                bal_dict["image_classwise"] = bal_out.plot(plot_classwise=True)

            div_out = diversity(metadata, method=self.diversity_method)
            div_dict = div_out.data()
            div_dict["image"] = div_out.plot()

        else:
            bal_dict = None
            div_dict = None

        num_observations = min(max(3, int(np.sqrt(len(images)))), 20)

        if num_observations >= len(embeddings):
            raise ValueError(
                f"Need at least (num_observations + 1) points to compute k-NN coverage, "
                f"got N={len(embeddings)} points, requested num_observations={num_observations}. "
                "Please provide more images."
            )

        cov_out = coverage(
            embeddings,
            num_observations=num_observations,
            radius_type=self.radius_type,
            percent=self.percent,
        )
        cov_dict = cov_out.data()
        cov_dict["total"] = len(self.dataset)

        if len({image.shape for image in images}) == 1:
            cov_dict["image"] = cov_out.plot(images)

        return DataevalBiasOutputs.model_validate({"balance": bal_dict, "diversity": div_dict, "coverage": cov_dict})

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect consumables for the report.

        Gathers the results from the bias analysis run and formats them
        for inclusion in a Gradient report.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, each representing a slide or section
            in the Gradient report.

        Raises
        ------
        RuntimeError
            If `run()` has not been called before collecting consumables.
        """
        if self._stored_run is None:
            raise RuntimeError("Can only collect consumables after run() was called")
        outputs: DataevalBiasOutputs = self._stored_run.outputs

        report_list = [self._generate_table_of_contents()]

        report_list.append(self._report_coverage(outputs.coverage))

        if outputs.balance is not None:
            report_list.append(self._report_balance_metadata_factors(outputs.balance))
            report_list.append(self._report_balance_classwise(outputs.balance))

        if outputs.diversity is not None:
            report_list.append(self._report_diversity(outputs.diversity))

        # TODO: reactivate this when we have a next steps section from Team Aria
        # report_list.append(self._report_next_steps())

        return report_list

    def _generate_table_of_contents(self) -> dict[str, Any]:
        """Generates a table of contents for the report.

        Returns
        -------
        dict[str, Any]
            A dictionary representing the table of contents slide.
        """
        right_item = [
            "\n",
            "* Coverage Analysis",
            "* Balance Analysis",
            "* Diversity Analysis",
            "* Next Steps",
        ]

        left_item = GradientImage(
            src=Path(PACKAGE_DIR.joinpath("_sample_imgs/toc.png")), width=100, height=100, top=0.5, left=0.5
        )
        return create_two_item_text_slide(
            deck=self._deck, title="Bias Table of Contents", left_item=left_item, right_item=right_item
        )

    def _report_coverage(self, coverage: DataevalBiasCoverageOutputs) -> dict[str, Any]:
        """Format coverage results for Gradient consumption.

        Parameters
        ----------
        coverage : DataevalBiasCoverageOutputs
            The coverage analysis outputs.

        Returns
        -------
        dict[str, Any]
            A dictionary representing the coverage analysis slide.
        """
        title = "Coverage Analysis"

        uncovered_count = len(coverage.uncovered_indices)
        uncovered_percent = round(uncovered_count / coverage.total, 2)

        cov_df = pd.DataFrame(
            {
                "Potentially under-represented images": [
                    f"{uncovered_count} of {coverage.total} ({uncovered_percent*100}%)"
                ]
            },
        )

        content = Text(
            [
                SubText("Description: ", bold=True),
                SubText(
                    "Coverage uses AI to identify potentially under-represented images "
                    "that warrant further investigation. Under-represented images are "
                    "those which are closely-related to, at most, a small amount of other "
                    "images in the dataset.\n"
                ),
            ],
            fontsize=22,
        )

        return create_table_text_slide(deck=self._deck, title=title, text=content, data=cov_df)

    def _report_balance_metadata_factors(self, outputs: DataevalBiasBalanceOutputs) -> dict[str, Any]:
        """Format balance results (metadata factors) for Gradient consumption.

        Parameters
        ----------
        outputs : DataevalBiasBalanceOutputs
            The balance analysis outputs.

        Returns
        -------
        dict[str, Any]
            A dictionary representing the balance analysis slide for metadata factors.
        """
        title = "Balance Analysis 1"
        heading = "   "

        text = [
            [
                SubText("Description: ", bold=True, fontsize=20),
                SubText(
                    "Balance can help uncover potential model bias by identifying "
                    "spurious correlations between metadata and class labels. For "
                    "example, a model might incorrectly learn to associate vehicles"
                    "with the metadata ‘occlusions’ if training images always show "
                    "vehicles partially hidden by other objects. This learned behaviour "
                    "might then fail if a vehicle was to appear fully visible.",
                    fontsize=20,
                ),
            ],
            [
                SubText(
                    "Values approaching or exceeding 0.5 in the heat map should be "
                    "further investigated to prevent a model from potentially learning "
                    "a harmful shortcut.",
                    fontsize=20,
                )
            ],
        ]

        return create_section_by_item_slide(
            deck=self._deck,
            title=title,
            heading=heading,
            text=text,
            image_path=temp_image_file(outputs.image_metadata),
        )

    def _report_balance_classwise(self, outputs: DataevalBiasBalanceOutputs) -> dict[str, Any]:
        """Format balance results (classwise) for Gradient consumption.

        Parameters
        ----------
        outputs : DataevalBiasBalanceOutputs
            The balance analysis outputs.

        Returns
        -------
        dict[str, Any]
            A dictionary representing the balance analysis slide for classwise balance.
        """
        title = "Balance Analysis 2"
        heading = "   "

        text = [
            [
                SubText("Description: ", bold=True, fontsize=20),
                SubText(
                    "Balance can also help uncover potential model bias by identifying "
                    "relative class imbalance. Correlations between an individual class "
                    "and all other class labels indicate that a specific class is "
                    "over-represented compared to other classes. This can become a problem "
                    "if operational data does not also have this imbalance.",
                    fontsize=20,
                ),
            ],
            [
                SubText(
                    "Values approaching or exceeding 0.5 in the heat map should be further " "investigated.",
                    fontsize=20,
                )
            ],
        ]

        return create_section_by_item_slide(
            deck=self._deck,
            title=title,
            heading=heading,
            text=text,
            image_path=temp_image_file(outputs.image_classwise),
        )

    def _report_diversity(self, outputs: DataevalBiasDiversityOutputs) -> dict[str, Any]:
        """Format diversity results for Gradient consumption.

        Parameters
        ----------
        outputs : DataevalBiasDiversityOutputs
            The diversity analysis outputs.

        Returns
        -------
        dict[str, Any]
            A dictionary representing the diversity analysis slide.
        """
        title = "Diversity Analysis"
        heading = "   "

        text = [
            [
                SubText("Description: ", bold=True, fontsize=20),
                SubText(
                    "Diversity measures how well each metadata factor is sampled over its range of "
                    "possible values. Values near 1 indicate wide sampling, while values near 0 "
                    "indicate imbalanced sampling e.g. all datapoints taking a single value.",
                    fontsize=20,
                ),
            ],
            [
                SubText(
                    "The categories of most interest are those with values that are between 0.1 "
                    "and 0.4. The data for each metadata factor in these ranges should be inspected "
                    " to see if the sampled values are appropriate for operational data.",
                    fontsize=20,
                )
            ],
            [
                SubText(
                    "Values below 0.1 are generally so heavily imbalanced that a genuine problem "
                    "should be immediately obvious.",
                    fontsize=20,
                )
            ],
        ]

        return create_section_by_item_slide(
            deck=self._deck,
            title=title,
            heading=heading,
            text=text,
            image_path=temp_image_file(outputs.image),
        )

    def _report_next_steps(self) -> dict[str, Any]:
        """Generate a report for the next steps.

        This outlines how to investigate issues that may arise during analysis.

        Returns
        -------
        dict[str, Any]
            A dictionary representing the next steps slide.
        """
        dir_ = Path(cache_path() / "bias-test-stage-artifacts")
        dir_.mkdir(parents=True, exist_ok=True)
        filepath = dir_ / "blank_img.png"
        plot_blank_or_single_image(filepath)

        title = "Bias Analysis"
        heading = "Next Steps\n"
        content = [
            Text(t, fontsize=14)
            for t in (
                "Below are the recommended next steps to investigating issues that may arise during analysis.",
                [SubText("In general:", bold=True)],
                "1. Insert text here",
                "2. Insert text here",
                "3. Insert text here",
            )
        ]

        return {
            "deck": self._deck,
            "layout_name": "SectionByItem",
            "layout_arguments": {
                SectionByItem.ArgKeys.TITLE.value: title,
                SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
                SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
                SectionByItem.ArgKeys.LINE_SECTION_HALF.value: True,
                SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: filepath,
            },
        }
