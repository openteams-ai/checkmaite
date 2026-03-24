import logging
from pathlib import Path
from typing import Any, Literal, cast

import dataeval_plots as dep
import numpy as np
import pandas as pd
import polars as pl
import pydantic
from dataeval import Embeddings, Metadata
from dataeval.bias import Balance, Diversity
from dataeval.core import coverage_adaptive, coverage_naive
from dataeval.extractors import TorchExtractor
from pydantic import Field

from checkmaite import cache_path
from checkmaite.core._common.feature_extractor import load_feature_extractor, pca_projector, to_unit_interval_01
from checkmaite.core._types import Device, Image, ModelSpec, TorchvisionModelSpec
from checkmaite.core._utils import deprecated, requires_optional_dependency, set_device
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
from checkmaite.core.report._plotting_utils import plot_blank_or_single_image, temp_image_file

logger = logging.getLogger(__name__)


class DataevalBiasConfig(CapabilityConfigBase):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    batch_size: int = Field(default=32, description="Batch size to use when encoding images.")
    device: Device = Field(default_factory=lambda: set_device(None))

    metadata_to_exclude: list[str] = Field(
        default_factory=lambda: ["id", "file_name"], description="Dataset metadata to exclude from bias analysis"
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


class DataevalBiasBalanceOutputs(CapabilityOutputsBase):
    balance: pl.DataFrame
    factors: pl.DataFrame
    classwise: pl.DataFrame
    image_classwise: Image
    image_metadata: Image | None = None


class DataevalBiasDiversityOutputs(CapabilityOutputsBase):
    factors: pl.DataFrame
    classwise: pl.DataFrame
    image: Image


class DataevalBiasCoverageOutputs(CapabilityOutputsBase):
    total: int
    uncovered_indices: np.ndarray
    critical_value_radii: np.ndarray
    coverage_radius: float
    image: Image | None = None


class DataevalBiasOutputs(pydantic.BaseModel):
    balance: DataevalBiasBalanceOutputs | None = None
    diversity: DataevalBiasDiversityOutputs | None = None
    coverage: DataevalBiasCoverageOutputs


class DataevalBiasRecord(BaseRecord, table_name="dataeval_bias"):
    """Record for DataevalBias capability results.

    Stores dataset bias summary metrics including coverage, balance,
    and diversity. Balance and diversity fields are None when the
    dataset has no usable metadata factors.
    """

    dataset_id: str

    # Coverage (always present) — identifies under-represented images.
    coverage_total: int
    coverage_uncovered_count: int
    coverage_uncovered_ratio: float
    coverage_radius: float

    # Balance (None when no metadata factors) — mutual information between metadata and class labels.
    balance_num_factors: int | None = None
    balance_mean: float | None = None
    balance_max: float | None = None
    balance_factors_above_05: int | None = None

    # Diversity (None when no metadata factors) — how well each factor is sampled.
    diversity_num_factors: int | None = None
    diversity_mean: float | None = None
    diversity_min: float | None = None
    diversity_factors_below_04: int | None = None


class DataevalBiasRun(CapabilityRunBase[DataevalBiasConfig, DataevalBiasOutputs]):
    config: DataevalBiasConfig
    outputs: DataevalBiasOutputs

    def extract(self) -> list[DataevalBiasRecord]:
        """Extract summary metrics from this DataevalBias run.

        Returns a single record per dataset with aggregate coverage,
        balance, and diversity scalars.  Balance and diversity fields
        are None when the dataset had no usable metadata factors.
        """
        outputs = self.outputs
        coverage = outputs.coverage

        uncovered_count = len(coverage.uncovered_indices)

        # Balance aggregates (optional)
        balance_num_factors: int | None = None
        balance_mean: float | None = None
        balance_max: float | None = None
        balance_factors_above_05: int | None = None

        if outputs.balance is not None:
            bal = outputs.balance.balance["Balance Score"].to_numpy()
            balance_num_factors = len(bal)
            balance_mean = float(np.mean(bal))
            balance_max = float(np.max(bal))
            balance_factors_above_05 = int(np.sum(bal >= 0.5))

        # Diversity aggregates (optional)
        diversity_num_factors: int | None = None
        diversity_mean: float | None = None
        diversity_min: float | None = None
        diversity_factors_below_04: int | None = None

        if outputs.diversity is not None:
            div = outputs.diversity.factors["Diversity Index"].to_numpy()
            diversity_num_factors = len(div)
            diversity_mean = float(np.mean(div))
            diversity_min = float(np.min(div))
            diversity_factors_below_04 = int(np.sum(div < 0.4))

        return [
            DataevalBiasRecord(
                run_uid=self.run_uid,
                dataset_id=self.dataset_metadata[0]["id"],
                coverage_total=coverage.total,
                coverage_uncovered_count=uncovered_count,
                coverage_uncovered_ratio=uncovered_count / coverage.total if coverage.total > 0 else 0.0,
                coverage_radius=coverage.coverage_radius,
                balance_num_factors=balance_num_factors,
                balance_mean=balance_mean,
                balance_max=balance_max,
                balance_factors_above_05=balance_factors_above_05,
                diversity_num_factors=diversity_num_factors,
                diversity_mean=diversity_mean,
                diversity_min=diversity_min,
                diversity_factors_below_04=diversity_factors_below_04,
            )
        ]

    # The order is important
    @requires_optional_dependency("gradient", install_hint="pip install '.[unsupported]'")
    @deprecated(replacement="collect_md_report")
    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:  # noqa: ARG002  # pragma: no cover
        """Collect consumables for the report.

        Gathers the results from the bias analysis run and formats them
        for inclusion in a Gradient report.

        Parameters
        ----------
        threshold : float
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, each representing a slide or section
            in the Gradient report.
        """

        outputs: DataevalBiasOutputs = self.outputs

        deck = self.capability_id

        report_list = [generate_table_of_contents(deck)]

        report_list.append(report_coverage(deck, outputs.coverage))

        if outputs.balance is not None:
            report_list.append(report_balance_metadata_factors(deck, outputs.balance))
            report_list.append(report_balance_classwise(deck, outputs.balance))

        if outputs.diversity is not None:
            report_list.append(report_diversity(deck, outputs.diversity))

        # TODO: reactivate this when we have a next steps section from Team Aria
        # report_list.append(report_next_steps(deck))

        return report_list

    def collect_md_report(self, threshold: float) -> str:  # noqa: ARG002
        """Collect Markdown-formatted report content.

        Gathers the results from the bias analysis run and formats them
        as Markdown text, without dependencies on Gradient.

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

        outputs: DataevalBiasOutputs = self.outputs

        md = MarkdownOutput(title="Bias Analysis Report")

        generate_table_of_contents_md(md)

        md.add_section_divider()

        report_coverage_md(md, outputs.coverage)

        if outputs.balance is not None:
            md.add_section_divider()
            report_balance_metadata_factors_md(md, outputs.balance)
            report_balance_classwise_md(md, outputs.balance)

        if outputs.diversity is not None:
            md.add_section_divider()
            report_diversity_md(md, outputs.diversity)

        return md.render()


class DataevalBiasBase(Capability[DataevalBiasOutputs, TDataset, TModel, TMetric, DataevalBiasConfig]):
    """Measures bias in a single dataset.

    Generates a Gradient report with bias measurements, potential risks, and
    actions to reduce bias. Bias is measured using four metrics: balance,
    coverage, diversity, and parity. Balance, diversity, and parity assess
    correlations between metadata factors and class labels. Coverage is
    calculated using only the images.

    Attributes
    ----------
    _RUN_TYPE
        The type of the run object associated with this capability.
    """

    _RUN_TYPE = DataevalBiasRun

    @classmethod
    def _create_config(cls) -> DataevalBiasConfig:
        return DataevalBiasConfig()

    @property
    def supports_datasets(self) -> Number:
        """Number of datasets this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating dataset support.
        """
        return Number.ONE

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

    def _run(  # noqa: C901
        self,
        models: list[TModel],  # noqa: ARG002
        datasets: list[TDataset],
        metrics: list[TMetric],  # noqa: ARG002
        config: DataevalBiasConfig,
        use_prediction_and_evaluation_cache: bool,  # noqa: ARG002
    ) -> DataevalBiasOutputs:
        """Performs bias analysis using coverage, and optionally balance, diversity,
        and parity if metadata is available.

        Returns
        -------
            The outputs of the bias analysis.
        """
        dataset = datasets[0]
        metadata = Metadata(dataset, exclude=config.metadata_to_exclude)
        class_names = list(metadata.index2label.values())

        # metadata is not empty and hence valid to run balance, diversity, parity
        if metadata.factor_names:
            bal_out = Balance(num_neighbors=config.num_neighbors).evaluate(metadata)
            bal_dict = bal_out.data()
            if len(bal_dict["factors"]) > 0:
                bal_dict["image_metadata"] = dep.plot(bal_out, plot_classwise=False)
            if len(np.unique(metadata.class_labels)) != len(class_names):
                bal_dict["image_classwise"] = dep.plot(
                    bal_out, plot_classwise=True, row_labels=np.unique(metadata.class_labels)
                )
            else:
                bal_dict["image_classwise"] = dep.plot(bal_out, plot_classwise=True)
            # rename columns in balance dataframe. It should done after creating images
            # otherwise required column names wont be found
            bal_dict["balance"].columns = ["Factor", "Balance Score"]

            div_out = Diversity(method=config.diversity_method).evaluate(metadata)
            div_dict = div_out.data()
            div_dict["image"] = dep.plot(div_out)
            # rename columns in diversity factors dataframe:
            div_dict["factors"].columns = ["Factor", "Diversity Index", "Col3"]
            div_dict["factors"] = div_dict["factors"][["Factor", "Diversity Index"]]
        else:
            bal_dict = None
            div_dict = None

        # We need a stable, user-tunable numeric representation of images so that distance-based checks
        # (like coverage) behave sensibly. We therefore (1) load a feature extractor with matching
        # preprocessing, (2) optionally shrink them to the user’s requested size with PCA (only when
        # shrinking is possible), and (3) rescale to [0,1] because the coverage algorithm assumes features
        # live on that range.
        fe = load_feature_extractor(device=config.device, model_spec=config.feature_extractor_spec)
        extractor = TorchExtractor(fe.model, transforms=fe.transforms, device=config.device)
        embeddings = Embeddings(
            dataset,
            extractor=extractor,
            batch_size=config.batch_size,
        )[:]

        n, d = embeddings.shape

        if config.target_embedding_dim < d:
            k_max = min(n, d)
            k = min(config.target_embedding_dim, k_max)

            if k != config.target_embedding_dim:
                logger.warning(
                    f"Requested target_embedding_dim={config.target_embedding_dim}, but PCA is limited to "
                    f"min(N, D)={k_max} (N={n} samples, D={d} features). Using {k} components.",
                )

            if k < d:
                proj = pca_projector(embeddings, out_dim=k)
                embeddings = proj.transform(embeddings)
        else:
            logger.warning(
                f"Cannot reduce embeddings to target_embedding_dim={config.target_embedding_dim}: feature extractor "
                f"already outputs {d}-D embeddings. Using {d} dimensions as-is.",
            )

        embeddings_01 = to_unit_interval_01(embeddings)
        # coverage expects 2D array, shape (N, P), P embedding dimensions
        if embeddings_01.ndim == 1:
            embeddings_01 = embeddings_01[None, :]

        num_observations = min(max(3, int(np.sqrt(len(dataset)))), 20)
        if num_observations >= len(embeddings_01):
            raise ValueError(
                f"Need at least (num_observations + 1) points to compute k-NN coverage, "
                f"got N={len(embeddings_01)} points, requested num_observations={num_observations}. "
                "Please provide more images."
            )

        if config.radius_type == "naive":
            cov_dict = cast(dict, coverage_naive(embeddings_01, num_observations=num_observations))
        else:  # config.radius_type == "adaptive"
            cov_dict = cast(
                dict, coverage_adaptive(embeddings_01, num_observations=num_observations, percent=config.percent)
            )
        cov_dict["total"] = len(dataset)

        # if all uncovered images have the same shape, we can plot them.
        # important not to check shape across all images (expensive), only uncovered ones
        idxs = cov_dict["uncovered_indices"]
        if len({np.asarray(dataset[i][0]).shape for i in idxs}) == 1:
            cov_dict["image"] = dep.plot(dataset, indices=idxs)

        return DataevalBiasOutputs.model_validate({"balance": bal_dict, "diversity": div_dict, "coverage": cov_dict})


def generate_table_of_contents(deck: str) -> dict[str, Any]:  # pragma: no cover
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

    left_item = gd.GradientImage(
        src=Path(__file__).parents[2] / "assets/toc.png",
        width=100,
        height=100,
        top=0.5,
        left=0.5,
    )
    return gd.create_two_item_text_slide(
        deck=deck, title="Bias Table of Contents", left_item=left_item, right_item=right_item
    )


def report_coverage(deck: str, coverage: DataevalBiasCoverageOutputs) -> dict[str, Any]:  # pragma: no cover
    """Format coverage results for Gradient consumption.

    Parameters
    ----------
    coverage
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
        {"Potentially under-represented images": [f"{uncovered_count} of {coverage.total} ({uncovered_percent*100}%)"]},
    )

    content = gd.Text(
        [
            gd.SubText("Description: ", bold=True),
            gd.SubText(
                "Coverage uses AI to identify potentially under-represented images "
                "that warrant further investigation. Under-represented images are "
                "those which are closely-related to, at most, a small amount of other "
                "images in the dataset.\n"
            ),
        ],
        fontsize=22,
    )

    return gd.create_table_text_slide(deck=deck, title=title, text=content, data=cov_df)


def report_balance_metadata_factors(
    deck: str, outputs: DataevalBiasBalanceOutputs
) -> dict[str, Any]:  # pragma: no cover
    """Format balance results (metadata factors) for Gradient consumption.

    Parameters
    ----------
    outputs
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
            gd.SubText("Description: ", bold=True, fontsize=20),
            gd.SubText(
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
            gd.SubText(
                "Values approaching or exceeding 0.5 in the heat map should be "
                "further investigated to prevent a model from potentially learning "
                "a harmful shortcut.",
                fontsize=20,
            )
        ],
    ]

    return gd.create_section_by_item_slide(
        deck=deck,
        title=title,
        heading=heading,
        text=text,
        image_path=temp_image_file(outputs.image_metadata) if outputs.image_metadata is not None else None,
    )


def report_balance_classwise(deck: str, outputs: DataevalBiasBalanceOutputs) -> dict[str, Any]:  # pragma: no cover
    """Format balance results (classwise) for Gradient consumption.

    Parameters
    ----------
    outputs
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
            gd.SubText("Description: ", bold=True, fontsize=20),
            gd.SubText(
                "Balance can also help uncover potential model bias by identifying "
                "relative class imbalance. Correlations between an individual class "
                "and all other class labels indicate that a specific class is "
                "over-represented compared to other classes. This can become a problem "
                "if operational data does not also have this imbalance.",
                fontsize=20,
            ),
        ],
        [
            gd.SubText(
                "Values approaching or exceeding 0.5 in the heat map should be further " "investigated.",
                fontsize=20,
            )
        ],
    ]

    return gd.create_section_by_item_slide(
        deck=deck,
        title=title,
        heading=heading,
        text=text,
        image_path=temp_image_file(outputs.image_classwise),
    )


def report_diversity(deck: str, outputs: DataevalBiasDiversityOutputs) -> dict[str, Any]:  # pragma: no cover
    """Format diversity results for Gradient consumption.

    Parameters
    ----------
    outputs
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
            gd.SubText("Description: ", bold=True, fontsize=20),
            gd.SubText(
                "Diversity measures how well each metadata factor is sampled over its range of "
                "possible values. Values near 1 indicate wide sampling, while values near 0 "
                "indicate imbalanced sampling e.g. all datapoints taking a single value.",
                fontsize=20,
            ),
        ],
        [
            gd.SubText(
                "The categories of most interest are those with values that are between 0.1 "
                "and 0.4. The data for each metadata factor in these ranges should be inspected "
                " to see if the sampled values are appropriate for operational data.",
                fontsize=20,
            )
        ],
        [
            gd.SubText(
                "Values below 0.1 are generally so heavily imbalanced that a genuine problem "
                "should be immediately obvious.",
                fontsize=20,
            )
        ],
    ]

    return gd.create_section_by_item_slide(
        deck=deck,
        title=title,
        heading=heading,
        text=text,
        image_path=temp_image_file(outputs.image),
    )


def report_next_steps(deck: str) -> dict[str, Any]:  # pragma: no cover
    """Generate a report for the next steps.

    This outlines how to investigate issues that may arise during analysis.

    Returns
    -------
    dict[str, Any]
        A dictionary representing the next steps slide.
    """

    dir_ = Path(cache_path() / "bias-artifacts")
    dir_.mkdir(parents=True, exist_ok=True)
    filepath = dir_ / "blank_img.png"
    plot_blank_or_single_image(filepath)

    title = "Bias Analysis"
    heading = "Next Steps\n"
    content = [
        gd.Text(t, fontsize=14)
        for t in (
            "Below are the recommended next steps to investigating issues that may arise during analysis.",
            [gd.SubText("In general:", bold=True)],
            "1. Insert text here",
            "2. Insert text here",
            "3. Insert text here",
        )
    ]

    return {
        "deck": deck,
        "layout_name": "SectionByItem",
        "layout_arguments": {
            gd.SectionByItem.ArgKeys.TITLE.value: title,
            gd.SectionByItem.ArgKeys.LINE_SECTION_HEADING.value: heading,
            gd.SectionByItem.ArgKeys.LINE_SECTION_BODY.value: content,
            gd.SectionByItem.ArgKeys.LINE_SECTION_HALF.value: True,
            gd.SectionByItem.ArgKeys.ITEM_SECTION_BODY.value: filepath,
        },
    }


# ============================================================================
# Markdown Report Generation Functions
# ============================================================================


def generate_table_of_contents_md(md: MarkdownOutput) -> None:
    """Generate Markdown table of contents for the bias report.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    """
    md.add_section(heading="Table of Contents")
    md.add_bulleted_list(
        [
            "[Coverage Analysis](#coverage-analysis)",
            "[Balance Analysis](#balance-analysis)",
            "[Diversity Analysis](#diversity-analysis)",
        ]
    )


def report_coverage_md(md: MarkdownOutput, coverage: DataevalBiasCoverageOutputs) -> None:
    """Format coverage results as Markdown.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    coverage : DataevalBiasCoverageOutputs
        The coverage analysis outputs.
    """
    uncovered_count = len(coverage.uncovered_indices)
    uncovered_percent = round(uncovered_count / coverage.total, 2)

    md.add_section(heading="Coverage Analysis")
    md.add_text(
        "**Description:** Coverage uses AI to identify potentially under-represented images that warrant further "
        "investigation. Under-represented images are those which are closely-related to, at most, a small amount "
        "of other images in the dataset."
    )

    md.add_subsection(heading="Summary")
    md.add_table(
        headers=["Metric", "Value"],
        rows=[
            [
                "Potentially under-represented images",
                f"{uncovered_count} of {coverage.total} ({uncovered_percent*100}%)",
            ],
            ["Coverage radius", f"{coverage.coverage_radius:.4f}"],
        ],
    )

    if coverage.image is not None:
        img_path = temp_image_file(coverage.image)
        md.add_image(img_path, alt_text="Coverage Visualization")


def report_balance_metadata_factors_md(md: MarkdownOutput, outputs: DataevalBiasBalanceOutputs) -> None:
    """Format balance results (metadata factors) as Markdown.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    outputs : DataevalBiasBalanceOutputs
        The balance analysis outputs.
    """
    md.add_section(heading="Balance Analysis - Metadata Factors")
    md.add_text(
        "**Description:** Balance can help uncover potential model bias by identifying spurious correlations "
        "between metadata and class labels. For example, a model might incorrectly learn to associate vehicles "
        "with the metadata 'occlusions' if training images always show vehicles partially hidden by other objects. "
        "This learned behaviour might then fail if a vehicle was to appear fully visible."
    )
    md.add_text(
        "**Interpretation:** Values approaching or exceeding 0.5 in the heat map should be further investigated "
        "to prevent a model from potentially learning a harmful shortcut."
    )

    if outputs.image_metadata is not None:
        img_path = temp_image_file(outputs.image_metadata)
        md.add_image(img_path, alt_text="Balance Metadata Visualization")

    md.add_subsection(heading="Balance Scores")
    md.add_table(dataframe=outputs.balance)


def report_balance_classwise_md(md: MarkdownOutput, outputs: DataevalBiasBalanceOutputs) -> None:
    """Format balance results (classwise) as Markdown.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    outputs : DataevalBiasBalanceOutputs
        The balance analysis outputs.
    """
    md.add_section(heading="Balance Analysis - Class-wise")
    md.add_text(
        "**Description:** Balance can also help uncover potential model bias by identifying relative class "
        "imbalance. Correlations between an individual class and all other class labels indicate that a "
        "specific class is over-represented compared to other classes. This can become a problem if "
        "operational data does not also have this imbalance."
    )
    md.add_text(
        "**Interpretation:** Values approaching or exceeding 0.5 in the heat map should be further investigated."
    )

    img_path = temp_image_file(outputs.image_classwise)
    md.add_image(img_path, alt_text="Balance Classwise Visualization")


def report_diversity_md(md: MarkdownOutput, outputs: DataevalBiasDiversityOutputs) -> None:
    """Format diversity results as Markdown.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    outputs : DataevalBiasDiversityOutputs
        The diversity analysis outputs.
    """
    md.add_section(heading="Diversity Analysis")
    md.add_text(
        "**Description:** Diversity measures how well each metadata factor is sampled over its range of "
        "possible values. Values near 1 indicate wide sampling, while values near 0 indicate imbalanced "
        "sampling e.g. all datapoints taking a single value."
    )

    md.add_text("**Interpretation:**")
    md.add_bulleted_list(
        [
            "The categories of most interest are those with values that are between 0.1 and 0.4. The data for each "
            "metadata factor in these ranges should be inspected to see if the sampled values are appropriate for "
            "operational data.",
            "Values below 0.1 are generally so heavily imbalanced that a genuine problem should be immediately "
            "obvious.",
        ]
    )

    img_path = temp_image_file(outputs.image)
    md.add_image(img_path, alt_text="Diversity Visualization")
    md.add_subsection(heading="Diversity Index Scores")
    md.add_table(dataframe=outputs.factors)
