import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dataeval.detectors.linters import Outliers
from dataeval.metrics.stats import (
    DimensionStatsOutput,
    HashStatsOutput,
    LabelStatsOutput,
    VisualStatsOutput,
    dimensionstats,
    hashstats,
    labelstats,
    visualstats,
)
from dataeval.outputs import SourceIndex

from jatic_ri import cache_path
from jatic_ri.core._utils import deprecated, requires_optional_dependency, squash_repeated_warnings
from jatic_ri.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
    TDataset,
    TMetric,
    TModel,
)
from jatic_ri.core.report import _gradient as gd
from jatic_ri.core.report._markdown import MarkdownOutput
from jatic_ri.core.report._plotting_utils import (
    DIMENSION_LIST,
    RATIO_LIST,
    VISUAL_LIST,
    collect_issues,
    create_metric_dataframe_data,
    label_table,
    plot_blank_or_single_image,
    plot_stat_metrics,
    prepare_histograms,
    prepare_ratio_histograms,
    split_into_chunks,
)

logger = logging.getLogger(__name__)


class DataevalCleaningConfig(CapabilityConfigBase):
    """Configuration options for the Cleaning capability."""


class DataevalCleaningDuplicatesOutputs(CapabilityOutputsBase):
    exact: Sequence[Sequence[int]]
    near: Sequence[Sequence[int]]


class DataevalCleaningDimensionStatsOutputs(CapabilityOutputsBase):
    source_index: Sequence[SourceIndex]
    object_count: list[int]
    image_count: int
    offset_x: np.ndarray
    offset_y: np.ndarray
    width: np.ndarray
    height: np.ndarray
    channels: np.ndarray
    size: np.ndarray
    aspect_ratio: np.ndarray
    depth: np.ndarray
    center: np.ndarray
    distance_center: np.ndarray
    distance_edge: np.ndarray


class DataevalCleaningVisualStatsOutputs(CapabilityOutputsBase):
    source_index: Sequence[SourceIndex]
    object_count: list[int]
    image_count: int
    brightness: np.ndarray
    contrast: np.ndarray
    darkness: np.ndarray
    # missing: np.ndarray  restore after https://gitlab.jatic.net/jatic/aria/dataeval/-/issues/1056
    sharpness: np.ndarray
    # zeros: np.ndarray  restore after https://gitlab.jatic.net/jatic/aria/dataeval/-/issues/1056
    percentiles: np.ndarray


class DataevalCleaningLabelStatsOutputs(CapabilityOutputsBase):
    label_counts_per_class: Mapping[int, int]
    label_counts_per_image: Sequence[int]
    image_counts_per_class: Mapping[int, int]
    image_indices_per_class: Mapping[int, Sequence[int]]
    image_count: int
    class_count: int
    label_count: int
    class_names: Sequence[str]


class DataevalCleaningOutputs(CapabilityOutputsBase):
    duplicates: DataevalCleaningDuplicatesOutputs
    img_outliers: dict[int, dict[str, float]]
    img_dim_stats: DataevalCleaningDimensionStatsOutputs
    img_viz_stats: DataevalCleaningVisualStatsOutputs
    label_stats: DataevalCleaningLabelStatsOutputs
    target_outliers: dict[int, dict[str, float]] | None
    box_dim_stats: DataevalCleaningDimensionStatsOutputs | None
    box_viz_stats: DataevalCleaningVisualStatsOutputs | None
    box_ratio_stats: DataevalCleaningDimensionStatsOutputs | None


class DataevalCleaningRun(CapabilityRunBase[DataevalCleaningConfig, DataevalCleaningOutputs]):
    config: DataevalCleaningConfig
    outputs: DataevalCleaningOutputs

    @requires_optional_dependency("gradient", install_hint="pip install '.[unsupported]'")
    @deprecated(replacement="collect_md_report")
    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:  # noqa: ARG002 # pragma: no cover
        """Collect reports for duplicates and outliers for image and target data.

        Parameters
        ----------
        threshold
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
            A list of slide definitions for the full report.
        """

        deck = self.capability_id

        table_of_contents = generate_table_of_contents(deck)

        outputs = self.outputs

        index2label = self.dataset_metadata[0]["index2label"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        dataset_id = self.dataset_metadata[0]["id"]

        duplicates = generate_duplicates_report(
            deck=deck, duplicates=outputs.duplicates, dataset_size=outputs.label_stats.image_count
        )
        stat_list = generate_stats_report(
            deck=deck,
            img_stats=(outputs.img_dim_stats, outputs.img_viz_stats),
            label_stats=outputs.label_stats,
            box_stats=(
                (outputs.box_dim_stats, outputs.box_viz_stats)
                if outputs.box_dim_stats and outputs.box_viz_stats
                else None
            ),
            ratio_stats=outputs.box_ratio_stats,
            index2label=index2label,
        )
        image_list = generate_image_outliers_report(
            deck=deck,
            img_outliers=outputs.img_outliers,
            img_stats=(outputs.img_dim_stats, outputs.img_viz_stats),
            dataset_size=outputs.label_stats.image_count,
        )

        if outputs.box_dim_stats and outputs.box_viz_stats:
            target_list = generate_target_outliers_report(
                deck=deck,
                target_outliers=outputs.target_outliers,
                box_stats=(outputs.box_dim_stats, outputs.box_viz_stats),
                total_targets=outputs.label_stats.label_count,
            )

            return [
                table_of_contents,
                duplicates,
                stat_list[0],
                *image_list[0:],
                *stat_list[1:],
                *target_list[0:],
                generate_next_steps_report(deck=deck, dataset_id=dataset_id),
            ]

        return [
            table_of_contents,
            duplicates,
            stat_list[0],
            *image_list[0:],
            *stat_list[1:],
            generate_next_steps_report(deck=deck, dataset_id=dataset_id),
        ]

    def collect_md_report(self, threshold: float) -> str:  # noqa: ARG002
        """Collect Markdown-formatted report for duplicates and outliers.

        Parameters
        ----------
        threshold : float
            Minimum acceptable score. Results meeting or exceeding `threshold` are
            considered acceptable. Results below `threshold` require further
            inspection or are treated as failures.

        Returns
        -------
        str
            Markdown-formatted report content.
        """
        outputs = self.outputs
        dataset_id = self.dataset_metadata[0]["id"]
        index2label = self.dataset_metadata[0]["index2label"]  # pyright: ignore[reportTypedDictNotRequiredAccess]

        artifact_dir = Path(cache_path() / "cleaning-artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        md = MarkdownOutput(f"Dataset Cleaning Analysis - {dataset_id}")

        generate_table_of_contents_md(md)

        md.add_section_divider()
        generate_duplicates_report_md(md, outputs.duplicates, outputs.label_stats.image_count)

        md.add_section_divider()
        generate_image_property_histograms_report_md(
            md,
            img_stats=(outputs.img_dim_stats, outputs.img_viz_stats),
        )

        md.add_section_divider()
        generate_image_outliers_report_md(
            md,
            img_outliers=outputs.img_outliers,
            img_stats=(outputs.img_dim_stats, outputs.img_viz_stats),
            dataset_size=outputs.label_stats.image_count,
        )

        md.add_section_divider()
        generate_label_analysis_report_md(
            md,
            label_stats=outputs.label_stats,
            index2label=index2label,
        )

        if outputs.box_dim_stats and outputs.box_viz_stats:
            md.add_section_divider()
            generate_target_property_histograms_report_md(
                md,
                box_stats=(outputs.box_dim_stats, outputs.box_viz_stats),
                ratio_stats=outputs.box_ratio_stats,
            )

            md.add_section_divider()
            generate_target_outliers_report_md(
                md,
                target_outliers=outputs.target_outliers,
                box_stats=(outputs.box_dim_stats, outputs.box_viz_stats),
                total_targets=outputs.label_stats.label_count,
            )

        md.add_section_divider()
        generate_next_steps_report_md(md, dataset_id)

        return md.render()


class DataevalCleaningBase(Capability[DataevalCleaningOutputs, TDataset, TModel, TMetric, DataevalCleaningConfig]):
    """
    Performs dataset cleaning by identifying duplicates (exact and near)
    as well as statistical outliers using various pixel and image
    statistics on the dataset.
    """

    _RUN_TYPE = DataevalCleaningRun

    @classmethod
    def _create_config(cls) -> DataevalCleaningConfig:
        return DataevalCleaningConfig()

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

    def _run_basic_stats(
        self, dataset: TDataset
    ) -> tuple[
        HashStatsOutput,
        DimensionStatsOutput,
        VisualStatsOutput,
        LabelStatsOutput,
    ]:
        """Compute statistics for the images in the dataset.

        Returns
        -------
        tuple[
            HashStatsOutput,
            DimensionStatsOutput,
            VisualStatsOutput,
            LabelStatsOutput,
        ]
            A tuple containing hash, dimension, visual, and label statistics.
        """

        # dataeval hasher does not work for images smaller than 8x8
        # pixels e.g. bbox, resulting in very noisy, repeated warnings.
        # instead we squash the repeated warnings into a single warning
        # and explain to the user what has happened.
        def is_small_perceptual_hash_warning(record: logging.LogRecord) -> bool:
            """Identify dataeval warnings related to perceptual hashing of too-small images."""
            if record.levelno != logging.WARNING:
                return False
            if not record.name.startswith("dataeval."):
                return False
            msg = record.getMessage().lower()
            # Keep this loose so it survives minor wording / formatting changes in dataeval.
            return "perceptual" in msg and "hash" in msg

        with squash_repeated_warnings("dataeval", is_small_perceptual_hash_warning) as filt:
            hashes = hashstats(dataset)
            logger.warning(
                f"Suppressed {filt.count} dataeval perceptual-hash warnings. "
                "This usually occurs when hashing very small crops (e.g., <8x8 px), "
                f"which cannot be perceptually hashed. Example warning: {filt.first}"
            )

        img_dim_stats = dimensionstats(dataset)
        img_viz_stats = visualstats(dataset)

        label_stats = labelstats(dataset)

        return hashes, img_dim_stats, img_viz_stats, label_stats

    def _compute_basic_outliers(
        self, dim_stats: DimensionStatsOutput | None = None, viz_stats: VisualStatsOutput | None = None
    ) -> dict[int, dict[str, float]]:
        """
        Compute z-score-based outliers for selected dimension and visual metrics.

        This method applies a z-score threshold of 3 to identify outliers
        in the dataset's statistics. It filters the results to include
        only categories defined in `DIMENSION_LIST` or `VISUAL_LIST`.

        Parameters
        ----------
        dim_stats : DimensionStatsOutput | None, optional
            Dimension statistics, by default None.
        viz_stats : VisualStatsOutput | None, optional
            Visual statistics, by default None.

        Returns
        -------
        dict[int, dict[str, float]]
            A dictionary of outliers.
        """
        all_outliers_dict = {}

        for stats in [dim_stats, viz_stats]:
            outliers_dict = {}
            if stats is not None:
                base_outliers = Outliers(outlier_method="zscore", outlier_threshold=3).from_stats(stats)

                for k, v in base_outliers.issues.items():
                    filtered_values = {
                        category: value
                        for category, value in v.items()
                        if category in DIMENSION_LIST or category in VISUAL_LIST
                    }
                    if filtered_values:
                        outliers_dict[k] = filtered_values

            if outliers_dict:
                self._dictionary_merge(all_outliers_dict, outliers_dict)

        return all_outliers_dict

    def _outlier_at_1(
        self,
        outlier_result: dict[int, dict[str, float]],
        stats: DimensionStatsOutput,
        categories: list[str],
    ) -> dict[int, dict[str, float]]:
        """
        Identifies outliers in a given metric.

        Enforces an upper threshold of 1.0 for ratio-based metrics when
        necessary. (A ratio of bounding-box to image should never exceed
        one as the bounding-box should always be contained inside the image.)

        This function performs a simple check to see if the value is
        greater than 1.0. If so, it flags the value as an outlier.

        Parameters
        ----------
        outlier_result : dict[int, dict[str, float]]
            The dictionary to store outlier results.
        stats : DimensionStatsOutput
            Dimension statistics.
        categories : list[str]
            List of categories to check.

        Returns
        -------
        dict[int, dict[str, float]]
            The updated outlier results dictionary.
        """
        for category in categories:
            data = getattr(stats, category)
            if over_1 := np.flatnonzero(data > 1).tolist():
                for idx in over_1:
                    if idx not in outlier_result:
                        outlier_result[idx] = {}
                    if f"ratio_{category}" not in outlier_result[idx]:
                        outlier_result[idx].update({f"ratio_{category}": data[idx]})

        return outlier_result

    def _dictionary_merge(
        self, dict1: dict[int, dict[str, float]], dict2: dict[int, dict[str, float]]
    ) -> dict[int, dict[str, float]]:
        """Merge two outlier result dictionaries together.

        Parameters
        ----------
        dict1 : dict[int, dict[str, float]]
            The first dictionary.
        dict2 : dict[int, dict[str, float]]
            The second dictionary.

        Returns
        -------
        dict[int, dict[str, float]]
            The merged dictionary.
        """
        for key, inner in dict2.items():
            if key in dict1:
                dict1[key].update(inner)
            else:
                dict1[key] = inner
        return dict1

    def _compute_ratio_outliers(self, ratio_stats: DimensionStatsOutput) -> dict[int, dict[str, float]]:
        """
        Compute z-score-based outliers for selected ratio metrics.

        This method applies a z-score threshold of 3 to identify outliers
        in the dataset's statistics. It filters the results to include
        only categories defined in `RATIO_LIST`.

        Parameters
        ----------
        ratio_stats : DimensionStatsOutput
            Ratio statistics.

        Returns
        -------
        dict[int, dict[str, float]]
            A dictionary of outliers.
        """
        outliers_dict = {}

        base_outliers = Outliers(outlier_method="zscore", outlier_threshold=3).from_stats(ratio_stats)
        for k, v in base_outliers.issues.items():
            filtered_values = {f"ratio_{category}": value for category, value in v.items() if category in RATIO_LIST}
            if filtered_values:
                outliers_dict[k] = filtered_values

        return outliers_dict

    def _compute_box_outliers(
        self, box_dim_stats: DimensionStatsOutput, box_viz_stats: VisualStatsOutput, ratiostats: DimensionStatsOutput
    ) -> dict[int, dict[str, float]]:
        """
        Compute outliers related to bounding boxes.

        This includes standard visual/dimensional outliers from bounding
        box stats, as well as adjusted outliers for ratio metrics.

        Parameters
        ----------
        box_dim_stats : DimensionStatsOutput
            Bounding box dimension statistics.
        box_viz_stats : VisualStatsOutput
            Bounding box visual statistics.
        ratiostats : DimensionStatsOutput
            Ratio statistics.

        Returns
        -------
        dict[int, dict[str, float]]
            A dictionary of outliers.
        """
        box_result = self._compute_basic_outliers(dim_stats=box_dim_stats, viz_stats=box_viz_stats)
        ratio_result = self._compute_ratio_outliers(ratio_stats=ratiostats)

        ratio_categories = ["offset_x", "offset_y", "width", "height", "size"]
        adjusted_ratio_result = self._outlier_at_1(ratio_result, ratiostats, ratio_categories)

        return self._dictionary_merge(box_result, adjusted_ratio_result)


def add_slide(
    deck: str,
    title: str,
    text: gd.Text,
    metrics_subset: list[str],
    all_metrics: dict[str, list],
    total: int,
    is_images: bool = True,
) -> dict[str, Any]:  # pragma: no cover
    """Add a slide with a table of metrics to the report.

    Parameters
    ----------
    deck : str
        The deck to add the slide to.
    title : str
        The title of the slide.
    text : Text
        The text content of the slide.
    metrics_subset : list[str]
        A subset of metrics to display.
    all_metrics : dict[str, list]
        All available metrics.
    total : int
        The total number of items.
    is_images : bool, optional
        Whether the metrics are for images, by default True.

    Returns
    -------
    dict[str, Any]
        The slide definition.
    """
    metric_df = create_metric_dataframe_data(
        is_images=is_images, metrics_subset=metrics_subset, all_metrics=all_metrics, total=total
    )
    return gd.create_table_text_slide(deck=deck, title=title, text=text, data=metric_df)


def generate_table_of_contents(deck: str) -> dict[str, Any]:  # pragma: no cover
    """Generate a table of contents for the report.

    Returns
    -------
    dict[str, Any]
        The slide definition for the table of contents.
    """
    right_item = [
        "\n",
        "* Image Duplicate Analysis",
        "* Image Property Histograms",
        gd.Text("Used for adjusting the outlier analysis thresholds.", indent=1),
        "* Image Outlier Analysis",
        "* Label Analysis",
        "* Target Property Histograms",
        gd.Text("Used for adjusting the outlier analysis thresholds.", indent=1),
        "* Target Outlier Analysis",
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
        deck=deck, title="Table of Contents", left_item=left_item, right_item=right_item
    )


def generate_duplicates_report(
    deck: str, duplicates: DataevalCleaningDuplicatesOutputs, dataset_size: int
) -> dict[str, Any]:  # pragma: no cover
    """Generate a report for image duplicates.

    Parameters
    ----------
    deck
        Name of the slide deck
    duplicates
        The duplicate analysis outputs.
    dataset_size
        The total size of the dataset.

    Returns
    -------
    dict[str, Any]
        The slide definition for the duplicates report.
    """
    exact = duplicates.exact
    near = duplicates.near

    total_ed = sum(len(d) for d in exact)
    total_nd = sum(len(d) for d in near)

    title = "Image Duplicate Analysis"

    duplicates_df = pd.DataFrame(
        {
            "": ["Percentage of Images", "Number of Images"],
            "Exact Duplicates": [
                f"{total_ed / dataset_size:.2%}",
                f"{total_ed}",
            ],
            "Near Duplicates": [
                f"{total_nd / dataset_size:.2%}",
                f"{total_nd}",
            ],
        }
    )

    content = gd.Text(
        [
            gd.SubText("Description: ", bold=True),
            gd.SubText("Identify images which are identical or almost identical.\n"),
        ],
        fontsize=22,
    )

    return gd.create_table_text_slide(deck=deck, title=title, text=content, data=duplicates_df)


def generate_stats_report(
    deck: str,
    img_stats: tuple[DataevalCleaningDimensionStatsOutputs, DataevalCleaningVisualStatsOutputs],
    label_stats: DataevalCleaningLabelStatsOutputs,
    box_stats: tuple[DataevalCleaningDimensionStatsOutputs, DataevalCleaningVisualStatsOutputs] | None,
    ratio_stats: DataevalCleaningDimensionStatsOutputs | None,
    index2label: dict[int, str],
) -> list[dict[str, Any]]:  # pragma: no cover
    """Generate a report for image and target statistics.

    Parameters
    ----------
    deck
        Name of the slide deck
    img_stats
        Image dimension and visual statistics.
    label_stats
        Label statistics.
    box_stats
        Bounding box dimension and visual statistics.
    ratio_stats
        Ratio statistics.
    index2label
        Mapping from integer labels to corresponding string descriptions.

    Returns
    -------
        A list of slide definitions for the statistics report.
    """
    stat_slides = []

    content = [
        gd.Text("Description: ", bold=True, fontsize=22),
        gd.Text(
            "Visual overview of potential outliers in image properties. Vertical lines are the outlier thresholds"
            "(computed internally). Values outside of the vertical lines will be flagged as outliers.",
            fontsize=22,
        ),
    ]

    # build gradient slide for image outlier histograms
    img_hist_list = prepare_histograms(img_stats)
    dir_ = Path(cache_path() / "cleaning-artifacts")
    dir_.mkdir(parents=True, exist_ok=True)
    title = "Image Property Histograms"
    filepath = dir_ / "img_stats_histogram_plots.png"
    plot_stat_metrics(is_image=True, plot_list=img_hist_list, filepath=filepath)
    stat_slides.append(
        gd.create_item_by_narrow_text_slide(deck=deck, title=title, content=content, body_value=filepath)
    )

    # build gradient slide for label analysis
    result_content, label_df = label_table(label_stats, index2label=index2label)  # pyright: ignore[reportArgumentType]
    title = "Label Analysis"
    content = []
    content.append(gd.Text("Description: ", bold=True, fontsize=22))
    content.append(gd.Text("Numerical analysis of label properties.\n\n", fontsize=22))
    for t in result_content:
        content.append(gd.Text(t, fontsize=16))
    stat_slides.append(
        gd.create_section_by_item_slide_extra_caption(
            deck=deck, title=title, heading=gd.Text(" "), content=content, body_value=label_df
        )
    )

    content = [
        gd.Text("Description: ", bold=True, fontsize=22),
        gd.Text(
            "Visual overview of potential outliers in target properties. Vertical lines are the outlier thresholds"
            " (computed internally). Values outside of the vertical lines will be flagged as outliers.",
            fontsize=22,
        ),
    ]

    if box_stats and ratio_stats:
        box_hist_list = prepare_histograms(box_stats)
        box_hist_list = prepare_ratio_histograms(ratio_stats, box_hist_list)
        dir_ = Path(cache_path() / "cleaning-artifacts")
        dir_.mkdir(parents=True, exist_ok=True)
        filepath = dir_ / "box_stats_histogram_plots.png"
        plot_stat_metrics(is_image=False, plot_list=box_hist_list, filepath=filepath)
        title = "Target Property Histograms"
        stat_slides.append(
            gd.create_item_by_narrow_text_slide(deck=deck, title=title, content=content, body_value=filepath)
        )

    return stat_slides


def generate_image_outliers_report(
    deck: str,
    img_outliers: dict[int, dict[str, float]],
    img_stats: tuple[DataevalCleaningDimensionStatsOutputs, DataevalCleaningVisualStatsOutputs],
    dataset_size: int,
) -> list[dict[str, Any]]:  # pragma: no cover
    """Generate a report for image outliers.

    Parameters
    ----------
    deck
        Name of the slide deck
    img_outliers
        Image outlier data.
    img_stats
        Image dimension and visual statistics.
    dataset_size
        The total size of the dataset.

    Returns
    -------
    list[dict[str, Any]]
        A list of slide definitions for the image outliers report.
    """
    outlier_slides = []
    # chosen based on expert analysis on what is/isn't most relevant to users
    metrics = DIMENSION_LIST + VISUAL_LIST

    dim_box_output, viz_box_output = img_stats
    image_source_indices = list(dim_box_output.source_index) + list(viz_box_output.source_index)

    # construct collection of all bounding boxes with issues
    issues = collect_issues(
        outliers=img_outliers, source_indices=image_source_indices, valid_metrics=metrics, use_box_indices=False
    )

    # now construct slides for outlier data
    all_metrics = {k: issues.get(k, []) for k in metrics if k not in ["channels", "distance_center", "distance_edge"]}
    # looks better if we limit to 4 entries per slide...
    metric_chunks = split_into_chunks(all_metrics, chunk_sizes=[4])
    captions = ["Dimensional", "Visual", "Pixel"]
    for idx, chunk in enumerate(metric_chunks):
        title = f"Image {captions[idx]} Outliers"
        text = gd.Text(
            [
                gd.SubText("Description: ", bold=True),
                f" Numerical analysis of {captions[idx].lower()} outliers in images.",
            ],
            fontsize=21,
        )
        outlier_slides.append(
            add_slide(
                deck=deck,
                title=title,
                text=text,
                metrics_subset=chunk,
                all_metrics=all_metrics,
                total=dataset_size,
                is_images=True,
            )
        )

    return outlier_slides


def generate_target_outliers_report(
    deck: str,
    target_outliers: dict[int, dict[str, float]] | None,
    box_stats: tuple[DataevalCleaningDimensionStatsOutputs, DataevalCleaningVisualStatsOutputs],
    total_targets: int,
) -> list[dict[str, Any]]:  # pragma: no cover
    """Generate a report for target outliers.

    Parameters
    ----------
    deck
        Name of the slide deck
    target_outliers
        Target outlier data.
    box_stats
        Bounding box dimension and visual statistics.
    total_targets
        The total number of targets.

    Returns
    -------
    list[dict[str, Any]]
        A list of slide definitions for the target outliers report,
        or an empty list if no target outliers.
    """
    if target_outliers is None:
        return []

    outlier_slides = []
    metrics = DIMENSION_LIST + VISUAL_LIST + [f"ratio_{cat}" for cat in RATIO_LIST]

    dim_box_output, viz_box_output = box_stats
    box_source_indices = list(dim_box_output.source_index) + list(viz_box_output.source_index)
    total_targets = total_targets

    # construct collection of all bounding boxes with issues
    issues = collect_issues(
        outliers=target_outliers, source_indices=box_source_indices, valid_metrics=metrics, use_box_indices=True
    )

    # now construct slides for outlier data
    all_metrics = {k: issues.get(k, []) for k in metrics if k not in ["ratio_offset_x", "channels"]}
    all_metrics["ratio_offset_y"].extend(issues.get("ratio_offset_x", []))
    # looks better if we chunk according to the number of metrics in each category
    metric_chunks = split_into_chunks(all_metrics, chunk_sizes=[4, 4, 2, 5])
    captions = ["Dimensional", "Visual", "Pixel", "Ratio"]
    for idx, chunk in enumerate(metric_chunks):
        title = f"Target {captions[idx]} Outliers"
        text = gd.Text(
            [
                gd.SubText("Description: ", bold=True),
                f" Numerical analysis of {captions[idx].lower()} outliers in targets.",
            ],
            fontsize=21,
        )
        outlier_slides.append(
            add_slide(
                deck=deck,
                title=title,
                text=text,
                metrics_subset=chunk,
                all_metrics=all_metrics,
                total=total_targets,
                is_images=False,
            )
        )

    return outlier_slides


def generate_next_steps_report(deck: str, dataset_id: str) -> dict[str, Any]:  # pragma: no cover
    """Generate a report for next steps.

    Provides recommendations for investigating issues that may arise
    during analysis.

    Returns
    -------
    dict[str, Any]
        The slide definition for the next steps report.
    """
    dir_ = Path(cache_path() / "cleaning-artifacts")
    dir_.mkdir(parents=True, exist_ok=True)
    filepath = dir_ / "blank_img.png"
    plot_blank_or_single_image(filepath)

    title = f"Dataset: {dataset_id} | Category: Cleaning"
    heading = "Next Steps\n"
    content = [
        gd.Text(t, fontsize=14)
        for t in (
            "Below are the recommended next steps to investigating issues that may arise during analysis.",
            [gd.SubText("In general:", bold=True)],
            "- Remove the images/targets flagged in the Basic Check reports for images and targets",
            "- Manually review the images/targets flagged in the Outlier reports",
            [gd.SubText("For images:", bold=True)],
            "- Check if images come up in multiple outlier categories. If so, remove.",
            "- Make sure images are representative of their respective environment/class. If not, remove.",
            [gd.SubText("For targets:", bold=True)],
            "- Run bias analysis with bounding box stats and ensure there are no correlations between a statistic and a class",  # noqa: E501
            "- Make sure targets are representative of their respective class. If not, remove.",
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
    """Generate Markdown table of contents for the cleaning report.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    """
    md.add_section(heading="Table of Contents")
    md.add_bulleted_list(
        [
            "[Image Duplicate Analysis](#image-duplicate-analysis)",
            "[Image Property Histograms](#image-property-histograms)",
            "[Image Outlier Analysis](#image-outlier-analysis)",
            "[Label Analysis](#label-analysis)",
            "[Target Property Histograms](#target-property-histograms)",
            "[Target Outlier Analysis](#target-outlier-analysis)",
            "[Next Steps](#next-steps)",
        ]
    )


def generate_duplicates_report_md(
    md: MarkdownOutput,
    duplicates: DataevalCleaningDuplicatesOutputs,
    dataset_size: int,
) -> None:
    """Format duplicates results as Markdown.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    duplicates : DataevalCleaningDuplicatesOutputs
        The duplicate analysis outputs.
    dataset_size : int
        The total size of the dataset.
    """
    exact = duplicates.exact
    near = duplicates.near

    total_ed = sum(len(d) for d in exact)
    total_nd = sum(len(d) for d in near)

    md.add_section(heading="Image Duplicate Analysis")
    md.add_text("**Description:** Identify images which are identical or almost identical.")

    md.add_table(
        headers=["", "Exact Duplicates", "Near Duplicates"],
        rows=[
            [
                "Percentage of Images",
                f"{total_ed / dataset_size:.2%}",
                f"{total_nd / dataset_size:.2%}",
            ],
            ["Number of Images", f"{total_ed}", f"{total_nd}"],
        ],
    )


def generate_image_stats_report_md(
    md: MarkdownOutput,
    img_stats: tuple[
        DataevalCleaningDimensionStatsOutputs,
        DataevalCleaningVisualStatsOutputs,
    ],
    label_stats: DataevalCleaningLabelStatsOutputs,
    index2label: dict[int, str],
) -> None:
    """Format image statistics as Markdown.

    Convenience wrapper that combines image histograms and label analysis.
    """
    generate_image_property_histograms_report_md(md, img_stats=img_stats)
    generate_label_analysis_report_md(md, label_stats=label_stats, index2label=index2label)


def generate_image_outliers_report_md(
    md: MarkdownOutput,
    img_outliers: dict[int, dict[str, float]],
    img_stats: tuple[DataevalCleaningDimensionStatsOutputs, DataevalCleaningVisualStatsOutputs],
    dataset_size: int,
) -> None:
    """Format image outliers as Markdown.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    img_outliers : dict[int, dict[str, float]]
        Image outlier data.
    img_stats : tuple[DataevalCleaningDimensionStatsOutputs, DataevalCleaningVisualStatsOutputs]
        Image dimension and visual statistics.
    dataset_size : int
        The total size of the dataset.
    """
    metrics = DIMENSION_LIST + VISUAL_LIST

    dim_box_output, viz_box_output = img_stats
    image_source_indices = list(dim_box_output.source_index) + list(viz_box_output.source_index)

    issues = collect_issues(
        outliers=img_outliers,
        source_indices=image_source_indices,
        valid_metrics=metrics,
        use_box_indices=False,
    )

    all_metrics = {k: issues.get(k, []) for k in metrics if k not in ["channels", "distance_center", "distance_edge"]}

    md.add_section(heading="Image Outlier Analysis")
    md.add_text("**Description:** Numerical analysis of outliers in images.")

    md.add_subsection(heading="Summary")
    total_outlier_images = len(img_outliers)
    md.add_text(f"Total images with outliers: {total_outlier_images} ({total_outlier_images / dataset_size:.2%})")

    dimensional_metrics = [k for k in all_metrics if k in DIMENSION_LIST]
    if dimensional_metrics:
        md.add_subsection(heading="Dimensional Outliers")
        dim_rows = []
        for metric in dimensional_metrics:
            outlier_count = len(all_metrics[metric])
            percentage = outlier_count / dataset_size
            dim_rows.append([metric, str(outlier_count), f"{percentage:.2%}"])

        md.add_table(
            headers=["Metric", "Count", "Percentage"],
            rows=dim_rows,
        )

    visual_metrics = [k for k in all_metrics if k in VISUAL_LIST]
    if visual_metrics:
        md.add_subsection(heading="Visual Outliers")
        vis_rows = []
        for metric in visual_metrics:
            outlier_count = len(all_metrics[metric])
            percentage = outlier_count / dataset_size
            vis_rows.append([metric, str(outlier_count), f"{percentage:.2%}"])

        md.add_table(
            headers=["Metric", "Count", "Percentage"],
            rows=vis_rows,
        )


def generate_target_stats_report_md(
    md: MarkdownOutput,
    box_stats: tuple[
        DataevalCleaningDimensionStatsOutputs,
        DataevalCleaningVisualStatsOutputs,
    ],
    ratio_stats: DataevalCleaningDimensionStatsOutputs | None,
) -> None:
    """Format target statistics as Markdown.

    Convenience wrapper around target property histograms.
    """
    generate_target_property_histograms_report_md(md, box_stats=box_stats, ratio_stats=ratio_stats)


def generate_target_outliers_report_md(
    md: MarkdownOutput,
    target_outliers: dict[int, dict[str, float]] | None,
    box_stats: tuple[DataevalCleaningDimensionStatsOutputs, DataevalCleaningVisualStatsOutputs],
    total_targets: int,
) -> None:
    """Format target outliers as Markdown.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    target_outliers : dict[int, dict[str, float]] | None
        Target outlier data.
    box_stats : tuple[DataevalCleaningDimensionStatsOutputs, DataevalCleaningVisualStatsOutputs]
        Bounding box dimension and visual statistics.
    total_targets : int
        The total number of targets.
    """
    if target_outliers is None:
        return

    metrics = DIMENSION_LIST + VISUAL_LIST + [f"ratio_{cat}" for cat in RATIO_LIST]

    dim_box_output, viz_box_output = box_stats
    box_source_indices = list(dim_box_output.source_index) + list(viz_box_output.source_index)

    issues = collect_issues(
        outliers=target_outliers,
        source_indices=box_source_indices,
        valid_metrics=metrics,
        use_box_indices=True,
    )

    all_metrics = {k: issues.get(k, []) for k in metrics if k not in ["ratio_offset_x", "channels"]}
    if "ratio_offset_x" in issues:
        all_metrics.setdefault("ratio_offset_y", []).extend(issues["ratio_offset_x"])

    md.add_section(heading="Target Outlier Analysis")
    md.add_text("**Description:** Numerical analysis of outliers in targets (bounding boxes).")

    md.add_subsection(heading="Summary")
    total_outlier_targets = len(target_outliers)
    md.add_text(f"Total targets with outliers: {total_outlier_targets} ({total_outlier_targets / total_targets:.2%})")

    dimensional_metrics = [k for k in all_metrics if k in DIMENSION_LIST]
    if dimensional_metrics:
        md.add_subsection(heading="Dimensional Outliers")
        dim_rows = []
        for metric in dimensional_metrics:
            outlier_count = len(all_metrics[metric])
            percentage = outlier_count / total_targets
            dim_rows.append([metric, str(outlier_count), f"{percentage:.2%}"])

        md.add_table(
            headers=["Metric", "Count", "Percentage"],
            rows=dim_rows,
        )

    visual_metrics = [k for k in all_metrics if k in VISUAL_LIST]
    if visual_metrics:
        md.add_subsection(heading="Visual Outliers")
        vis_rows = []
        for metric in visual_metrics:
            outlier_count = len(all_metrics[metric])
            percentage = outlier_count / total_targets
            vis_rows.append([metric, str(outlier_count), f"{percentage:.2%}"])

        md.add_table(
            headers=["Metric", "Count", "Percentage"],
            rows=vis_rows,
        )

    ratio_metrics = [k for k in all_metrics if k.startswith("ratio_")]
    if ratio_metrics:
        md.add_subsection(heading="Ratio Outliers")
        ratio_rows = []
        for metric in ratio_metrics:
            outlier_count = len(all_metrics[metric])
            percentage = outlier_count / total_targets
            ratio_rows.append([metric, str(outlier_count), f"{percentage:.2%}"])

        md.add_table(
            headers=["Metric", "Count", "Percentage"],
            rows=ratio_rows,
        )


def generate_next_steps_report_md(
    md: MarkdownOutput,
    dataset_id: str,  # noqa: ARG001
) -> None:
    """Format next steps as Markdown.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    dataset_id : str
        The dataset identifier.
    """
    md.add_section(heading="Next Steps")
    md.add_text("Below are the recommended next steps to investigating issues that may arise during analysis.")

    md.add_subsection(heading="In General")
    md.add_bulleted_list(
        [
            "Remove the images/targets flagged in the Basic Check reports for images and targets",
            "Manually review the images/targets flagged in the Outlier reports",
        ]
    )

    md.add_subsection(heading="For Images")
    md.add_bulleted_list(
        [
            "Check if images come up in multiple outlier categories. If so, remove.",
            "Make sure images are representative of their respective environment/class. If not, remove.",
        ]
    )

    md.add_subsection(heading="For Targets")
    md.add_bulleted_list(
        [
            "Run bias analysis with bounding box stats and ensure there are no correlations between a statistic "
            "and a class",
            "Make sure targets are representative of their respective class. If not, remove.",
        ]
    )


def generate_image_property_histograms_report_md(
    md: MarkdownOutput,
    img_stats: tuple[
        DataevalCleaningDimensionStatsOutputs,
        DataevalCleaningVisualStatsOutputs,
    ],
) -> None:
    """Markdown analogue of the 'Image Property Histograms' slide."""
    md.add_section(heading="Image Property Histograms")
    md.add_text(
        "**Description:** Visual overview of potential outliers in image properties. "
        "Vertical lines are the outlier thresholds (computed internally). "
        "Values outside of the vertical lines will be flagged as outliers."
    )

    img_hist_list = prepare_histograms(img_stats)

    dir_ = Path(cache_path() / "cleaning-artifacts")
    dir_.mkdir(parents=True, exist_ok=True)
    filepath = dir_ / "img_stats_histogram_plots.png"

    plot_stat_metrics(is_image=True, plot_list=img_hist_list, filepath=filepath)

    md.add_image(filepath, alt_text="Image Property Histograms")


def generate_label_analysis_report_md(
    md: MarkdownOutput,
    label_stats: DataevalCleaningLabelStatsOutputs,
    index2label: dict[int, str],
) -> None:
    """Markdown analogue of the 'Label Analysis' slide.

    NOTE: Keep this Markdown-only. We avoid calling `label_table()` because it is typed
    to accept DataEval's `LabelStatsOutput`, while we hold an internal outputs model.
    """
    md.add_section(heading="Label Analysis")
    md.add_text("Numerical analysis of label properties.")

    avg_labels = float(np.mean(list(label_stats.label_counts_per_image))) if label_stats.label_counts_per_image else 0.0
    md.add_bulleted_list(
        [
            f"**Class Count:** {int(label_stats.class_count)}",
            f"**Label Count:** {int(label_stats.label_count)}",
            f"**Average # Labels per Image:** {round(avg_labels, 2)}",
        ]
    )

    # Prefer the class_names captured in outputs; fall back to index2label if needed.
    class_names = (
        list(label_stats.class_names)
        if label_stats.class_names
        else [index2label.get(i, str(i)) for i in range(int(label_stats.class_count))]
    )

    rows: list[list[str]] = []
    for cls_idx, name in enumerate(class_names):
        total = int(label_stats.label_counts_per_class.get(cls_idx, 0))
        img_count = int(label_stats.image_counts_per_class.get(cls_idx, 0))
        rows.append([str(name), str(total), str(img_count)])

    md.add_table(headers=["Label", "Total Count", "Image Count"], rows=rows)


def generate_target_property_histograms_report_md(
    md: MarkdownOutput,
    box_stats: (
        tuple[
            DataevalCleaningDimensionStatsOutputs,
            DataevalCleaningVisualStatsOutputs,
        ]
        | None
    ),
    ratio_stats: DataevalCleaningDimensionStatsOutputs | None,
) -> None:
    """Markdown analogue of the 'Target Property Histograms' slide."""
    if box_stats is None or ratio_stats is None:
        return

    md.add_section(heading="Target Property Histograms")
    md.add_text(
        "**Description:** Visual overview of potential outliers in target properties. "
        "Vertical lines are the outlier thresholds (computed internally). "
        "Values outside of the vertical lines will be flagged as outliers."
    )

    box_hist_list = prepare_histograms(box_stats)
    box_hist_list = prepare_ratio_histograms(ratio_stats, box_hist_list)

    dir_ = Path(cache_path() / "cleaning-artifacts")
    dir_.mkdir(parents=True, exist_ok=True)
    filepath = dir_ / "box_stats_histogram_plots.png"

    plot_stat_metrics(is_image=False, plot_list=box_hist_list, filepath=filepath)

    md.add_image(filepath, alt_text="Target Property Histograms")
