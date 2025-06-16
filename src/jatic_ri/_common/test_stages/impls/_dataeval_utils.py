from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as v2
from dataeval.metrics.stats import LabelStatsOutput
from dataeval.outputs import SourceIndex
from dataeval.typing import Transform
from dataeval.utils.torch.models import ResNet18
from gradient import SubText
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from pydantic import BaseModel

# type aliases for user readability
histogram_title = str
histogram_values = np.ndarray
histogram_bin_edges = np.ndarray
histogram_top_cutoff = float
histogram_bot_cutoff = float
histogram_plots_attrs = tuple[
    histogram_title, histogram_values, histogram_bin_edges, histogram_top_cutoff, histogram_bot_cutoff
]

MU = "\u03bc"
DIMENSION_LIST = [
    "width",
    "height",
    "size",
    "aspect_ratio",
]
VISUAL_LIST = [
    "brightness",
    "contrast",
    "darkness",
    "sharpness",
    "missing",
    "zeros",
]
RATIO_LIST = [
    "width",
    "height",
    "size",
    "offset_x",
    "offset_y",
    "aspect_ratio",
]

EXCEPTIONS = {
    "zeros": {
        "table": "High Percentage of Zero Pixels",
        "histogram": {True: "Percentage of Zero \n Values in Image", False: "Percentage of Zero \n Values in Target"},
    },
    "missing": {
        "table": "High Percentage of Missing \n Pixels",
        "histogram": {
            True: "Percentage of Missing \n Pixels in Image",
            False: "Percentage of Missing \n Pixels in Target",
        },
    },
    "no_boxes": {"table": "Image Has No Bounding Boxes", "histogram": "Images with No Bounding Boxes"},
    "ratio_offset_x": {"histogram": "Left Edge:Image \n Width"},
    "ratio_offset_y": {"table": "Uncommon Location of Top or Left Edge", "histogram": "Top Edge:Image \n Height"},
}


def _get_exception_title(metric: str, context: str, is_image: bool) -> str:
    """Check for exception titles, handling both shared and context-specific cases.

    Parameters
    ----------
    metric : str
        The metric name.
    context : str
        The context in which the title is used (e.g., "table", "histogram").
    is_image : bool
        Whether the metric pertains to an image or target.

    Returns
    -------
    str
        The exception title if found, otherwise an empty string.

    """
    if metric in EXCEPTIONS:
        for contexts, result in EXCEPTIONS[metric].items():
            if context in contexts and isinstance(result, str):
                return result
            if context in contexts and isinstance(result, dict):
                return result[is_image]

    return ""


def to_title(metric: str, context: Literal["table", "histogram"], is_image: bool = True) -> str:
    """Convert a metric name to a display-friendly title.

    Applies standard pattern rules based on context and metric type,
    handling exceptions for specific metrics.

    Parameters
    ----------
    metric : str
        The metric name (e.g., "aspect_ratio", "ratio_offset_x").
    context : {"table", "histogram"}
        The context in which the title will be used.
    is_image : bool, optional
        Whether the metric pertains to an image (True) or a target (False)
        (default is True).

    Returns
    -------
    str
        A formatted title string.

    """
    if exception_title := _get_exception_title(metric, context, is_image):
        return exception_title

    if context == "table":
        if metric.startswith("ratio_"):
            base_metric = metric.replace("ratio_", "").replace("_", " ").title()
            return f"Extreme Target {base_metric}:Image {base_metric}"
        # Standard table pattern: "Extreme" + title case metric
        return f"Extreme {metric.replace('_', ' ').title()}"

    if context == "histogram":
        if metric.startswith("ratio_"):
            base_metric = metric.replace("ratio_", "").replace("_", " ").title()
            return f"Target {base_metric}:Image \n {base_metric}"
        return metric.replace("_", " ").title()

    return metric.replace("_", " ").title()


def get_resnet18(dim: int | tuple[int, int] = 128) -> tuple[torch.nn.Module, Transform]:
    model = ResNet18().eval()
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((dim, dim) if isinstance(dim, int) else dim),
        ]
    )
    return model, transform


def prepare_single_histogram(
    metric: str,
    data: np.ndarray,
    top_cutoff: np.floating[Any] | int,
    bot_cutoff: np.floating[Any] | int,
) -> histogram_plots_attrs:
    """Prepare histogram data for a single metric.

    Parameters
    ----------
    metric : str
        Name of the metric.
    data : np.ndarray
        The data for the metric.
    top_cutoff : np.floating[Any] | int
        Upper cutoff value for the histogram.
    bot_cutoff : np.floating[Any] | int
        Lower cutoff value for the histogram.

    Returns
    -------
    histogram_plots_attrs
        A tuple containing the metric name, histogram counts, bin edges,
        and effective top and bottom cutoff edges.

    """
    unique_size = len(np.unique(data))

    if unique_size == 1:
        counts, bins = np.histogram(data, bins=1)
        # for single value histograms, we want to have cutoffs that surround the bin
        return metric, counts, bins, data[0] - 0.5, data[0] + 0.5

    # Empirical bin selection
    if unique_size <= 10:
        bin_count = 10
    elif unique_size < 30:
        bin_count = unique_size
    else:
        bin_count = 30

    counts, bins = np.histogram(data, bins=bin_count)

    # Closest edges to cutoffs
    top_edge = next((b for b in bins if b >= top_cutoff), top_cutoff)
    bot_edge = next((b for b in bins if b <= bot_cutoff), bot_cutoff)

    return metric, counts, bins, float(top_edge), float(bot_edge)


def get_cutoff_values(data: ArrayLike) -> tuple[np.floating, np.floating]:
    """Compute statistical cutoff values to identify outliers in a numeric array.

    Calculates mean and standard deviation, then determines cutoffs as
    mean +/- 3 * std. Ensures bottom cutoff is not unreasonably small.

    Parameters
    ----------
    data : ArrayLike
        Input array of numerical data.

    Returns
    -------
    tuple[np.floating, np.floating]
        A tuple containing the top and bottom cutoff values.

    """
    mean = np.mean(np.asarray(data, dtype=np.float32))
    std = np.std(np.asarray(data, dtype=np.float32))
    # typical z-score cutoff for outliers
    top_cutoff = mean + 3 * std
    bot_cutoff = mean - 3 * std

    # shouldn't be negative for our use-cases and looks better if we set to 0
    if bot_cutoff <= std * 0.1:
        bot_cutoff = np.float32(0)

    return top_cutoff, bot_cutoff


def prepare_histograms(stats: Sequence[BaseModel] | None) -> list[histogram_plots_attrs]:
    """Prepare histogram data for dimension or visual statistics.

    Iterates through metrics in the provided statistics, calculates cutoffs,
    and prepares histogram attributes for each.

    Parameters
    ----------
    stats : Sequence[BaseModel] | None
        A sequence of Pydantic models containing metric data (e.g.,
        DimensionStatsOutput, VisualStatsOutput).

    Returns
    -------
    list[histogram_plots_attrs]
        A list of tuples, each containing histogram attributes for a metric.

    """
    hist_list = []
    if stats:
        for stat in stats:
            for metric, data in stat:
                if metric in DIMENSION_LIST or metric in VISUAL_LIST:
                    # make sure no rogue inf or nan values...
                    data = data[np.isfinite(data)]

                    top_cutoff, bot_cutoff = get_cutoff_values(data=data)

                    hist_list.append(
                        prepare_single_histogram(metric=metric, data=data, top_cutoff=top_cutoff, bot_cutoff=bot_cutoff)
                    )

    return hist_list


def prepare_ratio_histograms(
    stats: BaseModel | None, current_list: list[histogram_plots_attrs]
) -> list[histogram_plots_attrs]:
    """Prepare histogram data for ratio statistics and append to an existing list.

    Iterates through metrics in the provided ratio statistics (e.g., from
    boxratiostats), calculates cutoffs, and prepares histogram attributes,
    then appends them to `current_list`.

    Parameters
    ----------
    stats : BaseModel | None
        A Pydantic model containing ratio metric data.
    current_list : list[histogram_plots_attrs]
        An existing list of histogram attributes to append to.

    Returns
    -------
    list[histogram_plots_attrs]
        The updated list of histogram attributes.

    """
    if stats:
        for metric, data in stats:
            if metric in RATIO_LIST:
                # make sure no rogue inf or nan values...
                data = data[np.isfinite(data)]

                top_cutoff, bot_cutoff = get_cutoff_values(data=data)

                # these metrics are typically normalized to [0, 1]
                if metric != "aspect_ratio":
                    top_cutoff = min(1, top_cutoff)
                    bot_cutoff = 0

                current_list.append(
                    prepare_single_histogram(
                        metric=f"ratio_{metric}", data=data, top_cutoff=top_cutoff, bot_cutoff=bot_cutoff
                    )
                )

    return current_list


def get_optimal_subplot_layout(plot_list_length: int) -> tuple[int, int]:
    """Determine the optimal number of rows and columns for subplots.

    The goal is to minimize empty spaces while preferring a layout
    with as close to 3 rows as possible, using a fixed set of column options.

    Parameters
    ----------
    plot_list_length : int
        The total number of subplots needed.

    Returns
    -------
    tuple[int, int]
        A tuple containing the optimal number of (rows, columns).

    """
    options = [6, 5, 4, 3]
    best_option = 3
    least_empty_spaces = float("inf")
    optimum_rows = 3

    for cols in options:
        rows = int(np.ceil(plot_list_length / cols))
        total_plots = rows * cols
        empty_spaces = total_plots - plot_list_length
        row_dif = rows - 3

        if empty_spaces < least_empty_spaces and row_dif <= optimum_rows and row_dif >= 0:
            least_empty_spaces = empty_spaces
            optimum_rows = row_dif
            best_option = cols

    rows = int(np.ceil(plot_list_length / best_option))
    return rows, best_option


def plot_stat_metrics(is_image: bool, plot_list: list[histogram_plots_attrs], filepath: Path) -> None:
    """Create and save a figure with multiple histogram subplots.

    Parameters
    ----------
    is_image : bool
        Whether the metrics pertain to images (True) or targets (False).
        Used for titling.
    plot_list : list[histogram_plots_attrs]
        A list of tuples, each containing attributes for a single histogram subplot.
    filepath : Path
        The path where the generated plot image will be saved.

    """
    num_rows, num_cols = get_optimal_subplot_layout(len(plot_list))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    axes = axs.flat

    for ax, (metric, counts, bins, top_cutoff, bot_cutoff) in zip(axes, plot_list, strict=False):
        ax.stairs(counts, bins, fill=True)
        ax.set_yscale("log")
        ax.axvline(x=top_cutoff, color="r", linestyle="--", linewidth=2)
        ax.axvline(x=bot_cutoff, color="r", linestyle="--", linewidth=2)
        title = to_title(metric=metric, context="histogram", is_image=is_image)
        ax.set_title(title)
    for j in range(len(plot_list), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(str(filepath))
    plt.close(fig)


def plot_blank_or_single_image(filepath: Path) -> None:
    """Create and save a blank placeholder image.

    Used when an image is expected in a report but none is available.

    Parameters
    ----------
    filepath : Path
        The path where the blank image will be saved.

    """
    fig, ax = plt.subplots(1, 1, figsize=(1, 1))
    ax.set_visible(False)
    fig.savefig(str(filepath))
    plt.close(fig)


def label_table(label_stats: LabelStatsOutput, index2label: dict[int, str]) -> tuple[list[str], pd.DataFrame]:
    """Create a table of label counts per class and per image.

    Parameters
    ----------
    label_stats : LabelStatsOutput
        Statistics about labels.
    index2label : dict[int, str]
        Mapping from class index to label name.

    Returns
    -------
    tuple[list[str], pd.DataFrame]
        A tuple containing a list of summary strings and a DataFrame
        with detailed label counts.

    """
    count_str = []
    count_str.append([SubText("Class Count:", bold=True), f" {label_stats.class_count}"])
    count_str.append([SubText("Label Count:", bold=True), f" {label_stats.label_count}"])
    count_str.append([SubText("Image Count:", bold=True), f" {label_stats.image_count}"])
    count_str.append(
        [
            SubText("Average number of labels per image:", bold=True),
            f"  {round(np.mean(label_stats.label_counts_per_image), 1)}",
        ]
    )
    table_df = pd.DataFrame(
        {
            "Label": [
                index2label[int(idx)] if idx in index2label else None for idx in label_stats.label_counts_per_class
            ],
            "Total Count": list(label_stats.label_counts_per_class.values()),
            "Image Count": list(label_stats.image_counts_per_class.values()),
        }
    )

    return count_str, table_df


def split_into_chunks(all_metrics: dict[str, Any], chunk_sizes: list[int]) -> list[list[str]]:
    """Split a list of metric names into chunks of specified sizes.

    Parameters
    ----------
    all_metrics : dict[str, Any]
        A dictionary where keys are metric names.
    chunk_sizes : list[int]
        A list of integers specifying the size of each chunk.
        If a single integer is provided, all chunks will be of that size.

    Returns
    -------
    list[list[str]]
        A list of lists, where each inner list is a chunk of metric names.

    """
    if len(chunk_sizes) == 1:
        chunk_size = chunk_sizes[0]
        metric_list = list(all_metrics)
        return [metric_list[i : i + chunk_size] for i in range(0, len(metric_list), chunk_size)]
    metric_list = list(all_metrics)
    chunked_metrics = []
    i = 0
    for chunk_size in chunk_sizes:
        chunked_metrics.append(metric_list[i : i + chunk_size])
        i += chunk_size
    return chunked_metrics


def create_metric_dataframe_data(
    is_images: bool, metrics_subset: list[str], all_metrics: dict[str, list], total: int
) -> pd.DataFrame:
    """Create DataFrame data for a subset of metrics, showing percentages and counts.

    Parameters
    ----------
    is_images : bool
        True if metrics pertain to images, False for targets. Used for titling.
    metrics_subset : list[str]
        A list of metric names to include in the DataFrame.
    all_metrics : dict[str, list]
        A dictionary where keys are metric names and values are lists of items
        (e.g., outlier instances) for that metric.
    total : int
        The total number of items (images or targets) in the dataset,
        used for calculating percentages.

    Returns
    -------
    pd.DataFrame
        A DataFrame formatted for display, with metrics as columns and
        "Percentage of Images/Targets" and "Number of Images/Targets" as rows.

    """
    df_data = {"": ["Percentage of Images", "Number of Images"]}

    if is_images:
        df_data.update(
            {
                to_title(k, context="table", is_image=True): [f"{len(v) / total:.2%}", f"{len(v)}"]
                for k, v in all_metrics.items()
                if k in metrics_subset
            }
        )

    else:
        df_data.update(
            {
                to_title(k, context="table", is_image=False): [f"{len(v) / total:.2%}", f"{len(v)}"]
                for k, v in all_metrics.items()
                if k in metrics_subset
            }
        )

    return pd.DataFrame(df_data)


def collect_issues(
    outliers: dict[int, dict[str, float]],
    source_indices: list[SourceIndex],
    valid_metrics: list[str],
    use_box_indices: bool = True,
) -> dict[str, list[tuple[int, int | None, float]]]:
    """Collect and organize outlier issues by metric.

    Parameters
    ----------
    outliers : dict[int, dict[str, float]]
        A dictionary where keys are item indices and values are dictionaries
        of metric names to outlier scores for that item.
    source_indices : list[SourceIndex]
        A list of SourceIndex tuples, mapping item indices to
        (image_index, box_index, original_index).
    valid_metrics : list[str]
        A list of metric names to consider.
    use_box_indices : bool, optional
        If True, include box_index in the output tuples. If False, box_index
        will be None (default is True).

    Returns
    -------
    dict[str, list[tuple[int, int | None, float]]]
        A dictionary where keys are metric names. Values are lists of tuples,
        each representing an outlier: (image_index, box_index or None, outlier_score).

    """
    metrics_set = set(valid_metrics)
    issues: dict[str, list[tuple[int, int | None, float]]] = {}
    for i, issue in outliers.items():
        indices = source_indices[int(i)]
        img_idx, box_idx, _ = indices
        for k in metrics_set.intersection(issue):
            if use_box_indices:
                issues.setdefault(k, []).append((img_idx, box_idx, issue[k]))
            else:
                issues.setdefault(k, []).append((img_idx, None, issue[k]))
    return issues
