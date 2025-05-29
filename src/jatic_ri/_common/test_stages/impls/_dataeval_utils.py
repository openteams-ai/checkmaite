import copy
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
from maite.protocols.object_detection import Dataset as ODDataset
from maite.protocols.object_detection import DatumMetadataType, ObjectDetectionTarget
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
    """Check for exception titles, handling both shared and context-specific cases."""
    if metric in EXCEPTIONS:
        for contexts, result in EXCEPTIONS[metric].items():
            if context in contexts and isinstance(result, str):
                return result
            if context in contexts and isinstance(result, dict):
                return result[is_image]

    return ""


def to_title(metric: str, context: Literal["table", "histogram"], is_image: bool = True) -> str:
    """Apply the standard pattern rules based on context and metric type."""

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
    "Prepares histograms for a single metric."
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
    "Computes statistical cutoff values to identify outliers in a numeric array."
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
    "Prepares histograms for each metric in the dimension or visual statistics."
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
    "Prepares histograms for each metric in the dimension statistics of boxratiostats."
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
    """Creates the histogram plot figures and saves them"""

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
    fig, ax = plt.subplots(1, 1, figsize=(1, 1))
    ax.set_visible(False)
    fig.savefig(str(filepath))
    plt.close(fig)


def label_table(label_stats: LabelStatsOutput, index2label: dict[int, str]) -> tuple[list[str], pd.DataFrame]:
    "Creates table of label counts per class and per image."

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


def split_into_chunks(all_metrics: dict[str, Any], chunk_sizes: list[int]) -> list:
    """Split into chunks of specified size."""
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


def create_metric_dataframe_data(is_images: bool, metrics_subset: list, all_metrics: dict, total: int) -> pd.DataFrame:
    """Create dataframe data for a subset of metrics."""

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
    metrics_set = set(valid_metrics)
    issues = {}
    for i, issue in outliers.items():
        indices = source_indices[int(i)]
        img_idx, box_idx, _ = indices
        for k in metrics_set.intersection(issue):
            if use_box_indices:
                issues.setdefault(k, []).append((img_idx, box_idx, issue[k]))
            else:
                issues.setdefault(k, []).append((img_idx, None, issue[k]))
    return issues


def increment_invalid_boxes(dataset: ODDataset) -> ODDataset:
    """
    Ensure bounding boxes have non-zero height and width by incrementing
    bottom and right edges when necessary.
    """

    class _BoxFixedDataset:
        def __init__(self, ds: ODDataset) -> None:
            self._ds = ds
            self.metadata = dataset.metadata

        def __len__(self) -> int:
            return len(self._ds)

        def __getitem__(self, idx: int) -> tuple[ArrayLike, ObjectDetectionTarget, DatumMetadataType]:
            img, target, dmeta = self._ds[idx]

            if not hasattr(target, "boxes"):
                return img, target, dmeta

            fixed_target = copy.deepcopy(target)

            boxes: np.ndarray = np.asarray(fixed_target.boxes)
            if boxes.size:
                invalid_w = boxes[:, 2] <= boxes[:, 0]
                invalid_h = boxes[:, 3] <= boxes[:, 1]

                boxes[invalid_w, 2] = boxes[invalid_w, 0] + 1
                boxes[invalid_h, 3] = boxes[invalid_h, 1] + 1

            return img, fixed_target, dmeta

    return _BoxFixedDataset(dataset)
