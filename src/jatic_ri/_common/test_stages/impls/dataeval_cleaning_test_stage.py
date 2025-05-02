"""DataEval Cleaning Common Test Stage"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dataeval.detectors.linters import Duplicates, Outliers
from dataeval.interop import as_numpy
from dataeval.metrics.stats import (
    DatasetStatsOutput,
    DimensionStatsOutput,
    HashStatsOutput,
    LabelStatsOutput,
    VisualStatsOutput,
)
from gradient.slide_deck.shapes import Text
from gradient.templates_and_layouts.generic_layouts import TextData, TextTableImages
from matplotlib import patches
from matplotlib import pyplot as plt
from more_itertools import take
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from jatic_ri import cache_path
from jatic_ri._common.test_stages.interfaces.plugins import SingleDatasetPlugin, TDataset
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.util.cache import JSONCache, NumpyEncoder

MU = "\u03bc"
DIMENSION_LIST = [
    "width",
    "height",
    "channels",
    "size",
    "aspect_ratio",
]
VISUAL_LIST = [
    "brightness",
    "contrast",
    "darkness",
    "missing",
    "sharpness",
    "zeros",
]
RATIO_LIST = [
    "left",
    "top",
    "width",
    "height",
    "size",
    "aspect_ratio",
    "distance",
]
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


class DatasetCleaningTestStageBase(TestStage[dict[str, Any]], SingleDatasetPlugin[TDataset]):
    """
    Dataset Cleaning TestStage Base implementation.

    Performs dataset cleaning by identifying duplicates (exact and near) as well as statistical outliers
    using various pixel and image statistics on the dataset.
    """

    cache: Cache[dict[str, Any]] | None = JSONCache(encoder=NumpyEncoder, compress=True)

    @property
    def cache_id(self) -> str:
        """Unique cache id for output"""
        return f"cleaning_{self._task}_{self.dataset_id}.dat"

    @property
    def cache_contents_path(self) -> Path:
        """Cache base folder for image artifacts"""
        return cache_path() / f"cleaning_{self._task}_{self.dataset_id}"

    @abstractmethod
    def _run_stats(
        self,
    ) -> tuple[
        HashStatsOutput, DatasetStatsOutput, LabelStatsOutput, DatasetStatsOutput | None, DimensionStatsOutput | None
    ]:
        """Run stats for specific dataset type"""

    @abstractmethod
    def _get_image_label_box(self, index: int, target: int | None) -> tuple[NDArray[Any], str, NDArray[np.int_] | None]:
        """Get image, label and box from dataset at specified index and target"""

    def _run(self) -> dict[str, Any]:
        """Run cleaning"""
        hashes, imgstats, labelstats, boxstats, ratiostats = self._run_stats()
        dupes = Duplicates().from_stats(hashes)
        img_outlier_result = self._basic_outliers(imgstats)

        if boxstats is not None and ratiostats is not None:
            box_outlier_result = self._box_outliers(boxstats, ratiostats)
            return {
                "duplicates": dupes.dict(),
                "imgoutliers": img_outlier_result,
                "targetoutliers": box_outlier_result,
                "imgstats": {
                    k: v for o in [imgstats.dimensionstats, imgstats.visualstats] for k, v in o.dict().items()
                },
                "boxstats": {
                    k: v for o in [boxstats.dimensionstats, boxstats.visualstats] for k, v in o.dict().items()
                },
                "ratiostats": ratiostats.dict(),
                "labelstats": self._adjust_labelstats_keys(labelstats.dict()),
            }
        return {
            "duplicates": dupes.dict(),
            "imgoutliers": img_outlier_result,
            "imgstats": {k: v for o in [imgstats.dimensionstats, imgstats.visualstats] for k, v in o.dict().items()},
            "labelstats": labelstats.dict(),
        }

    def _adjust_labelstats_keys(self, label_dict: dict[str, Any]) -> dict[str, Any]:
        bad_keys = []
        for key, item in label_dict.items():
            if isinstance(item, dict):
                bad_keys.append(key)

        for key in bad_keys:
            old_dict = label_dict[key]
            label_dict[key] = {int(k): v for k, v in old_dict.items()}

        return label_dict

    def _dictionary_update(
        self, dict1: dict[int, dict[str, float]], dict2: dict[int, dict[str, float]]
    ) -> dict[int, dict[str, float]]:
        """Merges two outlier result dictionaries together"""
        set1 = set(dict1.keys())
        set2 = set(dict2.keys())
        intersection = set1.intersection(set2)
        for key in set2:
            if key in intersection:
                dict1[key].update(dict2[key])
            else:
                dict1[key] = dict2[key]
        return dict1

    def _outlier_at_1(
        self,
        stats: DatasetStatsOutput | DimensionStatsOutput,
        data: np.ndarray,
        category: str,
        ratio: bool = False,
    ) -> dict[int, dict[str, float]]:
        """Adjusts outlier threshold to be <= 1"""
        outlier_dict = {}
        mean = np.mean(data)
        std = np.std(as_numpy(data).astype(float))
        cutoff = 3 * std + mean

        if cutoff > 1.0:
            new_thres = float((1 - mean) / std)
            result = Outliers(outlier_method="zscore", outlier_threshold=new_thres).from_stats(stats)

            for k, v in result.issues.items():
                if ratio:
                    keep = {f"ratio_{cat}": val for cat, val in v.items() if cat == category and val > 1.0}
                else:
                    keep = {cat: val for cat, val in v.items() if cat == category and val > 1.0}
                if keep:
                    outlier_dict[int(k)] = keep

        return outlier_dict

    def _basic_outliers(self, stats: DatasetStatsOutput) -> dict[int, dict[str, float]]:
        """Calculates outliers for dimensionstats and visualstats metrics"""
        outlier_result = {}
        bright = self._outlier_at_1(stats, stats.visualstats.brightness, "brightness")
        if bright:
            outlier_result.update(bright)
        dark = self._outlier_at_1(stats, stats.visualstats.darkness, "darkness")
        if dark:
            outlier_result = self._dictionary_update(outlier_result, dark)
        baseoutliers = Outliers(outlier_method="zscore", outlier_threshold=3).from_stats(stats)

        outliers_dict = {}
        for k, v in baseoutliers.issues.items():
            adj_v = {}
            for cat, val in v.items():
                if cat == "brightness" and bright or cat == "darkness" and dark:
                    continue
                if cat in DIMENSION_LIST or cat in VISUAL_LIST:
                    adj_v[cat] = val
            if adj_v:
                outliers_dict[int(k)] = adj_v

        return self._dictionary_update(outlier_result, outliers_dict)

    def _box_outliers(
        self, boxstats: DatasetStatsOutput, ratiostats: DimensionStatsOutput
    ) -> dict[int, dict[str, float]]:
        """Adjusts the outlier threshold for boxratiostats metrics"""
        box_result = self._basic_outliers(boxstats)
        left = self._outlier_at_1(ratiostats, ratiostats.left, "left", True)
        if left:
            box_result = self._dictionary_update(box_result, left)
        top = self._outlier_at_1(ratiostats, ratiostats.top, "top", True)
        if top:
            box_result = self._dictionary_update(box_result, top)
        width = self._outlier_at_1(ratiostats, ratiostats.width, "width", True)
        if width:
            box_result = self._dictionary_update(box_result, width)
        height = self._outlier_at_1(ratiostats, ratiostats.height, "height", True)
        if height:
            box_result = self._dictionary_update(box_result, height)
        size = self._outlier_at_1(ratiostats, ratiostats.size, "size", True)
        if size:
            box_result = self._dictionary_update(box_result, size)

        ar = Outliers(outlier_method="zscore", outlier_threshold=3).from_stats(ratiostats)
        outliers_dict = {}
        for k, v in ar.issues.items():
            adj_v = {f"ratio_{cat}": val for cat, val in v.items() if cat == "aspect_ratio"}
            if adj_v:
                outliers_dict[int(k)] = adj_v

        return self._dictionary_update(box_result, outliers_dict)

    def _histogram_property_list(
        self, stat: dict[str, Any]
    ) -> list[tuple[str, np.ndarray, np.ndarray, float, float, bool]]:
        """Collates the information for plotting the histograms for the dimensionstats and visualstats metrics"""
        hist_list = []
        for metric, data in stat.items():
            if metric in DIMENSION_LIST or metric in VISUAL_LIST:
                data = as_numpy(data).astype(float)
                unique_size = len(np.unique(data))
                mean = np.mean(data)
                std = np.std(data)
                top_cutoff = 3 * std + mean
                bot_cutoff = mean - 3 * std
                if bot_cutoff < std * 0.1:
                    bot_cutoff = 0

                if metric == "brightness" or metric == "darkness":
                    top_cutoff = min(1, top_cutoff)
                    bot_cutoff = 0

                if unique_size == 1:
                    counts, bins = np.histogram(data, bins=1)
                    hist_list.append((metric, counts, bins, -1000.0, -1001.0, False))
                elif unique_size > 1 and unique_size <= 10:
                    counts, bins = np.histogram(data, bins=10)
                    top_closest_edge = bins[bins >= top_cutoff][0] if any(bins >= top_cutoff) else top_cutoff
                    bot_closest_edge = bins[bins <= bot_cutoff][0] if any(bins <= bot_cutoff) else bot_cutoff
                    hist_list.append((metric, counts, bins, float(top_closest_edge), float(bot_closest_edge), False))
                elif unique_size > 10 and unique_size < 30:
                    counts, bins = np.histogram(data, bins=unique_size)
                    top_closest_edge = bins[bins >= top_cutoff][0] if any(bins >= top_cutoff) else top_cutoff
                    bot_closest_edge = bins[bins <= bot_cutoff][0] if any(bins <= bot_cutoff) else bot_cutoff
                    hist_list.append((metric, counts, bins, float(top_closest_edge), float(bot_closest_edge), False))
                elif unique_size > 30:
                    counts, bins = np.histogram(data, bins=30)
                    top_closest_edge = bins[bins >= top_cutoff][0] if any(bins >= top_cutoff) else top_cutoff
                    bot_closest_edge = bins[bins <= bot_cutoff][0] if any(bins <= bot_cutoff) else bot_cutoff
                    hist_list.append((metric, counts, bins, float(top_closest_edge), float(bot_closest_edge), False))

        return hist_list

    def _histogram_ratio_list(
        self, stat: dict[str, Any], current_list: list[tuple[str, np.ndarray, np.ndarray, float, float, bool]]
    ) -> list[tuple[str, np.ndarray, np.ndarray, float, float, bool]]:
        """Collates the information for plotting the histograms for the boxratiostats metrics"""
        for metric, data in stat.items():
            if metric in RATIO_LIST:
                data = as_numpy(data).astype(float)
                unique_size = len(np.unique(data))
                mean = np.mean(data)
                std = np.std(data)
                top_cutoff = 3 * std + mean
                bot_cutoff = mean - 3 * std
                if bot_cutoff < std * 0.1:
                    bot_cutoff = 0

                if metric != "aspect_ratio":
                    top_cutoff = min(1, top_cutoff)
                    bot_cutoff = 0

                if unique_size == 1:
                    counts, bins = np.histogram(data, bins=1)
                    current_list.append((metric, counts, bins, -1000.0, -1001.0, True))
                elif unique_size > 1 and unique_size <= 10:
                    counts, bins = np.histogram(data, bins=10)
                    top_closest_edge = bins[bins >= top_cutoff][0] if any(bins >= top_cutoff) else top_cutoff
                    bot_closest_edge = bins[bins <= bot_cutoff][0] if any(bins <= bot_cutoff) else bot_cutoff
                    current_list.append((metric, counts, bins, float(top_closest_edge), float(bot_closest_edge), True))
                elif unique_size > 10 and unique_size < 30:
                    counts, bins = np.histogram(data, bins=unique_size)
                    top_closest_edge = bins[bins >= top_cutoff][0] if any(bins >= top_cutoff) else top_cutoff
                    bot_closest_edge = bins[bins <= bot_cutoff][0] if any(bins <= bot_cutoff) else bot_cutoff
                    current_list.append((metric, counts, bins, float(top_closest_edge), float(bot_closest_edge), True))
                elif unique_size > 30:
                    counts, bins = np.histogram(data, bins=30)
                    top_closest_edge = bins[bins >= top_cutoff][0] if any(bins >= top_cutoff) else top_cutoff
                    bot_closest_edge = bins[bins <= bot_cutoff][0] if any(bins <= bot_cutoff) else bot_cutoff
                    current_list.append((metric, counts, bins, float(top_closest_edge), float(bot_closest_edge), True))

        return current_list

    def _label_table(self, labelstat: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
        """Creates strings for the resulting counts and per class table from the labelstats"""
        # Display basic counts
        count_str = ["**Result:**"]
        count_str += [f"Class Count:    {labelstat['class_count']}"]
        count_str += [f"Label Count:    {labelstat['label_count']}"]
        count_str += [f"Image Count:   {labelstat['image_count']}"]
        count_str += [f"Average # of\nLabels per Image:  {round(np.mean(labelstat['label_counts_per_image']), 1)}"]

        # Display counts per class in a table
        table_df = pd.DataFrame(
            {
                "Label": [
                    self._get_key_from_dict(self.dataset.metadata["index2label"], idx)
                    if "index2label" in self.dataset.metadata
                    else idx
                    for idx in labelstat["label_counts_per_class"]
                ],
                "Total Count": [
                    self._get_key_from_dict(labelstat["label_counts_per_class"], idx)
                    for idx in labelstat["label_counts_per_class"]
                ],
                "Image Count": [
                    self._get_key_from_dict(labelstat["image_counts_per_label"], idx)
                    for idx in labelstat["label_counts_per_class"]
                ],
            }
        )

        return count_str, table_df

    def _to_pct_2(self, dividend: float, divisor: float) -> tuple[str, str]:
        """Format a dividend and divisor as tuple of 'percent%' 'dividend/divisor'"""
        return f"{dividend/divisor:.2%}", f"{dividend}/{divisor}"

    def _to_pct_1(self, dividend: float, divisor: float) -> str:
        """Format a dividend and divisor as a single string of 'percent% dividend/divisor'"""
        return " ".join(self._to_pct_2(dividend, divisor))

    def _to_title(self, text: str) -> str:
        """Format a string as a title by replacing '_' with ' ' and capitalizing"""
        if text.startswith("ratio_"):
            title = self._adjust_plot_title(text[6:]).title()
            if title == "Aspect Ratio":
                title = "Target:Image Ratio\nof Aspect Ratio"
        else:
            title = self._adjust_plot_title(text).title()
        return title

    def _adjust_plot_title(self, metric: str) -> str:
        """Creates histogram plot titles for specific metrics"""
        if metric == "zeros":
            title = "% of Pixels with Value 0"
        elif metric == "missing":
            title = "% of Pixels Missing"
        elif metric == "top":
            title = "Target Location to\nImage Height Ratio"
        elif metric == "left":
            title = "Target Location to\nImage Width Ratio"
        elif metric == "distance":
            title = "Target Center Distance\nFrom Image Center"
        else:
            title = metric.replace("_", " ")

        return title

    def _get_optimal_subplot_layout(self, plot_list_length: int) -> tuple[int, int]:
        """Calculates the most efficient layout for the histogram plots"""
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

    def _plot_stat_metrics(self, name: str, plot_list: list) -> Path:
        """Creates the histogram plot figures and saves them"""
        self.cache_contents_path.mkdir(parents=True, exist_ok=True)
        filepath = Path(self.cache_contents_path, f"{name}_histogram_plots.png")
        num_rows, num_cols = self._get_optimal_subplot_layout(len(plot_list))

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
        axes = axs.flat

        for ax, (metric, counts, bins, top_cutoff, bot_cutoff, ratio) in zip(axes, plot_list):
            ax.stairs(counts, bins, fill=True)
            ax.set_yscale("log")
            if top_cutoff > 0:
                ax.axvline(x=top_cutoff, color="r", linestyle="--", linewidth=2)
            if bot_cutoff >= 0:
                ax.axvline(x=bot_cutoff, color="r", linestyle="--", linewidth=2)
            title = self._adjust_plot_title(metric).title()
            if metric == "aspect_ratio" and ratio:
                title = "Target:Image Ratio of\n" + title
            elif metric in RATIO_LIST and ratio:
                title = "Target:Image " + title + " Ratio"
            ax.set_title(title)
        for j in range(len(plot_list), len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        fig.savefig(str(filepath))
        plt.close(fig)

        return filepath

    def _create_png_from_text(
        self,
        text: str,
        filename: str = "labelstats_table.png",
        font_size: int = 24,
        font_color: str = "black",
        bg_color: str = "white",
    ) -> Path:
        """Creates a PNG image from the labelstats table to ensure it fits in the space"""
        self.cache_contents_path.mkdir(parents=True, exist_ok=True)
        filepath = Path(self.cache_contents_path, filename)
        # Choose a font (you might need to specify the full path to the font file)
        try:
            font = ImageFont.truetype("FreeMono.ttf", font_size)  # Common font, adjust as needed
        except OSError:
            font = ImageFont.load_default()

        # Calculate text size
        dummy_draw = ImageDraw.Draw(Image.new("RGB", (0, 0), bg_color))
        left, top, right, bottom = dummy_draw.multiline_textbbox((0, 0), text, font=font, font_size=font_size)
        text_width, text_height = right - left, bottom - top

        # Create image with appropriate size
        img = Image.new("RGB", (int(text_width + 20), int(text_height + 20)), bg_color)  # Add padding
        d = ImageDraw.Draw(img)

        # Draw text in the center
        d.multiline_text((10, 10), text, font=font, fill=font_color, font_size=font_size)

        img.save(filepath)
        return filepath

    def _plot_blank_or_single_image(self, check: bool, filepath: Path, plot_items: list) -> None:
        """Plot a blank or single image"""
        if not check:
            fig, ax = plt.subplots(1, 1, figsize=(1, 1))
            ax.set_visible(False)
            fig.savefig(str(filepath))
            plt.close(fig)
        else:
            metric, img_num, box_num, value = plot_items
            fig, ax = plt.subplots(1, 1, figsize=(2, 2))
            group_title = "Targets Outside of Image" if metric == "ratio_top" else self._to_title(metric)
            image, label, box = self._get_image_label_box(img_num, box_num)
            if value is not None:
                val = str(round(value, 3)) if value < 1 else str(round(value, 2))
                title = f"{group_title}\nimg:{img_num}, label:{label}\n value:{val}"
            else:
                title = f"{group_title}\nimg:{img_num}"
            ax.axis("off")
            ax.set_title(title)
            ax.imshow(np.moveaxis(as_numpy(image), 0, -1))
            if box is not None:
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
            fig.savefig(str(filepath))
            plt.close(fig)

    def _plot_sanity_images(self, name: str, sanity_dict: dict[str, list[Any]], image_check: bool) -> Path:
        """Create and save the plot of all of the sample images for the sanity checks"""
        self.cache_contents_path.mkdir(parents=True, exist_ok=True)
        filepath = Path(self.cache_contents_path, f"{name}_sanity_plot.png")
        selected_groups = [
            (metric, min(3, len(sanity_dict[metric]))) for metric in sanity_dict if len(sanity_dict[metric]) > 0
        ]
        num_rows = max([group[1] for group in selected_groups] + [1])
        num_cols = len(selected_groups)

        if not image_check or len(selected_groups) == 1:
            metric = selected_groups[0][0] if image_check else ""
            img_num, box_num, value = sanity_dict[metric][0][0] if image_check else 0, 0, 0
            plot_params = [metric, img_num, box_num, value]
            self._plot_blank_or_single_image(image_check, filepath, plot_params)
            return filepath

        fig = plt.figure(figsize=(num_cols * 3, num_rows * 3))
        subfigs = fig.subfigures(1, num_cols) if num_cols > 1 else fig
        for i in range(num_cols):
            metric = selected_groups[i][0]
            axs = subfigs[i].subplots(num_rows, 1) if isinstance(subfigs, np.ndarray) else fig.subplots(num_rows, 1)
            group_title = "Targets Outside of Image" if metric == "ratio_top" else self._to_title(metric)
            if isinstance(subfigs, np.ndarray):
                subfigs[i].suptitle(group_title)
            else:
                fig.suptitle(group_title)
            metric_list = sorted(sanity_dict[metric], key=lambda x: x[0])
            for j in range(num_rows):
                ax = axs[j]
                if j >= selected_groups[i][1]:
                    ax.set_visible(False)
                    continue
                img_num, box_num, value = metric_list[j]
                image, label, box = self._get_image_label_box(img_num, box_num)
                if value is not None:
                    val = str(round(value, 3)) if value < 1 else str(round(value, 2))
                    title = f"img:{img_num}, label:{label}\n value:{val}"
                else:
                    title = f"img:{img_num}"
                ax.axis("off")
                ax.set_title(title)
                ax.imshow(np.moveaxis(as_numpy(image), 0, -1))
                if box is not None:
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none",
                    )
                    ax.add_patch(rect)
        fig.tight_layout()
        fig.savefig(str(filepath))
        plt.close(fig)

        return filepath

    def _plot_outlier_images(
        self, name: str, issues: dict[str, list[tuple[int, int | None, float]]]
    ) -> dict[str, list[Path]]:
        """Create and save each individual outlier image for gradient"""
        images: dict[str, list[Path]] = {}
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.set_aspect("equal")
        self.cache_contents_path.mkdir(parents=True, exist_ok=True)
        for k, targets in issues.items():
            for i, ti, value in (targets[_] for _ in range(min(len(targets), 3))):
                filepath = Path(self.cache_contents_path, f"{name}_{k}_{i}_{ti}.png")
                images.setdefault(k, []).append(filepath)
                if not filepath.exists():
                    image, label, box = self._get_image_label_box(i, ti)
                    title = f"img:{i} label:{label}\n value: {str(round(value, 2))}"
                    ax.clear()
                    ax.axis("off")
                    ax.set_title(title)
                    ax.imshow(np.moveaxis(as_numpy(image), 0, -1))
                    if box is not None:
                        rect = patches.Rectangle(
                            (box[0], box[1]),
                            box[2] - box[0],
                            box[3] - box[1],
                            linewidth=2,
                            edgecolor="r",
                            facecolor="none",
                        )
                        ax.add_patch(rect)
                    fig.tight_layout()
                    fig.savefig(str(filepath))
        plt.close(fig)

        return images

    def _get_image_size(self, num_cols: int, num_imgs: int) -> float:
        """Determine the number of columns and images to get the correct size"""
        if num_cols == 2:
            img_size = 3.0 if num_imgs <= 2 else 1.5
        elif num_cols <= 4:
            img_size = 2.0 if num_imgs <= 2 else 1.5
        elif num_cols < 7:
            img_size = 1.25
        else:
            img_size = 1.0
        return img_size

    def _split_data_into_chunks(self, all_categories: dict, chunk_size: int) -> list:
        """Split the categories into chunks of the specified size."""
        category_list = list(all_categories)
        return [category_list[i : i + chunk_size] for i in range(0, len(category_list), chunk_size)]

    def _create_category_dataframe_data(
        self, categories_subset: list, all_categories: dict, category_means: dict, total: int
    ) -> dict:
        """Create dataframe data for a subset of categories."""
        df_data = {"Category": ["% of Total", "# of Images", "Mean"]}
        df_data.update(
            {
                self._to_title(k): [
                    *self._to_pct_2(len(v), total),
                    f"{MU} = {category_means[k]:.2f}",
                ]
                for k, v in all_categories.items()
                if k in categories_subset
            }
        )
        return df_data

    def _get_key_from_dict(self, dictionary: dict, key: Any) -> Any:
        """Get the correct key type"""
        if key in dictionary:
            return dictionary[key]

        alternate_key = str(key) if isinstance(key, int) else int(key)
        if alternate_key in dictionary:
            return dictionary[alternate_key]

        return None

    def _generate_duplicates_report(self, with_images: bool = False) -> dict[str, Any]:
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
        source_index = self.outputs["imgstats"]["source_index"]
        len_ds = len(source_index)

        total_ed = sum(len(d) for d in exact)
        total_nd = sum(len(d) for d in near)
        total = total_ed + total_nd

        if with_images:
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
            self.cache_contents_path.mkdir(parents=True, exist_ok=True)

            near_dp_kwargs = {}
            for i, nd in enumerate(examples):
                for j, dl in enumerate(nd):
                    filepath = Path(self.cache_contents_path, f"dupe_{i}_{dl[0]}_{len(dl)}.png")
                    if not filepath.exists():
                        for di, ax in zip(dl, axes):
                            ii, ti, _ = self._get_key_from_dict(source_index, di)
                            image, _, _ = self._get_image_label_box(ii, ti)
                            title = f"image: {ii}"
                            ax.clear()
                            ax.axis("off")
                            ax.set_title(title)
                            ax.imshow(np.moveaxis(as_numpy(image), 0, -1))
                        fig.tight_layout(pad=0.3)
                        fig.savefig(str(filepath))
                    near_dp_kwargs[IMAGE_ARGKEYS[i][j]] = filepath

            plt.close(fig)

            # Gradient slide kwargs
            title = f"Dataset: {self.dataset_id} | Category: Cleaning"
            heading = "Duplicates"
            content = [
                Text(t, fontsize=14)
                for t in (
                    "**Result:**",
                    f"- Total Duplicates: {self._to_pct_1(total, len_ds)}",
                    "**Tests for:**",
                    "- Uncleaned data",
                    "**Risks:**",
                    "- Leakage",
                    "- Lack of robustness",
                    "- Poor generalization",
                    "**Action:**",
                    f"- {'No action required' if total == 0 else 'Evaluate duplicates and clean data'}",
                    "",
                    "Near duplicates uses a perceptual hash which detects rescaling/noise/smoothing,"
                    " but cannot detect rotation, skew, large transformations or large crops.",
                )
            ]
            max_num_imgs = max([len(img_set) for img_set in examples])
            img_size = 4.0 if max_num_imgs <= 2 else 2.5

            # Set up Gradient slide
            return {
                "deck": self._deck,
                "layout_name": "TextTableImages",
                "layout_arguments": {
                    TextTableImages.ArgKeys.TITLE.value: title,
                    TextTableImages.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                    TextTableImages.ArgKeys.TEXT_COLUMN_BODY.value: content,
                    TextTableImages.ArgKeys.DATA_COLUMN_TABLE.value: duplicates_df,
                    TextTableImages.ArgKeys.IMAGE_WIDTH.value: img_size,
                    TextTableImages.ArgKeys.IMAGE_HEIGHT.value: img_size / 2,
                    TextTableImages.ArgKeys.IMAGE_Y_PADDING.value: 0.01,
                    **near_dp_kwargs,
                },
            }

        duplicates_df = pd.DataFrame(
            {
                "Duplicate Type": ["% of Total Images", "# Duplicates/Total Images", "# of Unique Images"],
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

        # Gradient slide kwargs
        title = f"Dataset: {self.dataset_id} | Category: Cleaning"
        heading = "Duplicates"
        content = [
            Text(t, fontsize=14)
            for t in (
                "**Result:**",
                f"- Total Duplicates: {self._to_pct_1(total, len_ds)}",
                "**Tests for:**",
                "- Uncleaned data",
                "**Risks:**",
                "- Leakage",
                "- Lack of robustness",
                "- Poor generalization",
                "**Action:**",
                f"- {'No action required' if total == 0 else 'Evaluate duplicates and clean data'}",
                "",
                "Near duplicates uses a perceptual hash which detects rescaling/noise/smoothing,"
                " but cannot detect rotation, skew, large transformations or large crops.",
            )
        ]

        # Set up Gradient slide
        return {
            "deck": self._deck,
            "layout_name": "TextData",
            "layout_arguments": {
                TextData.ArgKeys.TITLE.value: title,
                TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                TextData.ArgKeys.DATA_COLUMN_TABLE.value: duplicates_df,
            },
        }

    def _generate_stats_report(self) -> list[dict[str, Any]]:
        """
        Generate the statistics report which consists of two slides with
        histogram plots from the statistic results.

        - Slide Format: `OneImageText`
        - Text Content: Explanation of slide
        - Image Content: Histogram plots
        """
        histogram_paths = {}
        imgstats: dict[str, Any] = self.outputs["imgstats"]
        labelstats: dict[str, Any] = self.outputs["labelstats"]
        boxstats: dict[str, Any] = self.outputs.get("boxstats", {})
        ratiostats: dict[str, Any] = self.outputs.get("ratiostats", {})

        img_hist_list = self._histogram_property_list(imgstats)
        histogram_paths["image"] = self._plot_stat_metrics("imgstats", img_hist_list)

        if boxstats:
            box_hist_list = self._histogram_property_list(boxstats)
            box_hist_list = self._histogram_ratio_list(ratiostats, box_hist_list)
            histogram_paths["target"] = self._plot_stat_metrics("boxstats", box_hist_list)

        result_content, label_df = self._label_table(labelstats)
        additional_content = [
            "",
            "**Label Distribution Check:**",
            "- Use slide for a quick check on the result\n   of the label counts across classes and images.",
            "- Helps with understanding your dataset",
        ]

        # Gradient slide kwargs
        title = f"Dataset: {self.dataset_id} | Category: Cleaning"
        heading = "**Label Analysis**\n"
        content = [Text(t, fontsize=14) for t in result_content + additional_content]

        # Set up Gradient slide
        stat_slides = [
            {
                "deck": self._deck,
                "layout_name": "TextData",
                "layout_arguments": {
                    TextData.ArgKeys.TITLE.value: title,
                    TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                    TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                    TextData.ArgKeys.DATA_COLUMN_TABLE.value: label_df,
                },
            }
        ]

        for group in histogram_paths:
            # Gradient slide kwargs
            title = f"Dataset: {self.dataset_id} | Category: Cleaning"
            heading = f"**{str(group).capitalize()} Property Histograms**\n"
            content = [
                Text(t, fontsize=14)
                for t in (
                    "**Distribution Check:**",
                    "- Red vertical lines are the Outlier Thresholds",
                    "- Use slide to check threshold placement",
                    "- Values outside of the vertical lines will be flagged as outliers",
                )
            ]

            # Set up Gradient slide
            stat_slides.append(
                {
                    "deck": self._deck,
                    "layout_name": "TextData",
                    "layout_arguments": {
                        TextData.ArgKeys.TITLE.value: title,
                        TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                        TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                        TextData.ArgKeys.DATA_COLUMN_IMAGE.value: histogram_paths[group],
                    },
                }
            )

        return stat_slides

    def _image_outliers_with_images(
        self,
        sanity_check: dict[str, list[tuple[int, int | None, float]]],
        issues: dict[str, list[tuple[int, int | None, float]]],
        img_stats: dict[str, Any],
        box_stats: dict[str, Any],
        total_sanity: int,
        total_imgs: int,
        total_out: int,
    ) -> list[dict[str, Any]]:
        """Generate the slides for Image Outliers with images on slides"""
        sanity_plot = self._plot_sanity_images("image", sanity_check, total_sanity > 0)

        # Gradient slide kwargs
        title = f"Dataset: {self.dataset_id} | Category: Cleaning"
        heading = "Basic Image Check\n"
        action_text = "No action required" if total_sanity == 0 else "Recommended removing the images"
        if box_stats:
            content = [
                Text(t, fontsize=14)
                for t in (
                    "**Result:**",
                    f"- Images without Targets:   {len(sanity_check['no_boxes'])}",
                    f"- Extreme % of 0 Values:    {len(sanity_check['zeros'])}",
                    f"- Extreme % of Missing Values: {len(sanity_check['missing'])}",
                    f"- Extreme Target Sizes:     {len(sanity_check['size'])}",
                    f"- Extreme Target Aspect Ratios: {len(sanity_check['aspect_ratio'])}",
                    "**Tests for:**",
                    "- Data processing errors",
                    "**Action:**",
                    f"- {action_text}",
                )
            ]
        else:
            content = [
                Text(t, fontsize=14)
                for t in (
                    "**Result:**",
                    f"- Extreme % of 0 Values:    {len(sanity_check['zeros'])}",
                    f"- Extreme % of Missing Values: {len(sanity_check['missing'])}",
                    f"- Extreme Target Sizes:     {len(sanity_check['size'])}",
                    f"- Extreme Target Aspect Ratios: {len(sanity_check['aspect_ratio'])}",
                    "**Tests for:**",
                    "- Data processing errors",
                    "**Action:**",
                    f"- {action_text}",
                )
            ]

        # Set up gradient slide
        outlier_slide = [
            {
                "deck": self._deck,
                "layout_name": "TextData",
                "layout_arguments": {
                    TextData.ArgKeys.TITLE.value: title,
                    TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                    TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                    TextData.ArgKeys.DATA_COLUMN_IMAGE.value: sanity_plot,
                },
            }
        ]

        issues = dict(take(8, sorted(issues.items(), key=lambda x: -len(x[1]))))

        issue_means: dict[str, float] = {}
        for k in issues:
            value = np.mean(img_stats[k])
            issue_means[k] = value
            issues[k] = sorted(issues[k], key=lambda x: -np.abs(x[2] - value))

        outliers_df = pd.DataFrame(
            {
                self._to_title(k): [
                    self._to_pct_2(len(v), total_imgs)[1],
                    f"{MU} = {issue_means[k]:.2f}",
                ]
                for k, v in issues.items()
            }
        )

        for i in range(max(2 - len(outliers_df), 0)):
            outliers_df[" " * i] = ["", ""]

        images = self._plot_outlier_images("image", issues)

        image_kwargs = {
            IMAGE_ARGKEYS[i][j]: img_path
            for i, img_list in enumerate(images.values())
            for j, img_path in enumerate(img_list)
        }

        # Gradient slide kwargs
        title = f"Dataset: {self.dataset_id} | Category: Cleaning"
        heading = "Image Outliers\n"
        content = [
            Text(t, fontsize=14)
            for t in (
                "**Result:**",
                f"- Total Outliers: {self._to_pct_1(total_out, total_imgs)}",
                "**Tests for:**",
                "- Uncleaned data",
                "**Risks:**",
                "- Lack of robustness",
                "- Poor real-world performance",
                "- Poor generalization",
                "**Action:**",
                f"- {'No action required' if total_out == 0 else 'Evaluate outliers and clean data'}",
                "",
                "",
                "Last row of table is the mean for the metric",
            )
        ]
        max_num_cols = len(outliers_df)
        max_num_imgs = max([len(img_set) for img_set in images.values()] + [0])
        img_size = self._get_image_size(max_num_cols, max_num_imgs)

        # Set up Gradient slide
        outlier_slide.append(
            {
                "deck": self._deck,
                "layout_name": "TextTableImages",
                "layout_arguments": {
                    TextTableImages.ArgKeys.TITLE.value: title,
                    TextTableImages.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                    TextTableImages.ArgKeys.TEXT_COLUMN_BODY.value: content,
                    TextTableImages.ArgKeys.DATA_COLUMN_TABLE.value: outliers_df,
                    TextTableImages.ArgKeys.IMAGE_WIDTH.value: img_size,
                    TextTableImages.ArgKeys.IMAGE_HEIGHT.value: img_size,
                    TextTableImages.ArgKeys.IMAGE_Y_PADDING.value: 0.01,
                    **image_kwargs,
                },
            }
        )

        return outlier_slide

    def _generate_image_outliers_report(self, with_images: bool = False) -> list[dict[str, Any]]:
        """
        Generate the image outliers report which consists of two slides.
        The first slide for both images and targets is a basic data check slide.
        The second slide then displays the top (up to 8) outlier categories
        from `DimensionStatsOutput` and `VisualStatsOutput`.

        - Slide Format: `TextTableImages`
        - Text Content: Explanation of test, risks and action item
        - Table Content: Table of outliers identified for the analysis performed
        - Image Content: Sample images of identified outliers with labels
        """
        outliers: dict[int, dict[str, float]] = self.outputs["imgoutliers"]
        img_stats: dict[str, Any] = self.outputs["imgstats"]
        box_stats: dict[str, Any] = self.outputs.get("boxstats", {})

        # Formatting the issues for use
        source_index: list[tuple[int, int | None, int | None]] = img_stats["source_index"]
        issues: dict[str, list[tuple[int, int | None, float]]] = {}
        categories = [
            s
            for output_type in (DimensionStatsOutput, VisualStatsOutput)
            for s in output_type.__annotations__
            if s not in ["left", "top", "depth", "center", "distance", "percentiles"]
        ]
        for i, issue in outliers.items():
            img_num, _, _ = source_index[int(i)]
            for k in (s for s in categories if s in issue):
                issues.setdefault(k, []).append((img_num, None, issue[k]))

        # Image Sanity check
        sanity_check = {}
        for metric in ["zeros", "missing", "size", "aspect_ratio"]:
            sanity_check[metric] = issues.get(metric, [])
        if box_stats:
            sanity_check["no_boxes"] = [
                (int(val), None, None) for val in np.nonzero(as_numpy(box_stats["box_count"]) == 0)[0].tolist()
            ]
        total_sanity = sum(len(sanity_check[metric]) for metric in sanity_check)
        total_imgs = len(img_stats["box_count"])
        total_out = len(outliers)

        if with_images:
            return self._image_outliers_with_images(
                sanity_check, issues, img_stats, box_stats, total_sanity, total_imgs, total_out
            )

        # Gradient slide kwargs
        title = f"Dataset: {self.dataset_id} | Category: Cleaning"
        heading = "Basic Image Check\n"
        action_text = "No action required" if total_sanity == 0 else "Recommended removing the images"
        content = [
            Text(t, fontsize=14)
            for t in (
                "**Tests for:**",
                "- Data processing errors",
                "**Risks:**",
                "- Lack of robustness",
                "- Poor real-world performance",
                "- Poor generalization",
                "**Action:**",
                f"- {action_text}",
            )
        ]

        if box_stats:
            result_df = pd.DataFrame(
                {
                    "Category": [
                        "% of Total",
                        "# of Images",
                    ],
                    "Missing Bounding Boxes": [
                        self._to_pct_2(len(sanity_check["no_boxes"]), total_imgs)[0],
                        len(sanity_check["no_boxes"]),
                    ],
                    "Large % of 0 Values": [
                        self._to_pct_2(len(sanity_check["zeros"]), total_imgs)[0],
                        len(sanity_check["zeros"]),
                    ],
                    "Large % of Missing Values": [
                        self._to_pct_2(len(sanity_check["missing"]), total_imgs)[0],
                        len(sanity_check["missing"]),
                    ],
                    "Extreme Image Size": [
                        self._to_pct_2(len(sanity_check["size"]), total_imgs)[0],
                        len(sanity_check["size"]),
                    ],
                    "Extreme Image Aspect Ratio": [
                        self._to_pct_2(len(sanity_check["aspect_ratio"]), total_imgs)[0],
                        len(sanity_check["aspect_ratio"]),
                    ],
                }
            )

        else:
            result_df = pd.DataFrame(
                {
                    "Category": [
                        "% of Total",
                        "# of Images",
                    ],
                    "Large % of 0 Values": [
                        self._to_pct_2(len(sanity_check["zeros"]), total_imgs)[0],
                        len(sanity_check["zeros"]),
                    ],
                    "Large % of Missing Values": [
                        self._to_pct_2(len(sanity_check["missing"]), total_imgs)[0],
                        len(sanity_check["missing"]),
                    ],
                    "Extreme Image Size": [
                        self._to_pct_2(len(sanity_check["size"]), total_imgs)[0],
                        len(sanity_check["size"]),
                    ],
                    "Extreme Image Aspect Ratio": [
                        self._to_pct_2(len(sanity_check["aspect_ratio"]), total_imgs)[0],
                        len(sanity_check["aspect_ratio"]),
                    ],
                }
            )

        # Set up gradient slide
        outlier_slide = [
            {
                "deck": self._deck,
                "layout_name": "TextData",
                "layout_arguments": {
                    TextData.ArgKeys.TITLE.value: title,
                    TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                    TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                    TextData.ArgKeys.DATA_COLUMN_TABLE.value: result_df,
                },
            }
        ]

        all_categories = {}
        for metric in categories:
            all_categories[metric] = issues.get(metric, [])

        category_means: dict[str, float] = {}
        for k in all_categories:
            category_means[k] = float(np.mean(as_numpy(img_stats[k]).astype(float)))

        category_chunks = self._split_data_into_chunks(all_categories, 6)
        for chunk in category_chunks:
            df_data = self._create_category_dataframe_data(chunk, all_categories, category_means, total_imgs)
            outliers_df = pd.DataFrame(df_data)

            # Gradient slide kwargs
            title = f"Dataset: {self.dataset_id} | Category: Cleaning"
            heading = "Image Outliers\n"
            content = [
                Text(t, fontsize=14)
                for t in (
                    "**Result:**",
                    f"- Total Outliers: {self._to_pct_1(total_out, total_imgs)}",
                    "**Tests for:**",
                    "- Uncleaned data",
                    "**Risks:**",
                    "- Lack of robustness",
                    "- Poor real-world performance",
                    "- Poor generalization",
                    "**Action:**",
                    f"- {'No action required' if total_out == 0 else 'Evaluate outliers and clean data'}",
                )
            ]

            # Set up Gradient slide
            outlier_slide.append(
                {
                    "deck": self._deck,
                    "layout_name": "TextData",
                    "layout_arguments": {
                        TextData.ArgKeys.TITLE.value: title,
                        TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                        TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                        TextData.ArgKeys.DATA_COLUMN_TABLE.value: outliers_df,
                    },
                }
            )

        return outlier_slide

    def _target_outliers_with_images(
        self,
        sanity_check: dict[str, list[tuple[int, int | None, float]]],
        issues: dict[str, list[tuple[int, int | None, float]]],
        box_stats: dict[str, Any],
        total_sanity: int,
        total_targets: int,
        total_out: int,
    ) -> list[dict[str, Any]]:
        sanity_plot = self._plot_sanity_images("target", sanity_check, total_sanity > 0)

        # Gradient slide kwargs
        title = f"Dataset: {self.dataset_id} | Category: Cleaning"
        heading = "Basic Target Check\n"
        action_text = (
            "No action required" if total_sanity == 0 else "Evaluate Target boxes and remove or adjust as needed"
        )
        content = [
            Text(t, fontsize=14)
            for t in (
                "**Result:**",
                f"- Extreme % of 0 Values:    {len(sanity_check['zeros'])}",
                f"- Extreme % of Missing Values: {len(sanity_check['missing'])}",
                f"- Target:Image Extreme Sizes: {len(sanity_check['ratio_size'])}",
                f"- Extreme Target Sizes:     {len(sanity_check['size'])}",
                f"- Extreme Target Aspect Ratios: {len(sanity_check['aspect_ratio'])}",
                f"- Targets Outside of Image:   {len(sanity_check['ratio_top'])}",
                "**Tests for:**",
                "- Data processing errors",
                "**Action:**",
                f"- {action_text}",
            )
        ]

        # Set up gradient slide
        outlier_slide = [
            {
                "deck": self._deck,
                "layout_name": "TextData",
                "layout_arguments": {
                    TextData.ArgKeys.TITLE.value: title,
                    TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                    TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                    TextData.ArgKeys.DATA_COLUMN_IMAGE.value: sanity_plot,
                },
            }
        ]

        # Get the outliers
        remove = [k for k in issues if k.startswith("ratio_") or k == "distance"]
        for k in remove:
            issues.pop(k)
        issues = dict(take(8, sorted(issues.items(), key=lambda x: -len(x[1]))))

        issue_means: dict[str, float] = {}
        for k in issues:
            value = np.mean(box_stats[k])
            issue_means[k] = value
            issues[k] = sorted(issues[k], key=lambda x: -np.abs(x[2] - value))

        outliers_df = pd.DataFrame(
            {
                self._to_title(k): [
                    self._to_pct_2(len(v), total_targets)[1],
                    f"{MU} = {issue_means[k]:.2f}",
                ]
                for k, v in issues.items()
                for _ in range(2)
            }
        )

        for i in range(max(2 - len(outliers_df), 0)):
            outliers_df[" " * i] = ["", ""]

        images = self._plot_outlier_images("target", issues)

        image_kwargs = {
            IMAGE_ARGKEYS[i][j]: img_path
            for i, img_list in enumerate(images.values())
            for j, img_path in enumerate(img_list)
        }

        # Gradient slide kwargs
        title = f"Dataset: {self.dataset_id} | Category: Cleaning"
        heading = "Target Outliers\n"
        content = [
            Text(t, fontsize=14)
            for t in (
                "**Result:**",
                f"- Total Outliers: {self._to_pct_1(total_out, total_targets)}",
                "**Tests for:**",
                "- Uncleaned data",
                "**Risks:**",
                "- Lack of robustness",
                "- Poor real-world performance",
                "- Poor generalization",
                "**Action:**",
                f"- {'No action required' if total_out == 0 else 'Evaluate outliers and clean data'}",
                "",
                "",
                "Last row of table is the mean for the metric",
            )
        ]
        max_num_cols = len(outliers_df.columns)
        max_num_imgs = max([len(img_set) for img_set in images.values()] + [0])
        img_size = self._get_image_size(max_num_cols, max_num_imgs)

        # Set up Gradient slide
        outlier_slide.append(
            {
                "deck": self._deck,
                "layout_name": "TextTableImages",
                "layout_arguments": {
                    TextTableImages.ArgKeys.TITLE.value: title,
                    TextTableImages.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                    TextTableImages.ArgKeys.TEXT_COLUMN_BODY.value: content,
                    TextTableImages.ArgKeys.DATA_COLUMN_TABLE.value: outliers_df,
                    TextTableImages.ArgKeys.IMAGE_WIDTH.value: img_size,
                    TextTableImages.ArgKeys.IMAGE_HEIGHT.value: img_size,
                    TextTableImages.ArgKeys.IMAGE_Y_PADDING.value: 0.01,
                    **image_kwargs,
                },
            }
        )

        return outlier_slide

    def _generate_target_outliers_report(self, with_images: bool = False) -> list[dict[str, Any]]:
        """
        Generate the target outliers report which consists of two slides.
        The first slide for both images and targets is a basic data check slide.
        The second slide then displays the top (up to 8) outlier categories
        from `DimensionStatsOutput` and `VisualStatsOutput`.

        - Slide Format: `TextTableImages`
        - Text Content: Explanation of test, risks and action item
        - Table Content: Table of outliers identified for the analysis performed
        - Image Content: Sample images of identified outliers with labels
        """
        outliers: dict[int, dict[str, float]] = self.outputs["targetoutliers"]
        box_stats: dict[str, Any] = self.outputs["boxstats"]
        ratio_stats: dict[str, Any] = self.outputs["ratiostats"]

        # Format the issues for use
        source_index: list[tuple[int, int | None, int | None]] = box_stats["source_index"]
        categories = list(box_stats) + [f"ratio_{cat}" for cat in ratio_stats]
        issues: dict[str, list[tuple[int, int | None, float]]] = {}
        for i, issue in outliers.items():
            img_num, box_num, _ = source_index[int(i)]
            for k in (s for s in categories if s in issue):
                issues.setdefault(k, []).append((img_num, box_num, issue[k]))

        # Target Sanity check
        sanity_check: dict[str, list[tuple[int, int | None, float]]] = {}
        for metric in ["zeros", "missing", "ratio_size", "size", "aspect_ratio", "ratio_top"]:
            sanity_check[metric] = issues.get(metric, [])
        sanity_check["ratio_top"].extend(issues.pop("ratio_left", []))
        total_sanity = sum(len(sanity_check[metric]) for metric in sanity_check)
        total_out = len(outliers)
        total_targets = len(source_index)

        if with_images:
            return self._target_outliers_with_images(
                sanity_check, issues, box_stats, total_sanity, total_targets, total_out
            )

        # Gradient slide kwargs
        title = f"Dataset: {self.dataset_id} | Category: Cleaning"
        heading = "Basic Target Check\n"
        action_text = (
            "No action required" if total_sanity == 0 else "Evaluate Target boxes and remove or adjust as needed"
        )
        content = [
            Text(t, fontsize=14)
            for t in (
                "**Result:**",
                f"- Extreme % of 0 Values:    {len(sanity_check['zeros'])}",
                f"- Extreme % of Missing Values: {len(sanity_check['missing'])}",
                f"- Target:Image Extreme Sizes: {len(sanity_check['ratio_size'])}",
                f"- Extreme Target Sizes:     {len(sanity_check['size'])}",
                f"- Extreme Target Aspect Ratios: {len(sanity_check['aspect_ratio'])}",
                f"- Targets Outside of Image:   {len(sanity_check['ratio_top'])}",
                "**Tests for:**",
                "- Data processing errors",
                "**Risks:**",
                "- Lack of robustness",
                "- Poor real-world performance",
                "- Poor generalization",
                "**Action:**",
                f"- {action_text}",
            )
        ]

        result_df = pd.DataFrame(
            {
                "Category": [
                    "% of Total",
                    "# of Images",
                ],
                "Large % of 0 Values": [
                    self._to_pct_2(len(sanity_check["zeros"]), total_targets)[0],
                    len(sanity_check["zeros"]),
                ],
                "Large % of Missing Values": [
                    self._to_pct_2(len(sanity_check["missing"]), total_targets)[0],
                    len(sanity_check["missing"]),
                ],
                "Target:Image Extreme Size": [
                    self._to_pct_2(len(sanity_check["ratio_size"]), total_targets)[0],
                    len(sanity_check["ratio_size"]),
                ],
                "Extreme Image Size": [
                    self._to_pct_2(len(sanity_check["size"]), total_targets)[0],
                    len(sanity_check["size"]),
                ],
                "Extreme Image Aspect Ratio": [
                    self._to_pct_2(len(sanity_check["aspect_ratio"]), total_targets)[0],
                    len(sanity_check["aspect_ratio"]),
                ],
                "Targets Outside of Image": [
                    self._to_pct_2(len(sanity_check["ratio_top"]), total_targets)[0],
                    len(sanity_check["ratio_top"]),
                ],
            }
        )

        # Set up gradient slide
        outlier_slide = [
            {
                "deck": self._deck,
                "layout_name": "TextData",
                "layout_arguments": {
                    TextData.ArgKeys.TITLE.value: title,
                    TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                    TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                    TextData.ArgKeys.DATA_COLUMN_TABLE.value: result_df,
                },
            }
        ]

        all_categories = {}
        for metric in categories:
            if metric not in ["distance"]:
                all_categories[metric] = issues.get(metric, [])

        category_means: dict[str, float] = {}
        for k in all_categories:
            category_means[k] = (
                float(np.mean(as_numpy(ratio_stats[k[6:]]).astype(float)))
                if k.startswith("ratio_")
                else float(np.mean(as_numpy(box_stats[k]).astype(float)))
            )

        category_chunks = self._split_data_into_chunks(all_categories, 6)
        for chunk in category_chunks:
            df_data = self._create_category_dataframe_data(chunk, all_categories, category_means, total_targets)
            outliers_df = pd.DataFrame(df_data)

            # Gradient slide kwargs
            title = f"Dataset: {self.dataset_id} | Category: Cleaning"
            heading = "Target Outliers\n"
            content = [
                Text(t, fontsize=14)
                for t in (
                    "**Result:**",
                    f"- Total Outliers: {self._to_pct_1(total_out, total_targets)}",
                    "**Tests for:**",
                    "- Uncleaned data",
                    "**Risks:**",
                    "- Lack of robustness",
                    "- Poor real-world performance",
                    "- Poor generalization",
                    "**Action:**",
                    f"- {'No action required' if total_out == 0 else 'Evaluate outliers and clean data'}",
                )
            ]

            # Set up Gradient slide
            outlier_slide.append(
                {
                    "deck": self._deck,
                    "layout_name": "TextData",
                    "layout_arguments": {
                        TextData.ArgKeys.TITLE.value: title,
                        TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                        TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                        TextData.ArgKeys.DATA_COLUMN_TABLE.value: outliers_df,
                    },
                }
            )

        return outlier_slide

    def _generate_next_steps_report(self) -> dict[str, Any]:
        """
        Generate conclustion slide.

        - Slide Format: `TextData`
        - Text Content: Next setps
        - Image Content: Blank Image
        """
        # Create blank image for display
        self.cache_contents_path.mkdir(parents=True, exist_ok=True)
        filepath = Path(self.cache_contents_path, "blank_img.png")
        self._plot_blank_or_single_image(False, filepath, [])

        # Gradient slide kwargs
        title = f"Dataset: {self.dataset_id} | Category: Cleaning"
        heading = "Next Steps\n"
        content = [
            Text(t, fontsize=14)
            for t in (
                "- Remove the images/targets flagged in the Basic Check reports",
                "- Manually review the images/targets flagged in the Outlier reports",
                "",
                "For images:",
                "- Check if images come up in multiple outlier categories. If so, remove.",
                "- Make sure images are representative of their respective environment/class. If not, remove.",
                "",
                "For targets:",
                "- Need to run bias, with bounding box stats to make sure there"
                " are no correlations between a statistic and a class",
                "- Make sure targets are representative of their respective class. If not, remove.",
            )
        ]

        # Set up Gradient slide
        return {
            "deck": self._deck,
            "layout_name": "TextData",
            "layout_arguments": {
                TextData.ArgKeys.TITLE.value: title,
                TextData.ArgKeys.TEXT_COLUMN_HEADING.value: heading,
                TextData.ArgKeys.TEXT_COLUMN_BODY.value: content,
                TextData.ArgKeys.TEXT_COLUMN_HALF.value: True,
                TextData.ArgKeys.DATA_COLUMN_IMAGE.value: filepath,
            },
        }

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Generate reports for duplicates and outliers and include if not empty"""
        duplicates = self._generate_duplicates_report()
        stat_list = self._generate_stats_report()
        image_list = self._generate_image_outliers_report()
        if "boxstats" in self.outputs:
            target_list = self._generate_target_outliers_report()

            ordered_list: list[dict[str, Any]] = [
                stat_list[0],
                duplicates,
                image_list[0],
                stat_list[1],
                image_list[1],
                image_list[2],
                target_list[0],
                stat_list[2],
                target_list[1],
                target_list[2],
                target_list[3],
                self._generate_next_steps_report(),
            ]

            return ordered_list

        ordered_list: list[dict[str, Any]] = [
            stat_list[0],
            duplicates,
            image_list[0],
            stat_list[1],
            image_list[1],
            image_list[2],
            self._generate_next_steps_report(),
        ]
        return ordered_list
