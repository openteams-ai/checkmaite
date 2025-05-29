from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dataeval.detectors.linters import Duplicates, Outliers
from dataeval.metrics.stats import (
    DimensionStatsOutput,
    HashStatsOutput,
    LabelStatsOutput,
    VisualStatsOutput,
    boxratiostats,
    dimensionstats,
    hashstats,
    labelstats,
    visualstats,
)
from dataeval.outputs import SourceIndex
from gradient import SubText
from gradient.slide_deck.shapes import Text
from gradient.slide_deck.shapes.image_shapes import GradientImage
from gradient.templates_and_layouts.generic_layouts import SectionByItem

from jatic_ri import cache_path
from jatic_ri._common.test_stages.impls._dataeval_utils import (
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
from jatic_ri._common.test_stages.interfaces.plugins import SingleDatasetPlugin, TDataset
from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, OutputsBase, RunBase, TestStage
from jatic_ri.util.slide_deck import (
    create_item_by_narrow_text_slide,
    create_section_by_item_slide_extra_caption,
    create_table_text_slide,
    create_two_item_text_slide,
)


class DataevalCleaningConfig(ConfigBase):
    "Configuration options for the DataevalCleaningTestStage can be specified here."


class DataevalCleaningDuplicatesOutputs(OutputsBase):
    exact: Sequence[Sequence[int]]
    near: Sequence[Sequence[int]]


class DataevalCleaningDimensionStatsOutputs(OutputsBase):
    source_index: Sequence[SourceIndex]
    object_count: np.ndarray
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


class DataevalCleaningVisualStatsOutputs(OutputsBase):
    source_index: Sequence[SourceIndex]
    object_count: np.ndarray
    image_count: int
    brightness: np.ndarray
    contrast: np.ndarray
    darkness: np.ndarray
    missing: np.ndarray
    sharpness: np.ndarray
    zeros: np.ndarray
    percentiles: np.ndarray


class DataevalCleaningLabelStatsOutputs(OutputsBase):
    label_counts_per_class: Mapping[int, int]
    label_counts_per_image: Sequence[int]
    image_counts_per_class: Mapping[int, int]
    image_indices_per_class: Mapping[int, Sequence[int]]
    image_count: int
    class_count: int
    label_count: int
    class_names: Sequence[str]


class DataevalCleaningOutputs(OutputsBase):
    duplicates: DataevalCleaningDuplicatesOutputs
    img_outliers: dict[int, dict[str, float]]
    img_dim_stats: DataevalCleaningDimensionStatsOutputs
    img_viz_stats: DataevalCleaningVisualStatsOutputs
    label_stats: DataevalCleaningLabelStatsOutputs
    target_outliers: dict[int, dict[str, float]] | None
    box_dim_stats: DataevalCleaningDimensionStatsOutputs | None
    box_viz_stats: DataevalCleaningVisualStatsOutputs | None
    box_ratio_stats: DataevalCleaningDimensionStatsOutputs | None


class DataevalCleaningRun(RunBase):
    config: DataevalCleaningConfig
    outputs: DataevalCleaningOutputs


class DatasetCleaningTestStageBase(TestStage[DataevalCleaningOutputs], SingleDatasetPlugin[TDataset]):
    """
    Dataset Cleaning TestStage Base implementation.

    Performs dataset cleaning by identifying duplicates (exact and near) as well as statistical outliers
    using various pixel and image statistics on the dataset.
    """

    _RUN_TYPE = DataevalCleaningRun

    def _create_config(self) -> ConfigBase:
        return DataevalCleaningConfig()

    def _run_basic_stats(
        self,
    ) -> tuple[
        HashStatsOutput,
        DimensionStatsOutput,
        VisualStatsOutput,
        LabelStatsOutput,
    ]:
        "Compute statistics for the images in the dataset."

        hashes = hashstats(self.dataset)

        img_dim_stats = dimensionstats(self.dataset)
        img_viz_stats = visualstats(self.dataset)

        if self._task == "od" or self._task == "ic":
            label_stats = labelstats(self.dataset)
        else:
            raise ValueError(f"Test Stage task must be one of 'ic' or 'od', current value of task: {self._task}.")

        return hashes, img_dim_stats, img_viz_stats, label_stats

    def _compute_basic_outliers(
        self, dim_stats: DimensionStatsOutput | None = None, viz_stats: VisualStatsOutput | None = None
    ) -> dict[int, dict[str, float]]:
        """
        Compute z-score-based outliers for selected dimension and visual metrics.

        This method applies a z-score threshold of 3 to identify outliers in the
        dataset's statistics. It filters the results to include only categories
        defined in `DIMENSION_LIST` or `VISUAL_LIST`.
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
        Identifies outliers in a given metric, enforcing an upper threshold of 1.0
        for ratio-based metrics when necessary. (A ratio of bounding-box to image should
        never exceed one as the bounding-box should always be contained inside the image.)

        This function perfoms a simple check to see if the value is greater than 1.0. If so,
        it flags the value as an outlier.
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
        "Merges two outlier result dictionaries together"
        for key, inner in dict2.items():
            if key in dict1:
                dict1[key].update(inner)
            else:
                dict1[key] = inner
        return dict1

    def _compute_ratio_outliers(self, ratio_stats: DimensionStatsOutput) -> dict[int, dict[str, float]]:
        """
        Compute z-score-based outliers for selected ratio metrics.

        This method applies a z-score threshold of 3 to identify outliers in the
        dataset's statistics. It filters the results to include only categories
        defined in `RATIO_LIST`.
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
        Computes outliers related to bounding boxes.

        This includes standard visual/dimensional outliers from bounding box stats,
        as well as adjusted outliers for ratio metrics.
        """
        box_result = self._compute_basic_outliers(dim_stats=box_dim_stats, viz_stats=box_viz_stats)
        ratio_result = self._compute_ratio_outliers(ratio_stats=ratiostats)

        ratio_categories = ["offset_x", "offset_y", "width", "height", "size"]
        adjusted_ratio_result = self._outlier_at_1(ratio_result, ratiostats, ratio_categories)

        return self._dictionary_merge(box_result, adjusted_ratio_result)

    def _run(self) -> DataevalCleaningOutputs:
        "Executes the full statistics and outlier detection pipeline for the dataset."

        hashes, img_dim_stats, img_viz_stats, label_stats = self._run_basic_stats()

        duplicates = Duplicates().from_stats(hashes)

        img_outliers = self._compute_basic_outliers(dim_stats=img_dim_stats, viz_stats=img_viz_stats)

        if self._task == "od":
            incremented_dataset = self.dataset

            box_dim_stats = dimensionstats(
                dataset=incremented_dataset,
                per_box=True,
            )
            box_viz_stats = visualstats(
                dataset=incremented_dataset,
                per_box=True,
            )

            dimensional_ratio_stats = boxratiostats(imgstats=img_dim_stats, boxstats=box_dim_stats)

            target_outliers = self._compute_box_outliers(
                box_dim_stats=box_dim_stats, box_viz_stats=box_viz_stats, ratiostats=dimensional_ratio_stats
            )

        elif self._task == "ic":
            target_outliers = None
            box_dim_stats = None
            box_viz_stats = None
            dimensional_ratio_stats = None

        else:
            raise ValueError(f"Test Stage task must be one of 'ic' or 'od', current value of task: {self._task}.")

        return DataevalCleaningOutputs.model_validate(
            {
                "duplicates": duplicates.data(),
                "img_outliers": img_outliers,
                "img_dim_stats": img_dim_stats.data(),
                "img_viz_stats": img_viz_stats.data(),
                "label_stats": label_stats.data(),
                "target_outliers": target_outliers,
                "box_dim_stats": box_dim_stats.data() if box_dim_stats is not None else None,
                "box_viz_stats": box_viz_stats.data() if box_viz_stats is not None else None,
                "box_ratio_stats": dimensional_ratio_stats.data() if dimensional_ratio_stats is not None else None,
            }
        )

    def add_slide(
        self,
        deck: str,
        title: str,
        text: Text,
        metrics_subset: list[str],
        all_metrics: dict[str, list],
        total: int,
        is_images: bool = True,
    ) -> dict[str, Any]:
        metric_df = create_metric_dataframe_data(
            is_images=is_images, metrics_subset=metrics_subset, all_metrics=all_metrics, total=total
        )
        return create_table_text_slide(deck=deck, title=title, text=text, data=metric_df)

    def _generate_table_of_contents(self) -> dict[str, Any]:
        "Generates a table of contents for the report."

        right_item = [
            "\n",
            "* Image Duplicate Analysis",
            "* Image Property Histograms",
            Text("Used for adjusting the outlier analysis thresholds.", indent=1),
            "* Image Outlier Analysis",
            "* Label Analysis",
            "* Target Property Histograms",
            Text("Used for adjusting the outlier analysis thresholds.", indent=1),
            "* Target Outlier Analysis",
            "* Next Steps",
        ]

        left_item = GradientImage(
            src=Path("src/jatic_ri/_sample_imgs/toc.png"), width=100, height=100, top=0.5, left=0.5
        )
        return create_two_item_text_slide(
            deck=self._deck, title="Table of Contents", left_item=left_item, right_item=right_item
        )

    def _generate_duplicates_report(
        self, duplicates: DataevalCleaningDuplicatesOutputs, dataset_size: int
    ) -> dict[str, Any]:
        "Generates a report for image duplicates."
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

        content = Text(
            [
                SubText("Description: ", bold=True),
                SubText("Identify images which are identical or almost identical.\n"),
            ],
            fontsize=22,
        )

        return create_table_text_slide(deck=self._deck, title=title, text=content, data=duplicates_df)

    def _generate_stats_report(
        self,
        img_stats: tuple[DataevalCleaningDimensionStatsOutputs, DataevalCleaningVisualStatsOutputs],
        label_stats: DataevalCleaningLabelStatsOutputs,
        box_stats: tuple[DataevalCleaningDimensionStatsOutputs, DataevalCleaningVisualStatsOutputs] | None,
        ratio_stats: DataevalCleaningDimensionStatsOutputs | None,
    ) -> list[dict[str, Any]]:
        "Generates a report for image and target statistics."

        stat_slides = []

        content = [
            Text("Description: ", bold=True, fontsize=22),
            Text(
                "Visual overview of potential outliers in image properties. Vertical lines are the outlier thresholds"
                "(computed internally). Values outside of the vertical lines will be flagged as outliers.",
                fontsize=22,
            ),
        ]

        # build gradient slide for image outlier histograms
        img_hist_list = prepare_histograms(img_stats)
        dir_ = Path(cache_path() / "cleaning-test-stage-artifacts")
        dir_.mkdir(parents=True, exist_ok=True)
        title = "Image Property Histograms"
        filepath = dir_ / "img_stats_histogram_plots.png"
        plot_stat_metrics(is_image=True, plot_list=img_hist_list, filepath=filepath)
        stat_slides.append(
            create_item_by_narrow_text_slide(deck=self._deck, title=title, content=content, body_value=filepath)
        )

        # build gradient slide for label analysis
        result_content, label_df = label_table(label_stats, index2label=self.dataset.metadata["index2label"])  # type: ignore
        title = "Label Analysis"
        content = []
        content.append(Text("Description: ", bold=True, fontsize=22))
        content.append(Text("Numerical analysis of label properties.\n\n", fontsize=22))
        for t in result_content:
            content.append(Text(t, fontsize=16))
        stat_slides.append(
            create_section_by_item_slide_extra_caption(
                deck=self._deck, title=title, heading=Text(" "), content=content, body_value=label_df
            )
        )

        content = [
            Text("Description: ", bold=True, fontsize=22),
            Text(
                "Visual overview of potential outliers in target properties. Vertical lines are the outlier thresholds"
                " (computed internally). Values outside of the vertical lines will be flagged as outliers.",
                fontsize=22,
            ),
        ]

        if box_stats and ratio_stats:
            box_hist_list = prepare_histograms(box_stats)
            box_hist_list = prepare_ratio_histograms(ratio_stats, box_hist_list)
            dir_ = Path(cache_path() / "cleaning-test-stage-artifacts")
            dir_.mkdir(parents=True, exist_ok=True)
            filepath = dir_ / "box_stats_histogram_plots.png"
            plot_stat_metrics(is_image=False, plot_list=box_hist_list, filepath=filepath)
            title = "Target Property Histograms"
            stat_slides.append(
                create_item_by_narrow_text_slide(deck=self._deck, title=title, content=content, body_value=filepath)
            )

        return stat_slides

    def _generate_image_outliers_report(
        self,
        img_outliers: dict[int, dict[str, float]],
        img_stats: tuple[DataevalCleaningDimensionStatsOutputs, DataevalCleaningVisualStatsOutputs],
        dataset_size: int,
    ) -> list[dict[str, Any]]:
        "Generates a report for image outliers."
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
        all_metrics = {
            k: issues.get(k, []) for k in metrics if k not in ["channels", "distance_center", "distance_edge"]
        }
        # looks better if we limit to 4 entries per slide...
        metric_chunks = split_into_chunks(all_metrics, chunk_sizes=[4])
        captions = ["Dimensional", "Visual", "Pixel"]
        for idx, chunk in enumerate(metric_chunks):
            title = f"Image {captions[idx]} Outliers"
            text = Text(
                [
                    SubText("Description: ", bold=True),
                    f" Numerical analysis of {captions[idx].lower()} outliers in images.",
                ],
                fontsize=21,
            )
            outlier_slides.append(
                self.add_slide(
                    deck=self._deck,
                    title=title,
                    text=text,
                    metrics_subset=chunk,
                    all_metrics=all_metrics,
                    total=dataset_size,
                    is_images=True,
                )
            )

        return outlier_slides

    def _generate_target_outliers_report(
        self,
        target_outliers: dict[int, dict[str, float]] | None,
        box_stats: tuple[DataevalCleaningDimensionStatsOutputs, DataevalCleaningVisualStatsOutputs],
        total_targets: int,
    ) -> list[dict[str, Any]]:
        "Generates a report for target outliers."

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
            text = Text(
                [
                    SubText("Description: ", bold=True),
                    f" Numerical analysis of {captions[idx].lower()} outliers in targets.",
                ],
                fontsize=21,
            )
            outlier_slides.append(
                self.add_slide(
                    deck=self._deck,
                    title=title,
                    text=text,
                    metrics_subset=chunk,
                    all_metrics=all_metrics,
                    total=total_targets,
                    is_images=False,
                )
            )

        return outlier_slides

    def _generate_next_steps_report(self) -> dict[str, Any]:
        "Generates a report for the next steps to investigating issues that may arise during analysis."

        dir_ = Path(cache_path() / "cleaning-test-stage-artifacts")
        dir_.mkdir(parents=True, exist_ok=True)
        filepath = dir_ / "blank_img.png"
        plot_blank_or_single_image(filepath)

        title = f"Dataset: {self.dataset_id} | Category: Cleaning"
        heading = "Next Steps\n"
        content = [
            Text(t, fontsize=14)
            for t in (
                "Below are the recommended next steps to investigating issues that may arise during analysis.",
                [SubText("In general:", bold=True)],
                "- Remove the images/targets flagged in the Basic Check reports for images and targets",
                "- Manually review the images/targets flagged in the Outlier reports",
                [SubText("For images:", bold=True)],
                "- Check if images come up in multiple outlier categories. If so, remove.",
                "- Make sure images are representative of their respective environment/class. If not, remove.",
                [SubText("For targets:", bold=True)],
                "- Run bias analysis with bounding box stats and ensure there are no correlations between a statistic and a class",  # noqa: E501
                "- Make sure targets are representative of their respective class. If not, remove.",
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

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        "Collects reports for duplicates and outliers for both image and target data."

        table_of_contents = self._generate_table_of_contents()

        duplicates = self._generate_duplicates_report(
            duplicates=self.outputs.duplicates, dataset_size=self.outputs.label_stats.image_count
        )
        stat_list = self._generate_stats_report(
            img_stats=(self.outputs.img_dim_stats, self.outputs.img_viz_stats),
            label_stats=self.outputs.label_stats,
            box_stats=(self.outputs.box_dim_stats, self.outputs.box_viz_stats)
            if self.outputs.box_dim_stats and self.outputs.box_viz_stats
            else None,
            ratio_stats=self.outputs.box_ratio_stats,
        )
        image_list = self._generate_image_outliers_report(
            img_outliers=self.outputs.img_outliers,
            img_stats=(self.outputs.img_dim_stats, self.outputs.img_viz_stats),
            dataset_size=self.outputs.label_stats.image_count,
        )

        if self.outputs.box_dim_stats and self.outputs.box_viz_stats:
            target_list = self._generate_target_outliers_report(
                target_outliers=self.outputs.target_outliers,
                box_stats=(self.outputs.box_dim_stats, self.outputs.box_viz_stats),
                total_targets=self.outputs.label_stats.label_count,
            )

            return [
                table_of_contents,
                duplicates,
                stat_list[0],
                *image_list[0:],
                *stat_list[1:],
                *target_list[0:],
                self._generate_next_steps_report(),
            ]

        return [
            table_of_contents,
            duplicates,
            stat_list[0],
            *image_list[0:],
            *stat_list[1:],
            self._generate_next_steps_report(),
        ]
