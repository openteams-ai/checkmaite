import maite.protocols.object_detection as od
from dataeval.core import compute_ratios, compute_stats
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates

from checkmaite.core._common.dataeval_cleaning_capability import (
    DataevalCleaningBase,
    DataevalCleaningConfig,
    DataevalCleaningOutputs,
    _normalize_duplicates_output,
)


class DataevalCleaning(DataevalCleaningBase[od.Dataset, od.Model, od.Metric]):
    "Object detection cleaning capability"

    def _run(
        self,
        models: list[od.Model],  # noqa: ARG002
        datasets: list[od.Dataset],
        metrics: list[od.Metric],  # noqa: ARG002
        config: DataevalCleaningConfig,  # noqa: ARG002
        use_prediction_and_evaluation_cache: bool,  # noqa: ARG002
    ) -> DataevalCleaningOutputs:
        """Execute the full statistics and outlier detection pipeline for the dataset.

        Returns
        -------
        CleaningOutputs
            The outputs of the cleaning process.
        """

        dataset = datasets[0]

        stats, label_stats_result = self._run_basic_stats(dataset=dataset)

        duplicates = Duplicates().from_stats(stats)
        image_outliers = self._compute_basic_outliers(stats=stats)

        # boxes stats
        boxes_stats_output = compute_stats(
            dataset,
            stats=(ImageStats.DIMENSION | ImageStats.VISUAL | ImageStats.PIXEL_ZEROS | ImageStats.PIXEL_MISSING),
            per_image=False,
            per_target=True,
        )
        dimensional_ratio_stats = compute_ratios(stats, target_stats_output=boxes_stats_output)
        box_outliers = self._compute_box_outliers(boxes_stats_output, ratiostats=dimensional_ratio_stats)

        image_stats = self._get_img_stats(stats)
        box_stats = self._get_box_stats(boxes_stats_output, dimensional_ratio_stats)
        label_stats_result = self._convert_label_stats(label_stats_result)

        return DataevalCleaningOutputs.model_validate(
            {
                "duplicates": _normalize_duplicates_output(duplicates),
                "image_outliers": image_outliers,
                "image_stats": image_stats,
                "label_stats": label_stats_result,
                "box_outliers": box_outliers,
                "box_stats": box_stats,
            }
        )
