import maite.protocols.image_classification as ic
from dataeval.quality import Duplicates

from checkmaite.core._common.dataeval_cleaning_capability import (
    DataevalCleaningBase,
    DataevalCleaningConfig,
    DataevalCleaningOutputs,
    _normalize_duplicates_output,
)


class DataevalCleaning(DataevalCleaningBase[ic.Dataset, ic.Model, ic.Metric]):
    "Image classification cleaning capability"

    def _run(
        self,
        models: list[ic.Model],  # noqa: ARG002
        datasets: list[ic.Dataset],
        metrics: list[ic.Metric],  # noqa: ARG002
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

        image_stats = self._get_img_stats(stats)
        label_stats_result = self._convert_label_stats(label_stats_result)

        return DataevalCleaningOutputs.model_validate(
            {
                "duplicates": _normalize_duplicates_output(duplicates),
                "image_outliers": image_outliers,
                "image_stats": image_stats,
                "label_stats": label_stats_result,
                "box_outliers": None,
                "box_stats": None,
            }
        )
