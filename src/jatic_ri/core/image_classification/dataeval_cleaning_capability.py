import maite.protocols.image_classification as ic
from dataeval.detectors.linters import Duplicates

from jatic_ri.core._common.dataeval_cleaning_capability import (
    DataevalCleaningBase,
    DataevalCleaningConfig,
    DataevalCleaningOutputs,
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

        hashes, img_dim_stats, img_viz_stats, label_stats = self._run_basic_stats(dataset=dataset)

        duplicates = Duplicates().from_stats(hashes)

        img_outliers = self._compute_basic_outliers(dim_stats=img_dim_stats, viz_stats=img_viz_stats)

        return DataevalCleaningOutputs.model_validate(
            {
                "duplicates": duplicates.data(),
                "img_outliers": img_outliers,
                "img_dim_stats": img_dim_stats.data(),
                "img_viz_stats": img_viz_stats.data(),
                "label_stats": label_stats.data(),
                "target_outliers": None,
                "box_dim_stats": None,
                "box_viz_stats": None,
                "box_ratio_stats": None,
            }
        )
