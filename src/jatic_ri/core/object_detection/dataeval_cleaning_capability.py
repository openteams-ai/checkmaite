import maite.protocols.object_detection as od
from dataeval.detectors.linters import Duplicates
from dataeval.metrics.stats import boxratiostats, dimensionstats, visualstats

from jatic_ri.core._common.dataeval_cleaning_capability import (
    DataevalCleaningBase,
    DataevalCleaningConfig,
    DataevalCleaningOutputs,
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

        hashes, img_dim_stats, img_viz_stats, label_stats = self._run_basic_stats(dataset=dataset)

        duplicates = Duplicates().from_stats(hashes)

        img_outliers = self._compute_basic_outliers(dim_stats=img_dim_stats, viz_stats=img_viz_stats)

        incremented_dataset = dataset

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

        return DataevalCleaningOutputs.model_validate(
            {
                "duplicates": duplicates.data(),
                "img_outliers": img_outliers,
                "img_dim_stats": img_dim_stats.data(),
                "img_viz_stats": img_viz_stats.data(),
                "label_stats": label_stats.data(),
                "target_outliers": target_outliers,
                "box_dim_stats": box_dim_stats.data(),
                "box_viz_stats": box_viz_stats.data(),
                "box_ratio_stats": dimensional_ratio_stats.data(),
            }
        )
