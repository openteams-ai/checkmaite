"""File for Image Classification Survivor test stage."""

import copy
import json
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional, Union

import maite.protocols.image_classification as ic
import numpy as np
import pyspark.sql.functions as sf
from gradient import Text
from maite import protocols as pr
from pyspark.sql import DataFrame, SparkSession
from survivor.analysis import HistogramBarPlot
from survivor.config import Config as SurvivorConfig
from survivor.maite_survivor import MAITESurvivor

from jatic_ri import cache_path
from jatic_ri._common.test_stages.impls.survivor_test_stage_cache import SurvivorCache
from jatic_ri._common.test_stages.interfaces.plugins import (
    EvalToolPlugin,
    MetricPlugin,
    MultiModelPlugin,
    SingleDatasetPlugin,
)
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage

# This constant represents the expected location of the Survivor output directory under the cache_path()
# directory where the SurvivorTestStage.run() function can store visualizations in the event of a cache miss. Since
# these results may be needed past the lifetime of the TestStage, these results will just be left until deleted by the
# user. NOTE: There should only ever be one file in here at a time since the image file will just be overwritten by
# the next cache miss.
_SURVIVOR_CACHE_MISS_OUTPUT_DIR = "survivor_cache_miss_outputs"

# The key that must be present in a MAITE evaluate()'s returned dictionary. This should correspond to another
# dictionary with MetricPlugin.metric_id as a key. This list will then be a datum-by-datum specification of the
# Metric values.
PER_DATUM_METRIC_KEY = "per_datum"


class SurvivorTestStage(
    TestStage[tuple[DataFrame, Path]],
    MultiModelPlugin[ic.Model],
    SingleDatasetPlugin[ic.Dataset],
    MetricPlugin[ic.Metric],
    EvalToolPlugin,
):
    """Survivor Test Stage Object.

    Survivor uses an ensemble of models and metrics based on model inference results, to provide insight into
    how difficult a set of image may be for models. Generally speaking, "Easy" images are those that most models
    perform well on, "Hard" images are those that most models perform poorly on, and "On the Bubble" images are those
    that models have a wide variety of performance ranges on.

    For more info, see our docs! https://jatic.pages.jatic.net/morse/survivor/

    This test stage also uses MAITE-wrapped models, datasets, and metrics, and MAITE itself, to produce the model
    metric results needed if they are not present in the cache before running Survivor itself.

    Attributes:
        config: The Survivor Config object that should be used when running Survivor.
        cache: The SurvivorCache object used to read from and write to cache locations.
        outputs: A tuple of Survivor results with the layout:
            [0]: The SurvivorResults.raw_output_df dataframe.
            [1]: A Path to a PNG with a histogram of the number of images per Survivor category: Easy, Hard, and
                On the Bubble.
        metric: The MAITE-wrapped metric object that should be fed the model inference results
            for metric calculation.
        dataset: The MAITE-wrapped dataset object on which the models should run inference and
            produce results.
        models: The dictionary of model names to their MAITE-wrapped model objects
            whose inference should be used when running Survivor.
    """

    _deck: str = "image_classification_survivor"
    _task: str = "ic"

    def __init__(
        self,
        config: Union[SurvivorConfig, dict[str, Any]],
    ) -> None:
        """Create instance of SurvivorTestStage

        Args:
            config: config for survivor run.
        """
        self.config: SurvivorConfig = SurvivorConfig(**config) if isinstance(config, dict) else config
        self.cache = SurvivorCache()

        # A dictionary of identifying information that will be hashed into an ID
        self._cache_configuration: Optional[dict[str, Any]] = None

        super().__init__()

    def _generate_cache_config(self) -> None:
        """Examines the Survivor config, the MAITE-models, and MAITE-dataset to update self._cache_configuration.

        Also updates the cache configuration of this instance's SurvivorCache().
        """
        self.validate_plugins()
        sorted_model_ids = list(self.models.keys())
        sorted_model_ids.sort()
        survivor_config = copy.deepcopy(self.config.__dict__)
        # column_names is a ColumnNameConfig which also needs to be unpacked.
        survivor_config["column_names"] = survivor_config["column_names"].__dict__

        self._cache_configuration = {
            "model_ids": sorted_model_ids,
            "dataset_id": self.dataset_id,
            "survivor_config": survivor_config,
        }
        self.cache: Optional[Cache[tuple[DataFrame, Path]]] = SurvivorCache(self._cache_configuration)

    @property
    def cache_id(self) -> str:
        """Generate the cache ID for based on the Model IDs, Dataset ID, Metric ID, and Survivor Config."""
        self._generate_cache_config()
        config_hash_string = sha256(json.dumps(self._cache_configuration).encode("utf-8")).hexdigest()

        return f"survivor_{self._task}_cache_{config_hash_string}"

    def __run_metrics(self) -> dict[str, pr.ArrayLike]:
        """Create metrics by model for use in MAITESurvivor."""
        all_model_metrics_per_datum = {}
        for model in self.models:
            model_metrics_per_datum = []
            # Since Survivor's implementation is unique in that it runs metric calculations on each individual in
            # a dataset, we first call the `predict()` method using the model against the entire dataset.
            # Otherwise, we could not leverage a cached result from a prior test stage's prediction generation.
            predictions, targets = self.eval_tool.predict(
                model=self.models[model],
                model_id=model,
                dataset=self.dataset,
                dataset_id=self.dataset_id,
                batch_size=self._batch_size,
            )
            # With the predictions for each target now in memory, we run only `compute_metric()` against each item.
            # Calling `evaluate()` to calculate the metrics would require doing model predictions again with each
            # target as its own dataset.
            for i, datum_prediction in enumerate(predictions):
                results: dict[str, Any] = self.eval_tool.compute_metric(
                    metric=self.metric,
                    filename=f"{model}_{self.dataset_id}-img{i}_{self.metric_id}_{self._batch_size}.json",
                    prediction=[datum_prediction],
                    data=[targets[i]],
                )
                model_metrics_per_datum.append(results[self.metric_id].numpy())

                # Reset the metric for the next evaluate() call so results don't bleed together.
                self.metric.reset()

            # Take the aggregated datum results from this model and turn it into a 1-d colum numpy array for
            # later consumption.
            all_model_metrics_per_datum[model] = np.hstack(model_metrics_per_datum)

        return all_model_metrics_per_datum

    def _run(self) -> tuple[DataFrame, Path]:
        """Run Survivor if no cached results are found."""
        self.validate_plugins()

        # run metrics
        metrics = self.__run_metrics()

        # create maite wrapper and run Survivor
        survivor = MAITESurvivor(
            maite_dataset=self.dataset,
            config=self.config,
            metrics=metrics,
            spark_session=SparkSession.builder.getOrCreate(),  # type: ignore
        )
        results_df = survivor.run().raw_output_df

        # Clear out the cache miss dir in preparation for our new results.
        cache_miss_output_output_dir = cache_path() / _SURVIVOR_CACHE_MISS_OUTPUT_DIR
        if cache_miss_output_output_dir.exists():
            shutil.rmtree(cache_miss_output_output_dir)
        cache_miss_output_output_dir.mkdir(parents=True)

        # generate histogram plot output.
        histogram_plot = HistogramBarPlot(
            title="Image count by type",
            col="suid_label",
            output_dir=cache_miss_output_output_dir,
        )

        # The returned name will be a bit variable since there's multiple histograms that are made,
        # we only need one, so we'll just standardize it.
        _, cache_miss_output_img_path = histogram_plot.plot(results_df)[0]
        cache_miss_output_img_path = cache_miss_output_img_path.rename(
            cache_miss_output_img_path.parent / self.cache.cache_image_path,  # type: ignore
        )

        return results_df, cache_miss_output_img_path

    def collect_metrics(self) -> dict[str, float]:
        """Collect metrics for the SurvivorTestStage."""
        survivor_results, _ = self.outputs

        total = survivor_results.count()
        otb_proportion = survivor_results.where(sf.col("suid_label") == "On the Bubble").count() / total

        # This is sort of assuming we're only allowed a single float as our return type here as is
        # sort of implied by the documents provided
        return {
            "Low_Val_Data": 1.0 - otb_proportion,
        }

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect report consumables for the SurvivorTestStage."""
        survivor_results, output_image_path = self.outputs

        total = survivor_results.count()
        hard_proportion = survivor_results.where(sf.col("suid_label") == "Hard").count() / total
        easy_proportion = survivor_results.where(sf.col("suid_label") == "Easy").count() / total
        otb_proportion = survivor_results.where(sf.col("suid_label") == "On the Bubble").count() / total

        # Simply return the title of the section and the data to be plotted
        definition_text = Text(
            content=f"**Types of Data**\n"
            f"• Easy: Models score the same and perform well.\n"
            f"• Hard: Models score the same and perform poorly.\n"
            f"• On the Bubble: Models score differently.\n\n"
            f"• Ideally, a dataset would be primarily On the Bubble, so all data is helping distinguish between model "
            f"performance.\n\n"
            f"• This dataset had {easy_proportion * 100:.1f}% Easy, {hard_proportion * 100:.1f}% Hard, and "
            f"{otb_proportion * 100:.1f}% On the Bubble data.",
            fontsize=22,
        )
        return [
            {
                "deck": self._deck,
                "layout_name": "TwoImageTextNoHeader",
                "layout_arguments": {
                    "title": "Survivor Dataset Breakdown",
                    "content_left": definition_text,
                    "content_right": output_image_path,
                },
            },
        ]
