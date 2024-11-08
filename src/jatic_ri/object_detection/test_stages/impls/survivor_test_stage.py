# COPYRIGHTS AND PERMISSIONS:
# Copyright 2024 MORSECORP, Inc. All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""File for Survivor test stage."""

import copy
import json
import shutil
import warnings
from hashlib import sha256
from pathlib import Path
from typing import Any, Generic, Optional, Union

import maite.protocols.object_detection as od
import numpy as np
import pyspark.sql.functions as sf
import pyspark.sql.types as st
from maite import protocols as pr
from maite.workflows import evaluate
from pyspark.errors import AnalysisException
from pyspark.sql import DataFrame, SparkSession
from survivor.analysis import HistogramBarPlot
from survivor.config import Config as SurvivorConfig
from survivor.maite_survivor import MAITESurvivor

from jatic_ri._common.test_stages.interfaces.plugins import MetricPlugin, MultiModelPlugin, SingleDatasetPlugin
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TData, TestStage

# This constant represents the expected location of the Survivor output directory under the test_stage.cache_base_path
# directory where the SurvivorTestStage.run() function can store visualizations in the event of a cache miss. Since
# these results may be needed past the lifetime of the TestStage, these results will just be left until deleted by the
# user. NOTE: There should only ever be one file in here at a time since the image file will just be overwritten by
# the next cache miss.
_SURVIVOR_CACHE_MISS_OUTPUT_DIR = Path("survivor_cache_miss_outputs")

# These constants represent the expected names and locations of the Survivor cache results to be found
# under the test_stage.cache_base_path / test_stage.cache_id directory,
# and should be used as `self.cache_base_dir/self.cache_id/_SURVIVOR_CACHE_CSV_PATH`
_SURVIVOR_CACHE_CSV_PATH = Path("survivor_standard_results.csv")
_SURVIVOR_CACHE_IMAGE_PATH = Path("survivor_result_visualization.png")
# This file will contain a json-ified list of the various parameters that went into generating the hash ID including
# all model IDs, the dataset ID, the metric ID, and Survivor Config values.
_SURVIVOR_CACHE_CONFIGURATION_PATH = Path("survivor_cache_configuration.json")

# The key that must be present in a MAITE evaluate()'s returned dictionary. This should correspond to another
# dictionary with MetricPlugin.metric_id as a key. This list will then be a datum-by-datum specification of the
# Metric values.
PER_DATUM_METRIC_KEY = "per_datum"


class SurvivorCache(Generic[TData], Cache[TData]):
    """Cache implementation for RealLabelTestStage.

    The cache directory will, at minimum, contain two files: The SurvivorResults.raw_output_df dataframe saved to a
    csv, and a png image with a histogram of the number of images per Survivor category, Easy, On the Bubble, and Hard.

    Attributes:
        cache_configuration (dict[str, Any]): A dictionary of information relating to the configuration of the
            RealLabelTestStage providing data to the cache. If set, when write_cache() is called, an additional
            json file will be added to the cache with the configuration information.
    """

    def __init__(self) -> None:
        self.cache_configuration: dict[str, Any] = None
        super().__init__()

    def read_cache(self, cache_path: str) -> Optional[TData]:
        """Read in cache from cache_path

        Args:
            cache_path (str): path to Survivor results cache

        Returns:
            tuple (DataFrame, Path):
                [0]: The cached Survivor results as a pyspark dataframe
                [1]: The path to the cached Survivor result image.
        """
        try:
            cached_results_csv_file_path = Path(cache_path) / _SURVIVOR_CACHE_CSV_PATH
            spark = SparkSession.builder.getOrCreate()
            cached_results_df = (
                spark.read.csv(
                    str(cached_results_csv_file_path),
                    header=True,
                    inferSchema=True,
                )
                .drop("_c0")
                .withColumn(
                    "image_id",
                    sf.col("image_id").cast(st.StringType()),
                )
            )

            cached_image_path = Path(cache_path) / _SURVIVOR_CACHE_IMAGE_PATH
            if not cached_image_path.exists():
                raise ValueError(f"Survivor cache path {cache_path} doesn't contain a cached result visualization!")

        except (OSError, ValueError, AnalysisException) as e:  # pragma: no cover
            warnings.warn(  # pragma: no cover
                f"Cache could not be read. "
                f"Set use_cache to `False` to run without caching."
                f"\nError Message: {e}",
                stacklevel=2,
            )
            return None

        return cached_results_df, cached_image_path

    def write_cache(self, cache_path: str, data: TData) -> None:
        """Write given data to cache.

        Args:
            cache_path (str): path to cache
            data (TData): data to write to cache consists of two elements in a tuple:
                [0]: The DataFrame of RealLabel results.
                [1]: The path to the image to cache.
        """
        results_df, results_img = data

        cached_results_csv_file_path = Path(cache_path) / _SURVIVOR_CACHE_CSV_PATH
        cached_results_img_file_path = Path(cache_path) / _SURVIVOR_CACHE_IMAGE_PATH

        cached_results_csv_file_path.parent.mkdir(parents=True, exist_ok=True)

        with cached_results_csv_file_path.open("w+") as file:
            file.write(results_df.toPandas().to_csv())

        if results_img:
            cached_results_img_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(results_img, cached_results_img_file_path)

        if self.cache_configuration:
            cached_results_configuration_file_path = Path(cache_path) / _SURVIVOR_CACHE_CONFIGURATION_PATH
            with cached_results_configuration_file_path.open("w+") as file:
                file.write(json.dumps(self.cache_configuration))


class SurvivorTestStage(
    TestStage[tuple[DataFrame, Path]],
    MultiModelPlugin[od.Model],
    SingleDatasetPlugin[od.Dataset],
    MetricPlugin[od.Metric],
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
        config (SurvivorConfig): The Survivor Config object that should be used when running Survivor.
        cache (SurvivorCache): The SurvivorCache object used to read from and write to cache locations.
        outputs (Optional[tuple[DataFrame, Path]]): A tuple of Survivor results with the layout:
            [0]: The SurvivorResults.raw_output_df dataframe.
            [1]: A Path to a PNG with a histogram of the number of images per Survivor category: Easy, Hard, and
                On the Bubble.
        metric (od.Metric): The MAITE-wrapped metric object that should be fed the model inference results
            for metric calculation.
        dataset (od.Dataset): The MAITE-wrapped dataset object on which the models should run inference and
            produce results.
        models (dict[str, od.Model]): The dictionary of model names to their MAITE-wrapped model objects
            whose inference should be used when running RealLabel.
    """

    def __init__(
        self,
        config: Union[SurvivorConfig, dict[str, Any]],
    ) -> None:
        """Create instance of SurvivorTestStage

        Args:
            config (Union[SurvivorConfig, dict[str, Any]]): config for survivor run.
        """
        self.config: SurvivorConfig = SurvivorConfig(**config) if isinstance(config, dict) else config
        self.cache: SurvivorCache[tuple[DataFrame, Path]] = SurvivorCache()
        # self.outputs is where we store `run()` results. It is a tuple containing the following:
        #  [0]: pyspark DataFrame containing output results
        #  [1]: Path object pointing to image of a bar plot showing distribution of survivor results
        self.outputs: Optional[tuple[DataFrame, Path]] = None
        self.metric: od.Metric = None
        self.dataset: od.Dataset = None
        self.models: dict[str, od.Model] = None
        # A dictionary of identifying information that will be hashed into an ID
        self._cache_configuration: dict[str, Any] = None

        super().__init__()

    def validate_input_present(self) -> None:
        """Validates that the requisite inputs have been provided: models, metric, and dataset."""
        if not self.metric:
            raise RuntimeError("Metric not set! Please use `load_metric()` function to set the metric.")
        if not self.dataset:
            raise RuntimeError("Dataset not set! Please use `load_dataset()` function to set the dataset.")
        if not self.models:
            raise RuntimeError("Models not set! Please use `load_models()` function to load the models.")

    def _generate_cache_config(self) -> None:
        """Examines the RealLabel config, the MAITE-models, and MAITE-dataset to update self._cache_configuration.

        Also updates the cache configuration of this instance's RealLabelCache().
        """
        self.validate_input_present()
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
        self.cache.cache_configuration = self._cache_configuration

    @property
    def cache_id(self) -> str:
        """Generate the cache ID for based on the Model IDs, Dataset ID, Metric ID, and Survivor Config."""
        if self._cache_configuration is None:
            self._generate_cache_config()
        config_hash_string = sha256(json.dumps(self._cache_configuration).encode("utf-8")).hexdigest()

        return f"survivor_cache_{config_hash_string}"

    def __run_metrics(self) -> dict[str, pr.ArrayLike]:
        """Create metrics by model for use in MAITESurvivor."""
        all_model_metrics_per_datum = {}
        for model in self.models:
            model_metrics_per_datum = []
            # Since evaluate() only works on a full Dataset, and we need metrics on a per-Datum basis,
            # just iterate through each Datum in the Dataset, and throw that single datum into evaluate()
            # in a list, which is compatible with MAITE's definition of a Dataset, therefore mimicking an
            # entire dataset of just one image.
            for datum_information_tuple in self.dataset:
                results, _, _ = evaluate(
                    model=self.models[model],
                    dataset=[datum_information_tuple],
                    metric=self.metric,
                    return_preds=True,
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
        self.validate_input_present()

        # run metrics
        metrics = self.__run_metrics()

        # create maite wrapper and run Survivor
        survivor = MAITESurvivor(
            maite_dataset=self.dataset,
            config=self.config,
            metrics=metrics,
            spark_session=SparkSession.builder.getOrCreate(),
        )
        results_df = survivor.run().raw_output_df

        # Clear out the cache miss dir in preparation for our new results.
        cache_miss_output_output_dir = self.cache_base_path / _SURVIVOR_CACHE_MISS_OUTPUT_DIR
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
            cache_miss_output_img_path.parent / _SURVIVOR_CACHE_IMAGE_PATH,
        )

        return results_df, cache_miss_output_img_path

    def collect_metrics(self) -> dict[str, float]:
        """Collect metrics for the SurvivorTestStage."""
        if self.outputs is None:
            raise RuntimeError("Test stage run() method must be called before collect_metrics().")

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

        if self.outputs is None:
            raise RuntimeError("Test stage run() method must be called before collect_report_consumables().")

        survivor_results, output_image_path = self.outputs

        total = survivor_results.count()
        hard_proportion = survivor_results.where(sf.col("suid_label") == "Hard").count() / total
        easy_proportion = survivor_results.where(sf.col("suid_label") == "Easy").count() / total
        otb_proportion = survivor_results.where(sf.col("suid_label") == "On the Bubble").count() / total

        # Simply return the title of the section and the data to be plotted
        definition_text = (
            '{"fontsize": 22}'
            f"**Types of Data**\n"
            f"* Easy: Models score the same and perform well.\n"
            f"* Hard: Models score the same and perform poorly.\n"
            f"* On the Bubble: Models score differently.\n\n"
            f"* Ideally, a dataset would be primarily On the Bubble, so all data is helping distinguish between model "
            f"performance.\n\n"
            f"* This dataset had {easy_proportion * 100:.1f}% Easy, {hard_proportion * 100:.1f}% Hard, and "
            f"{otb_proportion * 100:.1f}% On the Bubble data."
        )
        return [
            {
                "deck": "object_detection_dataset_evaluation",
                "layout_name": "TwoImageTextNoHeader",
                "layout_arguments": {
                    "title": "Survivor Dataset Breakdown",
                    "content_left": definition_text,
                    "content_right": output_image_path,
                },
            },
        ]
