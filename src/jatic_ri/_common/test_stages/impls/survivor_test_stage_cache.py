import json
import shutil
import warnings
from pathlib import Path
from typing import Any, Optional

import pyspark.sql.functions as sf
import pyspark.sql.types as st
from pyspark.errors import AnalysisException
from pyspark.sql import DataFrame, SparkSession

from jatic_ri._common.test_stages.interfaces.test_stage import Cache

# This file will contain a json-ified list of the various parameters that went into generating the hash ID including
# all model IDs, the dataset ID, the metric ID, and Survivor Config values.
_SURVIVOR_CACHE_CONFIGURATION_PATH = Path("survivor_cache_configuration.json")


class SurvivorCache(Cache[tuple[DataFrame, Path]]):
    """Cache implementation for SurvivorTestStage.

    The cache directory will, at minimum, contain two files: The SurvivorResults.raw_output_df dataframe saved to a
    csv, and a png image with a histogram of the number of images per Survivor category, Easy, On the Bubble, and Hard.

    Attributes:
        cache_configuration: A dictionary of information relating to the configuration of the
            RealLabelTestStage providing data to the cache. If set, when write_cache() is called, an additional
            json file will be added to the cache with the configuration information.
    """

    def __init__(self, cache_configuration: Optional[dict[str, Any]] = None, test_stage: str = "") -> None:
        """
        Initialize the SurvivorCache.

        Args:
            cache_configuration: A dictionary of information relating to the configuration of the
                RealLabelTestStage providing data to the cache. If set, when write_cache() is called, an additional
                json file will be added to the cache with the configuration information.
            test_stage: The name of the test stage."""

        self.cache_configuration: Optional[dict[str, Any]] = cache_configuration
        test_stage = test_stage + "_" if test_stage != "" else test_stage
        self.cache_csv_path: Path = Path(test_stage + "survivor_standard_results.csv")
        self.cache_image_path: Path = Path(test_stage + "survivor_result_visualization.png")
        super().__init__()

    def read_cache(self, cache_path: str) -> Optional[tuple[DataFrame, Path]]:
        """Read in cache from cache_path

        Args:
            cache_path: path to Survivor results cache

        Returns:
            tuple
                [0]: The cached Survivor results as a pyspark dataframe
                [1]: The path to the cached Survivor result image.
        """
        cached_results_csv_file_path = Path(cache_path) / self.cache_csv_path
        cached_image_path = Path(cache_path) / self.cache_image_path
        if not (cached_results_csv_file_path.exists() and cached_image_path.exists()):
            return None

        spark: SparkSession = SparkSession.builder.getOrCreate()  # type: ignore
        try:
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
        except AnalysisException as e:  # pragma: no cover
            warnings.warn(f"Cache could not be read: {e}", stacklevel=2)
            return None

        return cached_results_df, cached_image_path

    def write_cache(self, cache_path: str, data: tuple[DataFrame, Path]) -> None:
        """Write given data to cache.

        Args:
            cache_path: path to cache
            data: data to write to cache consists of two elements in a tuple:
                [0]: The DataFrame of RealLabel results.
                [1]: The path to the image to cache.
        """
        results_df, results_img = data

        cached_results_csv_file_path = Path(cache_path) / self.cache_csv_path
        cached_results_img_file_path = Path(cache_path) / self.cache_image_path

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
