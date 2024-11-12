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
"""RealLabel test stage."""

import copy
import json
import os
import shutil
import warnings
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional, Union

import maite.protocols.object_detection as od
import pyspark.sql.functions as sf
import pyspark.sql.types as st
from maite.workflows import evaluate
from pyspark.errors import AnalysisException
from pyspark.sql import DataFrame, SparkSession
from reallabel import Config, MAITERealLabel, RealLabelColumns, plot_reallabel_results

from jatic_ri._common.test_stages.interfaces.plugins import (
    MultiModelPlugin,
    SingleDatasetPlugin,
)
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage

FILENAME_INDEX = "img_filename"

OUTPUT_DIR = Path(os.path.abspath(__file__)).parent / "reallabel_outputs"

# These _METADATA_ constants are the names of the metadata fields that must be present in the MAITE Dataset for
# RealLabel to be able to retrieve the source images to overlay RealLabel visualization onto.
_METADATA_IMAGE_FILE_NAME_FIELD = "img_filename"
_METADATA_IMAGE_PARENT_DIR_FIELD = "local_filepath"
_REALLABEL_REQUIRED_METADATA_FIELDS = {_METADATA_IMAGE_FILE_NAME_FIELD, _METADATA_IMAGE_PARENT_DIR_FIELD}

# This constant represents the expected location of the RealLabel output directory under the test_stage.cache_base_path
# directory where the RealLabelTestStage.run() function can store visualizations in the event of a cache miss. Since
# these results may be needed past the lifetime of the TestStage, these results will just be left until deleted by the
# user. NOTE: There should only ever be one file in here at a time since the image file will just be overwritten by
# the next cache miss.
_REALLABEL_CACHE_MISS_OUTPUT_DIR = Path("reallabel_cache_miss_outputs")

# These constants represent the expected names and locations of the RealLabel cache results to be found
# under the test_stage.cache_base_path / test_stage.cache_id directory,
# and should be used as `self.cache_base_dir/self.cache_id/_REALLABEL_CACHE_CSV_PATH`
_REALLABEL_CACHE_CSV_PATH = Path("reallabel_standard_results.csv")
_REALLABEL_CACHE_IMAGE_PATH = Path("reallabel_result_visualization.png")
# This file will contain a json-ified list of the various parameters that went into generating the hash ID including
# all model IDs, the dataset ID, and RealLabel Config values.
_REALLABEL_CACHE_CONFIGURATION_PATH = Path("reallabel_cache_configuration.json")


class RealLabelCache(Cache[tuple[DataFrame, Path]]):
    """Cache implementation for RealLabelTestStage.

    The cache directory will, at minimum, contain two files: The RealLabelResults.results dataframe saved to a csv,
    and the png image with the most bounding boxes updated to include ReallLabel results visualized on top.

    Attributes:
        cache_configuration (dict[str, Any]): A dictionary of information relating to the configuration of the
            RealLabelTestStage providing data to the cache. If set, when write_cache() is called, an additional
            json file will be added to the cache with the configuration information.
    """

    def __init__(self, configuration: Optional[dict[str, Any]] = None) -> None:
        self.cache_configuration: Optional[dict[str, Any]] = configuration
        super().__init__()

    def read_cache(self, cache_path: str) -> Optional[tuple[DataFrame, Path]]:
        """Read in cache from cache_path.

        Args:
            cache_path (str): path to RealLabel results cache

        Returns:
            tuple (DataFrame, Path):
                [0]: The cached RealLabel results as a pyspark dataframe
                [1]: The path to the cached RealLabel result image.
        """
        try:
            cached_results_csv_file_path = Path(cache_path) / _REALLABEL_CACHE_CSV_PATH
            spark: SparkSession = SparkSession.builder.getOrCreate()  # type: ignore
            cached_results_df = spark.read.csv(str(cached_results_csv_file_path), header=True, inferSchema=True).drop(
                "_c0",
            )
            cached_results_df = cached_results_df.withColumn(
                "group_winner_box_coords",
                sf.from_json("group_winner_box_coords", st.ArrayType(st.IntegerType())),
            )
            cached_results_img_file_path = Path(cache_path) / _REALLABEL_CACHE_IMAGE_PATH
            if not cached_results_img_file_path.exists():
                raise ValueError(f"RealLabel cache path {cache_path} doesn't contain a cached result visualization!")

        except (OSError, ValueError, AnalysisException) as e:  # pragma: no cover
            warnings.warn(
                f"Cache could not be read. "
                f"Set use_cache to `False` to run without caching."
                f"\nError Message: {e}",
                stacklevel=2,
            )
            return None

        return cached_results_df, cached_results_img_file_path

    def write_cache(self, cache_path: str, data: tuple[DataFrame, Path]) -> None:
        """Write the given RealLabel result data to cache.

        Args:
            cache_path (str): path to cache
            data tuple(DataFrame, Path): data to write to cache consists of two elements:
                [0]: The DataFrame of RealLabel results.
                [1]: The path to the image to cache.
        """
        results_df, results_img = data

        cached_results_csv_file_path = Path(cache_path) / _REALLABEL_CACHE_CSV_PATH
        cached_results_img_file_path = Path(cache_path) / _REALLABEL_CACHE_IMAGE_PATH

        cached_results_csv_file_path.parent.mkdir(parents=True, exist_ok=True)

        with cached_results_csv_file_path.open("w+") as file:
            file.write(results_df.toPandas().to_csv())

        if results_img:
            Path(cached_results_img_file_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(results_img, cached_results_img_file_path)

        if self.cache_configuration:
            cached_results_configuration_file_path = Path(cache_path) / _REALLABEL_CACHE_CONFIGURATION_PATH
            with cached_results_configuration_file_path.open("w+") as file:
                file.write(json.dumps(self.cache_configuration))


class RealLabelTestStage(
    MultiModelPlugin[od.Model], SingleDatasetPlugin[od.Dataset], TestStage[tuple[DataFrame, Path]]
):
    """RealLabel test stage.

    RealLabel uses an ensemble of models and their inference results to provide insight into the potential
    correctness or incorrectness of ground truth labels, and which ones may need to be re-examined or re-labeled.

    For more info see our docs! https://jatic.pages.jatic.net/morse/reallabel/

    This test stage also uses MAITE-wrapped models and datasets, and MAITE itself, to produce the model inference
    results needed if they are not present in the cache before running RealLabel itself.

    Attributes:
        config (Config): The RealLabel Config object that should be used when running Reallabel.
        cache (RealLabelCache): The RealLabelCache object used to read from and write to cache locations.
        outputs (Optional[tuple[DataFrame, Path]]): A tuple of RealLabel results with the layout:
            [0]: The RealLabelResults.results dataframe.
            [1]: A Path to the visualization of the RealLabelResults on the image with the most bounding boxes.
                NOTE: The use of "the image with the most bounding boxes" is pretty arbitrary and just chosen
                      to show something interesting. Potentially configurable.
        models (dict[str, od.Model]): The dictionary of model names to their MAITE-wrapped model objects whose
            inference should be used when running RealLabel.
        dataset (od.Dataset): The MAITE-wrapped dataset object on which the models should run inference
            and produce results.
    """

    def __init__(
        self,
        config: Union[Config, dict[str, Any]],
    ) -> None:
        """Initialize the RealLabel test stage.

        Args:
            config (Union[Config, dict[str, Any]]): The RealLabel Config object that should be used when running
                Reallabel. Or a dict representing a RealLabel config in a json readable format.
        """
        self.config: Config = Config(**config) if isinstance(config, dict) else config
        # Need AG for visualization. Add it to the config if it's not provided. This is a RealLabel oversight
        # that we need to fix.
        if RealLabelColumns.AGGREGATED_CONFIDENCE.value not in self.config.additional_columns_clean_results:
            self.config.additional_columns_clean_results.append(RealLabelColumns.AGGREGATED_CONFIDENCE.value)

        # self.outputs is where we store `run()` results. It is a tuple containing the following:
        #  [0]: pyspark DataFrame containing output results
        #  [1]: Path object pointing to image of a bar plot showing distribution of RealLabel results

        # A dictionary of identifying information that will be hashed into an ID
        self._cache_configuration: Optional[dict[str, Any]] = None

        super().__init__()

    def validate_input_present(self) -> None:
        """Validates that the requisite inputs have been provided: models, metric, and dataset."""
        if not getattr(self, "dataset", None):
            raise RuntimeError("Dataset not set! Please use `load_dataset()` function to set the dataset.")
        if not getattr(self, "models", None):
            raise RuntimeError("Models not set! Please use `load_models()` function to load the models.")

    def _generate_cache_config(self) -> None:
        """Examines the RealLabel config, the MAITE-models, and MAITE-dataset to update self._cache_configuration.

        Also updates the cache configuration of this instance's RealLabelCache().
        """
        self.validate_input_present()

        sorted_model_ids = list(self.models.keys())
        sorted_model_ids.sort()
        reallabel_config = copy.deepcopy(self.config.__dict__)
        # column_names is a ColumnNameConfig which also needs to be unpacked.
        reallabel_config["column_names"] = reallabel_config["column_names"].__dict__

        self._cache_configuration = {
            "model_ids": sorted_model_ids,
            "dataset_id": self.dataset_id,
            "reallabel_config": reallabel_config,
        }
        self.cache: Optional[Cache[tuple[DataFrame, Path]]] = RealLabelCache(self._cache_configuration)

    @property
    def cache_id(self) -> str:
        """Generate the cache ID for this instance based on the Model IDs, Dataset ID, and RealLabel Config."""
        if self._cache_configuration is None:
            self._generate_cache_config()
        config_hash_string = sha256(json.dumps(self._cache_configuration).encode("utf-8")).hexdigest()

        return f"reallabel_cache_{config_hash_string}"

    def __run_metrics(self) -> dict[str, list]:  # pragma: no cover
        """Generate metrics from maite models."""
        clean_predictions = {}
        # Run all the models
        for model in self.models:
            _, predictions, _ = evaluate(
                model=self.models[model],
                dataset=self.dataset,
                return_preds=True,
            )
            clean_predictions[model] = [x[0] for x in predictions]
        return clean_predictions

    def __create_image_name_map(self, image_df: DataFrame) -> dict:
        """Create image name map from image_df.

        Args:
            image_df (DataFrame): Image dataframe, expected to only have one image, possibly spanning multiple rows.
        """
        out = {}
        try:
            image_row = image_df.first()  # issue 123
            if image_row is not None:
                for col in self.config.column_names.unique_identifier_columns:
                    out[image_row[_METADATA_IMAGE_FILE_NAME_FIELD]] = {col: image_row[col]}
        except Exception as e:
            raise ValueError(  # pragma: no cover
                "Unique Identifier Columns must include `img_filename`",
            ) from e

        return out

    def _run(self) -> tuple[DataFrame, Path]:
        """Run RealLabel test stage."""
        self.validate_input_present()

        if not hasattr(self.dataset, "_dataset_path"):
            raise AttributeError(
                "Not great, but we do need a _dataset_path attribute in the Dataset object to know where to retrieve "
                "raw image data from. The alternative is to try and load data straight from the Dataset, but we can't "
                "be guaranteed about its shape (i.e, compatible out-of-the-box with PIL.Image.fromarray()).",
            )

        # Create a spark DataFrame of Datum metadata (always index 2 in the tuple returned by iterating through a
        # Dataset), then check for required metadata fields.
        metadata_dicts_list = [datum[2] for datum in self.dataset]

        spark: SparkSession = SparkSession.builder.getOrCreate()  # type: ignore
        dataset_metadata_df = spark.createDataFrame(metadata_dicts_list)  # type: ignore  # issue 123
        if not _REALLABEL_REQUIRED_METADATA_FIELDS.issubset(set(dataset_metadata_df.columns)):
            raise ValueError(
                f"Dataset metadata does not contain the required fields: {_REALLABEL_REQUIRED_METADATA_FIELDS}",
            )

        # Run metrics
        model_inference_result = self.__run_metrics()

        # Create MAITERealLabel wrapper
        reallabel = MAITERealLabel(
            model_inference_results=model_inference_result,
            config=self.config,
            link_unique_identifier_columns_and_metadata=True,
            ground_truth_dataset=self.dataset,
        )

        # Run RealLabel
        reallabel_results = reallabel.run().results

        # Get the filename of the image in the results that has the most bounding boxes, so it is easily seen in
        # visualizer. Then retrieve its associated dataset sub-directory from the metadata and create a smaller
        # version of the full results df that's only for that image. These will then be turned into a RealLabel
        # visualization saved to a temporary directory that will later be copied into the proper cacheing directory.
        image_filenames = (
            reallabel_results.groupBy(_METADATA_IMAGE_FILE_NAME_FIELD).count().orderBy(sf.col("count").desc()).first()
        )
        image_filename = image_filenames[_METADATA_IMAGE_FILE_NAME_FIELD] if image_filenames else None  # issue 123:
        image_parent_dir = dataset_metadata_df.where(sf.col(_METADATA_IMAGE_FILE_NAME_FIELD) == image_filename).first()[
            _METADATA_IMAGE_PARENT_DIR_FIELD
        ]

        results_for_visualized_image_df = reallabel_results.where(
            sf.col(_METADATA_IMAGE_FILE_NAME_FIELD) == image_filename,
        )

        # Clear out the cache miss dir in preparation for our new results.
        cache_miss_output_img_path = (
            Path(self.cache_base_path) / _REALLABEL_CACHE_MISS_OUTPUT_DIR / _REALLABEL_CACHE_IMAGE_PATH
        )
        if cache_miss_output_img_path.parent.exists():
            shutil.rmtree(cache_miss_output_img_path.parent)
        cache_miss_output_img_path.parent.mkdir(parents=True, exist_ok=True)

        plot_reallabel_results(
            image_location=self.dataset._dataset_path  # noqa: SLF001  # type: ignore  # issue 123
            / image_parent_dir
            / image_filename,
            reallabel_results_df=results_for_visualized_image_df,
            output_location=cache_miss_output_img_path,
            reallabel_config=self.config,
            image_name_map=self.__create_image_name_map(results_for_visualized_image_df),
        )

        return reallabel_results, cache_miss_output_img_path

    def collect_metrics(self) -> dict[str, float]:
        """Collect metrics on total number of Re-labels found."""
        if self.outputs is None:
            raise RuntimeError(
                "Test stage `run()` function must be called before `collect_metrics()`.",
            )

        results_df, _ = self.outputs
        num_false_positives = results_df.where(
            sf.col("reallabel_type") == "Likely Wrong",
        ).count()
        num_false_negatives = results_df.where(
            sf.col("reallabel_type") == "Likely Missed",
        ).count()

        return {
            "NUM_Re-Label": num_false_positives + num_false_negatives,
        }

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect all report consumables."""
        if self.outputs is None:
            raise RuntimeError(
                "Test stage `run()` function must be called before `collect_report_consumables()`.",
            )

        results_df, results_img = self.outputs

        # Find RealLabel statistics
        num_false_positives = results_df.where(
            sf.col("reallabel_type") == "Likely Wrong",
        ).count()
        num_false_negatives = results_df.where(
            sf.col("reallabel_type") == "Likely Missed",
        ).count()
        num_true_positives = results_df.where(
            sf.col("reallabel_type") == "Likely Correct",
        ).count()

        return [
            {
                "deck": "object_detection_dataset_evaluation",
                "layout_name": "TwoImageTextNoHeader",
                "layout_arguments": {
                    "title": "RealLabel Label Breakdown",
                    "content_left": '{"fontsize": 22}'
                    f"**Description**\n"
                    f"* RealLabel aids re-labeling efforts by using model ensembling to determine if a label is a:\n"
                    f"* True Positive Label: probably correct label.\n"
                    f"* False Positive Label: potentially incorrect label.\n"
                    f"* False Negative Label: potentially missing label.\n"
                    f"* In an example subset of the data, RealLabel has found {num_true_positives} True Positive, "
                    f"{num_false_positives} False Positive, and {num_false_negatives} False Negative labels.\n"
                    f"Displayed is an example of a True Positive label.",
                    "content_right": results_img,
                },
            },
        ]
