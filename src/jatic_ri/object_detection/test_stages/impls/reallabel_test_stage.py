"""Object Detection RealLabel test stage."""

import copy
import json
import shutil
import warnings
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional, Union

import maite.protocols.object_detection as od
import numpy as np
import pyspark.sql.functions as sf
import pyspark.sql.types as st
import torch
from gradient import Text
from PIL import Image
from pyspark.errors import AnalysisException
from pyspark.sql import DataFrame, SparkSession
from reallabel import Config, MAITERealLabel, RealLabelColumns, plot_reallabel_results

from jatic_ri._common.test_stages.interfaces.plugins import EvalToolPlugin, MultiModelPlugin, SingleDatasetPlugin
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage

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
        cache_configuration: A dictionary of information relating to the configuration of the
            RealLabelTestStage providing data to the cache. If set, when write_cache() is called, an additional
            json file will be added to the cache with the configuration information.
    """

    def __init__(self, configuration: Optional[dict[str, Any]] = None) -> None:
        self.cache_configuration: Optional[dict[str, Any]] = configuration
        super().__init__()

    def read_cache(self, cache_path: str) -> Optional[tuple[DataFrame, Path]]:
        """Read in cache from cache_path.

        Args:
            cache_path: path to RealLabel results cache

        Returns:
            tuple:
                [0]: The cached RealLabel results as a pyspark dataframe
                [1]: The path to the cached RealLabel result image.
        """
        cached_results_csv_file_path = Path(cache_path) / _REALLABEL_CACHE_CSV_PATH
        cached_results_img_file_path = Path(cache_path) / _REALLABEL_CACHE_IMAGE_PATH
        if not (cached_results_csv_file_path.exists() and cached_results_img_file_path.exists()):
            return None

        try:
            spark: SparkSession = SparkSession.builder.getOrCreate()  # type: ignore
            cached_results_df = spark.read.csv(str(cached_results_csv_file_path), header=True, inferSchema=True).drop(
                "_c0",
            )
            cached_results_df = cached_results_df.withColumn(
                "group_winner_box_coords",
                sf.from_json("group_winner_box_coords", st.ArrayType(st.IntegerType())),
            )
        except AnalysisException as e:  # pragma: no cover
            warnings.warn(f"Cache could not be read: {e}", stacklevel=2)

            return None

        return cached_results_df, cached_results_img_file_path

    def write_cache(self, cache_path: str, data: tuple[DataFrame, Path]) -> None:
        """Write the given RealLabel result data to cache.

        Args:
            cache_path: path to cache
            data: data to write to cache consists of two elements:
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
    MultiModelPlugin[od.Model], SingleDatasetPlugin[od.Dataset], TestStage[tuple[DataFrame, Path]], EvalToolPlugin
):
    """RealLabel test stage.

    RealLabel uses an ensemble of models and their inference results to provide insight into the potential
    correctness or incorrectness of ground truth labels, and which ones may need to be re-examined or re-labeled.

    For more info see our docs! https://jatic.pages.jatic.net/morse/reallabel/

    This test stage also uses MAITE-wrapped models and datasets, and MAITE itself, to produce the model inference
    results needed if they are not present in the cache before running RealLabel itself.

    Attributes:
        config: The RealLabel Config object that should be used when running Reallabel.
        cache: The RealLabelCache object used to read from and write to cache locations.
        outputs: A tuple of RealLabel results with the layout:
            [0]: The RealLabelResults.results dataframe.
            [1]: A Path to the visualization of the RealLabelResults on the image with the most bounding boxes.
                NOTE: The use of "the image with the most bounding boxes" is pretty arbitrary and just chosen
                      to show something interesting. Potentially configurable.
        models: The dictionary of model names to their MAITE-wrapped model objects whose
            inference should be used when running RealLabel.
        dataset: The MAITE-wrapped dataset object on which the models should run inference
            and produce results.
    """

    _deck: str = "object_detection_reallabel"
    _task: str = "od"

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

        self.cache: Optional[Cache[tuple[DataFrame, Path]]] = RealLabelCache(self._cache_configuration)

        super().__init__()

    def _generate_cache_config(self) -> None:
        """Examines the RealLabel config, the MAITE-models, and MAITE-dataset to update self._cache_configuration.

        Also updates the cache configuration of this instance's RealLabelCache().
        """
        self.validate_plugins()

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

        return f"reallabel_{self._task}_cache_{config_hash_string}"

    def __run_metrics(self) -> dict[str, list]:  # pragma: no cover
        """Generate metrics from maite models."""
        clean_predictions = {}
        # Run all the models
        for model in self.models:
            predictions, _ = self.eval_tool.predict(
                model=self.models[model], model_id=model, dataset=self.dataset, dataset_id=self.dataset_id
            )
            clean_predictions[model] = [x[0] for x in predictions]
        return clean_predictions

    def _run(self) -> tuple[DataFrame, Path]:
        """Run RealLabel test stage."""
        self.validate_plugins()

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

        if reallabel_results.isEmpty():
            raise RuntimeError("Reallabel result pyspark df is empty!")

        # Clear out the cache miss dir in preparation for our new results.
        cache_miss_output_img_path = (
            Path(self.cache_base_path) / _REALLABEL_CACHE_MISS_OUTPUT_DIR / _REALLABEL_CACHE_IMAGE_PATH
        )
        if cache_miss_output_img_path.parent.exists():
            shutil.rmtree(cache_miss_output_img_path.parent)
        cache_miss_output_img_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a directory where we can save the tensor image for the image with the most bounding boxes.
        most_populous_image_temp_dir_path = (
            Path(self.cache_base_path)
            / _REALLABEL_CACHE_MISS_OUTPUT_DIR
            / "directory_for_visualization_input_base_images"
        )
        most_populous_image_temp_dir_path.mkdir()

        # Get the UUID of the image that has the most bounding boxes, so it is easily seen in
        # visualizer and get a dataframe with all rows associated with that image.
        most_populous_image_uuids_columns_to_values = (
            reallabel_results.groupBy(self.config.column_names.unique_identifier_columns)
            .count()
            .orderBy(sf.col("count").desc())
            .drop("count")
            .first()
            .asDict()  # type: ignore
        )
        results_for_most_populous_image_df = reallabel_results.filter(
            *[sf.col(key) == value for key, value in most_populous_image_uuids_columns_to_values.items()]
        )

        # Iterate through the dataset until we find the index with the matching UUID. Get the Image tensor from it.
        most_populous_image_array = None
        for image_tensor, _, image_metadata in self.dataset:
            if all(
                str(image_metadata[key]) == str(value)
                for key, value in most_populous_image_uuids_columns_to_values.items()
            ):
                most_populous_image_array = image_tensor
        if most_populous_image_array is None:
            raise ValueError("Where's the image?!")

        # Since we aren't guaranteed an actual file name metadata field (not all datasets have them),
        # make a new file name with the raw UUID values
        most_populous_image_file_name = (
            "_".join(value for value in most_populous_image_uuids_columns_to_values.values()) + ".jpeg"
        )
        # Ensure we have a mapping of this portmanteau file name to the actual UUID values that
        # will be found in the dataframe.
        most_populous_image_name_to_uuid_value_map = {
            most_populous_image_file_name: most_populous_image_uuids_columns_to_values
        }

        # Save the image to the temporary directory (workaround for potentially not knowing the original datapath)
        most_populous_image_final_path = most_populous_image_temp_dir_path / most_populous_image_file_name

        # PIL stores as HWC, but MAITE and the RI requires CHW
        if isinstance(most_populous_image_array, torch.Tensor):
            most_populous_image_numpy = most_populous_image_array.permute((1, 2, 0)).numpy()
        elif isinstance(most_populous_image_array, np.ndarray):
            most_populous_image_numpy = most_populous_image_array.transpose((1, 2, 0))
        else:
            raise RuntimeError(
                f"Reallabel Test Stage Error: image array type not understood ({type(most_populous_image_array)})"
            )

        Image.fromarray(most_populous_image_numpy).save(most_populous_image_final_path)

        # Plot the RealLabel results
        plot_reallabel_results(
            image_location=most_populous_image_final_path,
            reallabel_results_df=results_for_most_populous_image_df,
            output_location=cache_miss_output_img_path,
            reallabel_config=self.config,
            image_name_map=most_populous_image_name_to_uuid_value_map,
        )

        return reallabel_results, cache_miss_output_img_path

    def collect_metrics(self) -> dict[str, float]:
        """Collect metrics on total number of Re-labels found."""
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
                "deck": self._deck,
                "layout_name": "TwoImageTextNoHeader",
                "layout_arguments": {
                    "title": "RealLabel Label Breakdown",
                    "content_left": Text(
                        content=f"**Description**\n"
                        f"• RealLabel aids re-labeling efforts by using model ensembling to "
                        f"determine if a label is a:\n"
                        f"• True Positive Label: probably correct label.\n"
                        f"• False Positive Label: potentially incorrect label.\n"
                        f"• False Negative Label: potentially missing label.\n"
                        f"• In an example subset of the data, RealLabel has found {num_true_positives} True Positive, "
                        f"{num_false_positives} False Positive, and {num_false_negatives} False Negative labels.\n"
                        f"Displayed is an example of a True Positive label.",
                        fontsize=22,
                    ),
                    "content_right": results_img,
                },
            },
        ]
