"""Object Detection RealLabel test stage."""

import copy
import dataclasses
import json
import logging
import shutil
import textwrap
import warnings
from collections.abc import Hashable
from hashlib import sha256
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Optional,
    Union,
)

import maite.protocols.object_detection as od
import numpy as np
import pandas as pd
import pyspark.sql.functions as sf
import torch
from gradient import SubText, Text
from PIL import Image
from pptx.dml.color import RGBColor
from pydantic import BaseModel, BeforeValidator, ConfigDict, field_serializer
from pyspark.sql import DataFrame
from reallabel import (
    MAITERealLabel,
    RealLabelColumns,
    RealLabelConfig,
    plot_reallabel_results,
)
from reallabel import (
    RealLabelResults as RealLabelModuleResults,
)

from jatic_ri import cache_path
from jatic_ri._common.test_stages.interfaces.plugins import (
    EvalToolPlugin,
    MultiModelPlugin,
    SingleDatasetPlugin,
)
from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.util.utils import sanitize_gradient_markdown_text

logger = logging.getLogger()

# This constant represents the expected location of the RealLabel output directory under the cache_path()
# directory where the RealLabelTestStage.run() function can store visualizations in the event of a cache miss. Since
# these results may be needed past the lifetime of the TestStage, these results will just be left until deleted by the
# user. NOTE: There should only ever be one file in here at a time since the image file will just be overwritten by
# the next cache miss.
_REALLABEL_CACHE_MISS_OUTPUT_DIR = "reallabel_cache_miss_outputs"

# These constants represent the expected names and locations of the RealLabel cache results to be found
# under the cache_path() / test_stage.cache_id directory,
# and should be used as `self.cache_base_dir/self.cache_id/_REALLABEL_CACHE_CSV_PATH`
_REALLABEL_CACHE_JSON_PATH = "reallabel_standard_results.json"
_REALLABEL_CACHE_IMAGE_PATH = "reallabel_result_visualization.png"
# This file will contain a json-ified list of the various parameters that went into generating the hash ID including
# all model IDs, the dataset ID, and RealLabel Config values.
_REALLABEL_CACHE_CONFIGURATION_PATH = "reallabel_cache_configuration.json"


def parse_dataframe(value: Union[pd.DataFrame, dict]) -> pd.DataFrame:
    """
    Attempts to parse the input value into a pandas DataFrame.
    Accepts either an existing DataFrame or a dictionary
    in pandas 'split' orientation ({'index': ..., 'columns': ..., 'data': ...}).
    """
    if isinstance(value, pd.DataFrame):
        # Return a copy to prevent mutation issues if the original df is reused
        return value.copy()
    if isinstance(value, dict):
        try:
            # Check for required keys for 'split' orientation
            if not {"index", "columns", "data"}.issubset(value.keys()):
                raise ValueError("Dictionary must contain 'index', 'columns', and 'data' keys for 'split' orientation.")
            # Reconstruct DataFrame using the DataFrame constructor
            return pd.DataFrame(data=value["data"], index=value["index"], columns=value["columns"])
        except Exception as e:
            raise ValueError(f"Failed to create DataFrame from dict: {e}") from e
    raise TypeError(f"Expected a pandas.DataFrame or dict (split format), got {type(value)}")


DataFrameType = Annotated[pd.DataFrame, BeforeValidator(parse_dataframe)]


class RealLabelTestStageResults(BaseModel):
    """Results class for RealLabelTestStage"""

    example_image_path: Path
    default_results: DataFrameType
    classification_disagreements_df: Optional[DataFrameType] = None
    verbose_df: Optional[DataFrameType] = None
    sequence_priority_score_df: Optional[DataFrameType] = None
    sequence_priority_score_balanced_df: Optional[DataFrameType] = None
    wanrs_df: Optional[DataFrameType] = None
    aggregated_confidence_df: Optional[DataFrameType] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer(
        "default_results",
        "classification_disagreements_df",
        "verbose_df",
        "sequence_priority_score_df",
        "sequence_priority_score_balanced_df",
        "wanrs_df",
        "aggregated_confidence_df",
        when_used="always",
    )
    def serialize_df(self: BaseModel, df: Union[pd.DataFrame, None]) -> Union[dict[Hashable, Any], None]:
        """
        Serializes the DataFrame to a dictionary in 'split' orientation.
        Handles None for the optional field.
        'when_used=always' ensures this runs for both model_dump (python) and model_dump_json (json).
        """
        if df is None:
            return None
        return df.to_dict(orient="split")

    @classmethod
    def from_real_label_module_results(
        cls, reallabel_module_results: RealLabelModuleResults, example_image_path: Path
    ) -> "RealLabelTestStageResults":
        """Convert RealLabelModuleResults to RealLabelTestStageResults."""

        # ignore these fields in RealLabelModuleResults
        ignored_fields = {
            "_run_with_ground_truth",
        }
        # fields with different names between RealLabelModuleResults and RealLabelTestStageResults
        renamed_fields = {
            "results": "default_results",
        }
        # additional fields not in RealLabelModuleResults
        extra_fields: dict[str, Any] = {
            "example_image_path": example_image_path,
        }

        real_label_test_stage_results_kwargs = extra_fields
        for field in dataclasses.fields(RealLabelModuleResults):
            if (
                field.name not in cls.__annotations__
                and field.name not in ignored_fields
                and field.name not in renamed_fields
            ):
                # raise a warning
                warnings.warn(
                    f'Ignoring field "{field.name}" b/c not defined in RealLabelTestStageResults while '
                    f'converting "{RealLabelModuleResults.__name__}" to "{cls.__name__}"',
                    stacklevel=2,
                )
                continue
            field_value = getattr(reallabel_module_results, field.name)
            if isinstance(field_value, DataFrame):
                # Convert Spark DataFrame to Pandas DataFrame
                field_value = field_value.toPandas()

            # rename fields
            field_name = renamed_fields.get(field.name, field.name)

            real_label_test_stage_results_kwargs[field_name] = field_value

        return cls(
            **real_label_test_stage_results_kwargs,
        )


class RealLabelCache(Cache[RealLabelTestStageResults]):
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

    def read_cache(self, cache_path: str) -> Optional[RealLabelTestStageResults]:
        """Read in cache from cache_path.

        Args:
            cache_path: path to RealLabel results cache

        Returns:
            tuple:
                [0]: The cached RealLabel results object with all dataframes
                [1]: The path to the cached RealLabel result image.
        """
        logger.info(f"Checking for existing cache at {cache_path}")
        cache_dir = Path(cache_path)

        # Check for the results file
        if not (cache_dir / _REALLABEL_CACHE_JSON_PATH).exists():
            return None

        # Load the results object
        main_results_path = cache_dir / _REALLABEL_CACHE_JSON_PATH
        with main_results_path.open() as file:
            reallabel_results = RealLabelTestStageResults.model_validate_json(file.read())

        if not reallabel_results.example_image_path.exists():
            warnings.warn(
                f"Cached image path {reallabel_results.example_image_path} does not exist.  Ignoring cached results.",
                stacklevel=2,
            )
            return None

        return reallabel_results

    def write_cache(self, cache_path: str, data: RealLabelTestStageResults) -> None:
        """Write the given RealLabel result data to cache.

        Args:
            cache_path: path to cache
            data: data to write to cache consists of two elements:
                [0]: The RealLabelResults object with all dataframes.
                [1]: The path to the image to cache.
        """
        logger.info(f"Writing cache to {cache_path}")
        # reallabel_results = data
        cache_dir = Path(cache_path)

        # Create cache directory
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save the results
        data = data.model_copy(deep=True)  # Ensure we don't mutate the original data
        if data.example_image_path:
            cached_results_img_file_path = cache_dir / _REALLABEL_CACHE_IMAGE_PATH
            Path(cached_results_img_file_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(data.example_image_path, cached_results_img_file_path)
            data.example_image_path = cached_results_img_file_path

        with (cache_dir / _REALLABEL_CACHE_JSON_PATH).open("w+") as file:
            file.write(data.model_dump_json())

        # Save configuration if available
        if self.cache_configuration:
            cached_results_configuration_file_path = cache_dir / _REALLABEL_CACHE_CONFIGURATION_PATH
            with cached_results_configuration_file_path.open("w+") as file:
                file.write(json.dumps(self.cache_configuration))


class RealLabelTestStage(
    MultiModelPlugin[od.Model],
    SingleDatasetPlugin[od.Dataset],
    TestStage[RealLabelTestStageResults],
    EvalToolPlugin,
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
        config: Union[RealLabelConfig, dict[str, Any]],
    ) -> None:
        """Initialize the RealLabel test stage.

        Args:
            config (Union[Config, dict[str, Any]]): The RealLabel Config object that should be used when running
                Reallabel. Or a dict representing a RealLabel config in a json readable format.
        """
        self.config: RealLabelConfig = RealLabelConfig(**config) if isinstance(config, dict) else config
        # Need AC for visualization. Add it to the config if it's not provided. This is a RealLabel oversight
        # that we need to fix.
        if RealLabelColumns.AGGREGATED_CONFIDENCE.value not in self.config.additional_columns_clean_results:
            self.config.additional_columns_clean_results.append(RealLabelColumns.AGGREGATED_CONFIDENCE.value)

        # self.outputs is where we store `run()` results. It is a tuple containing the following:
        #  [0]: pyspark DataFrame containing output results
        #  [1]: Path object pointing to image of a bar plot showing distribution of RealLabel results

        # A dictionary of identifying information that will be hashed into an ID
        self._cache_configuration: Optional[dict[str, Any]] = None

        self.cache: Optional[Cache[RealLabelTestStageResults]] = RealLabelCache(self._cache_configuration)

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
        self.cache: Optional[Cache[RealLabelTestStageResults]] = RealLabelCache(self._cache_configuration)

    @property
    def cache_id(self) -> str:
        """Generate the cache ID for this instance based on the Model IDs, Dataset ID, and RealLabel Config."""
        self._generate_cache_config()
        config_hash_string = sha256(json.dumps(self._cache_configuration).encode("utf-8")).hexdigest()

        return f"reallabel_{self._task}_cache_{config_hash_string}"

    def __run_metrics(self) -> dict[str, list]:  # pragma: no cover
        """Generate metrics from maite models."""
        clean_predictions = {}
        # Run all the models
        for model in self.models:
            predictions, _ = self.eval_tool.predict(
                model=self.models[model],
                model_id=model,
                dataset=self.dataset,
                dataset_id=self.dataset_id,
            )
            clean_predictions[model] = [x[0] for x in predictions]
        return clean_predictions

    def _run(self) -> RealLabelTestStageResults:
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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="It is preferred to use 'applyInPandas' over this API.*",
                module="pyspark.sql.pandas.group_ops",
            )
            reallabel_results = reallabel.run()
            default_reallabel_results = reallabel_results.results

        if default_reallabel_results.isEmpty():
            raise RuntimeError("Reallabel result pyspark df is empty!")

        # Clear out the cache miss dir in preparation for our new results.
        cache_miss_output_img_path = cache_path() / _REALLABEL_CACHE_MISS_OUTPUT_DIR / _REALLABEL_CACHE_IMAGE_PATH
        if cache_miss_output_img_path.parent.exists():
            shutil.rmtree(cache_miss_output_img_path.parent)
        cache_miss_output_img_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a directory where we can save the tensor image for the image with the most bounding boxes.
        most_populous_image_temp_dir_path = (
            cache_path() / _REALLABEL_CACHE_MISS_OUTPUT_DIR / "directory_for_visualization_input_base_images"
        )
        most_populous_image_temp_dir_path.mkdir()

        # Get the UUID of the image that has the most bounding boxes, so it is easily seen in
        # visualizer and get a dataframe with all rows associated with that image.
        self._example_image_unique_id = (
            default_reallabel_results.groupBy(self.config.column_names.unique_identifier_columns)
            .count()
            .orderBy(sf.col("count").desc())
            .drop("count")
            .first()
            .asDict()  # type: ignore
        )

        results_for_most_populous_image_df = default_reallabel_results.filter(
            *[sf.col(key) == value for key, value in self._example_image_unique_id.items()]
        )

        # Iterate through the dataset until we find the index with the matching UUID. Get the Image tensor from it.
        most_populous_image_array = None
        for image_tensor, _, image_metadata in self.dataset:
            if all(str(image_metadata[key]) == str(value) for key, value in self._example_image_unique_id.items()):
                most_populous_image_array = image_tensor
                break
        if most_populous_image_array is None:
            raise ValueError("Where's the image?!")

        # Since we aren't guaranteed an actual file name metadata field (not all datasets have them),
        # make a new file name with the raw UUID values
        most_populous_image_file_name = "_".join(value for value in self._example_image_unique_id.values()) + ".jpeg"
        # Ensure we have a mapping of this portmanteau file name to the actual UUID values that
        # will be found in the dataframe.
        most_populous_image_name_to_uuid_value_map = {most_populous_image_file_name: self._example_image_unique_id}

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

        results = RealLabelTestStageResults.from_real_label_module_results(
            reallabel_results,
            cache_miss_output_img_path,
        )

        return results  # noqa: RET504

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect all report consumables."""
        reallabel_results = self.outputs
        default_results_df = reallabel_results.default_results

        # Find RealLabel statistics
        num_false_positives = (default_results_df["reallabel_type"] == "Likely Wrong").sum()
        num_false_negatives = (default_results_df["reallabel_type"] == "Likely Missed").sum()
        num_true_positives = (default_results_df["reallabel_type"] == "Likely Correct").sum()

        sentence_formatting = {
            "fontsize": 18,
            "bold": True,
        }

        bullet_formatting = {
            "fontsize": 16,
        }

        spaces_formatting = {
            "fontsize": 12,
        }

        def bullet_point(text: str, **text_formatting_kwargs: Any) -> tuple[SubText, SubText]:
            return (
                SubText("• ", fontsize=22),
                SubText(text, **{**bullet_formatting, **text_formatting_kwargs}),
            )

        slides = [
            {
                "deck": self._deck,
                "layout_name": "TwoImageTextNoHeader",
                "layout_arguments": {
                    "title": "RealLabel Label Breakdown",
                    "content_left": Text(
                        content=[
                            # Paragraph 1
                            SubText(
                                "RealLabel aids re-labeling efforts by "
                                "using model ensembling to determine if a label is a:\n",
                                **sentence_formatting,  # type: ignore
                            ),
                            *bullet_point("True Positive: (Probably Correct Label)\n"),
                            *bullet_point("False Positive: (Potentially Incorrect Label)\n"),
                            *bullet_point("False Negative: (Potentially Missing Label)\n"),
                            # Spacing
                            SubText("\n" * 1, **spaces_formatting),  # type: ignore
                            # Paragraph 2
                            SubText(
                                "In this dataset, RealLabel has found:\n",
                                **sentence_formatting,
                            ),
                            # colors defined in https://gitlab.jatic.net/jatic/morse/reallabel/-/blob/0.5.0/src/reallabel/visualizer.py#L525-527
                            *bullet_point(
                                "True Positive: ",
                                color=RGBColor(173, 214, 95),
                                bold=True,
                            ),
                            SubText(
                                f"{num_true_positives}\n",
                                **bullet_formatting,  # type: ignore
                                bold=True,
                            ),
                            *bullet_point(
                                "False Positive: ",
                                color=RGBColor(72, 127, 199),
                                bold=True,
                            ),
                            SubText(
                                f"{num_false_positives}\n",
                                **bullet_formatting,  # type: ignore
                                bold=True,
                            ),
                            *bullet_point(
                                "False Negative: ",
                                color=RGBColor(221, 0, 0),
                                bold=True,
                            ),
                            SubText(
                                f"{num_false_negatives}\n",
                                **bullet_formatting,  # type: ignore
                                bold=True,
                            ),
                            SubText("\n" * 1, **spaces_formatting),  # type: ignore
                            SubText(
                                sanitize_gradient_markdown_text(
                                    "An example image "
                                    f"({','.join({f'{k}: {v}' for k, v in self._example_image_unique_id.items()})})"
                                    " is shown to the right"
                                ),
                                fontsize=14,
                            ),
                        ]
                    ),
                    "content_right": reallabel_results.example_image_path,
                },
            },
        ]

        if self.outputs.wanrs_df is not None:
            wanrs_description_prelink_text = textwrap.dedent(
                "This table shows up to the top 10 images recommended for relabeling, ranked by "
                "RealLabel's WANRS (Weighted Average Normalized Relative Scores) metric.\nWANRS "
                "highlights images where the model is most likely to have labeling mistakes, based "
                "on the confidence and RealLabel assignment (i.e. false positives and false negatives) "
                "of each object detection. Images at the top of the list have a lower WANRS and are "
                "expected to have the most problematic or uncertain labels. "
                "See more info in the ".lstrip()
            )
            slides.append(
                {
                    "deck": self._deck,
                    "layout_name": "TextData",
                    "layout_arguments": {
                        "title": "Reallabel - Top Candidate Images for Relabeling Efforts",
                        "text_column_heading": "WANRS Ranking",
                        "text_column_half": True,
                        "text_column_body": [
                            Text(
                                content=[
                                    wanrs_description_prelink_text,
                                    SubText(
                                        content="RealLabel Documentation",
                                        hyperlink="https://jatic.pages.jatic.net/morse/reallabel/user_guide/explanation/how_reallabel_works.html#optional-output-dataframe-wanrs-output",
                                    ),
                                    ".",
                                    "\n\n",
                                ],
                                fontsize=16,
                            ),
                            Text(
                                "Note: The accuracy of this ranking depends on the quality of the model’s "
                                "predictions. If the model is not well-calibrated or misses certain types of "
                                "errors, some problematic images may not be prioritized.",
                                fontsize=14,
                            ),
                        ],
                        "data_column_table": self.outputs.wanrs_df.loc[:, ["id", "WANRS"]]
                        .head(10)
                        .reset_index()
                        .assign(index=lambda df: df["index"] + 1)
                        .assign(WANRS=lambda df: df.WANRS.apply(lambda x: f"{x:.3f}"))  # format with 3 decimal places
                        .rename(columns={"index": "Priority", "id": "Image ID", "WANRS": "WANRS Score"}),
                    },
                }
            )

        return slides
