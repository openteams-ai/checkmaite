"""Object Detection RealLabel test stage."""

import tempfile
import textwrap
import warnings
from pathlib import Path
from typing import (
    Any,
    Optional,
    Union,
)

import maite.protocols.object_detection as od
import numpy as np
import PIL.Image
import pydantic
import pyspark.sql.functions as sf
import torch
from gradient import SubText, Text
from pptx.dml.color import RGBColor
from reallabel import (
    MAITERealLabel,
    RealLabelColumns,
    plot_reallabel_results,
)
from reallabel import RealLabelConfig as _NativeRealLabelConfig

from jatic_ri._common.test_stages.interfaces.plugins import (
    EvalToolPlugin,
    MultiModelPlugin,
    SingleDatasetPlugin,
)
from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, OutputsBase, RunBase, TestStage
from jatic_ri.util._types import DataFrame, Image
from jatic_ri.util.utils import temp_image_file


class RealLabelOutputs(OutputsBase):
    """Results class for RealLabelTestStage"""

    results: DataFrame
    example_image: Image
    classification_disagreements_df: Optional[DataFrame] = None
    verbose_df: Optional[DataFrame] = None
    sequence_priority_score_df: Optional[DataFrame] = None
    sequence_priority_score_balanced_df: Optional[DataFrame] = None
    wanrs_df: Optional[DataFrame] = None
    aggregated_confidence_df: Optional[DataFrame] = None


# reallabel already provides a pydantic model for its configuration so we just mix in our base
class RealLabelConfig(_NativeRealLabelConfig, ConfigBase):
    """Config class for RealLabelTestStage"""

    pass


class RealLabelRun(RunBase):
    """Run class for RealLabelTestStage"""

    config: RealLabelConfig
    outputs: RealLabelOutputs


class RealLabelTestStage(
    MultiModelPlugin[od.Model],
    SingleDatasetPlugin[od.Dataset],
    TestStage[RealLabelOutputs],
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
        outputs: A `RealLabelTestStageResults` object containing the results of the RealLabel analysis,
            including DataFrames and an example visualization image path.
        models: The dictionary of model names to their MAITE-wrapped model objects whose
            inference should be used when running RealLabel.
        dataset: The MAITE-wrapped dataset object on which the models should run inference
            and produce results.
    """

    _RUN_TYPE = RealLabelRun

    _deck: str = "object_detection_reallabel"
    _task: str = "od"

    def __init__(
        self,
        config: Union[_NativeRealLabelConfig, dict[str, Any]],
    ) -> None:
        """Initialize the RealLabel test stage.

        Args:
            config (Union[Config, dict[str, Any]]): The RealLabel Config object that should be used when running
                Reallabel. Or a dict representing a RealLabel config in a json readable format.
        """
        super().__init__()

        if isinstance(config, pydantic.BaseModel):
            config = config.model_dump()
        self._config = RealLabelConfig.model_validate(config)

        # Need AC for visualization. Add it to the config if it's not provided. This is a RealLabel oversight
        # that we need to fix.
        if RealLabelColumns.AGGREGATED_CONFIDENCE.value not in self._config.additional_columns_clean_results:
            self._config.additional_columns_clean_results.append(RealLabelColumns.AGGREGATED_CONFIDENCE.value)

    def _create_config(self) -> RealLabelConfig:
        return self._config

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

    def _run(self) -> RealLabelOutputs:
        """Run RealLabel test stage."""
        self.validate_plugins()

        # Run metrics
        model_inference_result = self.__run_metrics()

        # Create MAITERealLabel wrapper
        reallabel = MAITERealLabel(
            model_inference_results=model_inference_result,
            config=self._config,
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

        # Get the UUID of the image that has the most bounding boxes, so it is easily seen in
        # visualizer and get a dataframe with all rows associated with that image.
        self._example_image_unique_id = (
            default_reallabel_results.groupBy(self._config.column_names.unique_identifier_columns)
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
        most_populous_image_file_name = "_".join(value for value in self._example_image_unique_id.values()) + ".png"
        # Ensure we have a mapping of this portmanteau file name to the actual UUID values that
        # will be found in the dataframe.
        most_populous_image_name_to_uuid_value_map = {most_populous_image_file_name: self._example_image_unique_id}

        # PIL stores as HWC, but MAITE and the RI requires CHW
        if isinstance(most_populous_image_array, torch.Tensor):
            most_populous_image_numpy = most_populous_image_array.permute((1, 2, 0)).numpy()
        elif isinstance(most_populous_image_array, np.ndarray):
            most_populous_image_numpy = most_populous_image_array.transpose((1, 2, 0))
        else:
            raise RuntimeError(
                f"Reallabel Test Stage Error: image array type not understood ({type(most_populous_image_array)})"
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir)

            image_location = p / most_populous_image_file_name
            PIL.Image.fromarray(most_populous_image_numpy).save(image_location)

            output_location = p / "output.png"

            # Plot the RealLabel results
            plot_reallabel_results(
                image_location=image_location,
                reallabel_results_df=results_for_most_populous_image_df,
                output_location=output_location,
                reallabel_config=self._config,
                image_name_map=most_populous_image_name_to_uuid_value_map,
            )

            return RealLabelOutputs(
                results=default_reallabel_results,  # type: ignore[reportArgumentType]
                example_image=image_location,  # type: ignore[reportArgumentType]
                classification_disagreements_df=reallabel_results.classification_disagreements_df,  # type: ignore[reportArgumentType]
                verbose_df=reallabel_results.verbose_df,  # type: ignore[reportArgumentType]
                sequence_priority_score_df=reallabel_results.sequence_priority_score_df,  # type: ignore[reportArgumentType]
                sequence_priority_score_balanced_df=reallabel_results.sequence_priority_score_balanced_df,  # type: ignore[reportArgumentType]
                wanrs_df=reallabel_results.wanrs_df,  # type: ignore[reportArgumentType]
                aggregated_confidence_df=reallabel_results.aggregated_confidence_df,  # type: ignore[reportArgumentType]
            )

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect all report consumables."""
        reallabel_results = self.outputs
        default_results_df = reallabel_results.results

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
                "layout_name": "TwoItem",
                "layout_arguments": {
                    "title": "RealLabel Label Breakdown",
                    "left_item": Text(
                        content=[
                            # Paragraph 1
                            SubText(
                                "RealLabel aids re-labeling efforts by "
                                "using model ensembling to determine if a label is a:\n",
                                **sentence_formatting,
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
                                "An example image "
                                f"({','.join({f'{k}: {v}' for k, v in self._example_image_unique_id.items()})})"
                                " is shown to the right",
                                fontsize=14,
                            ),
                        ]
                    ),
                    "right_item": temp_image_file(reallabel_results.example_image),
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
                    "layout_name": "SectionByItem",
                    "layout_arguments": {
                        "title": "Reallabel - Top Candidate Images for Relabeling Efforts",
                        "line_section_heading": "WANRS Ranking",
                        "line_section_half": True,
                        "line_section_body": [
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
                        "item_section_body": self.outputs.wanrs_df.loc[:, ["id", "WANRS"]]
                        .head(10)
                        .reset_index()
                        .assign(index=lambda df: df["index"] + 1)
                        .assign(WANRS=lambda df: df.WANRS.apply(lambda x: f"{x:.3f}"))  # format with 3 decimal places
                        .rename(columns={"index": "Priority", "id": "Image ID", "WANRS": "WANRS Score"}),
                    },
                }
            )

        return slides
