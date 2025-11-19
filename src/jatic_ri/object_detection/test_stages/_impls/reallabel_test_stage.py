"""Object Detection RealLabel test stage."""

import tempfile
import textwrap
import warnings
from pathlib import Path
from typing import Any, cast

import maite.protocols.object_detection as od
import numpy as np
import PIL.Image
import pyspark.sql.functions as sf
import torch
from gradient import SubText, Text
from maite.protocols import ArrayLike, DatasetMetadata, DatumMetadata
from pptx.dml.color import RGBColor
from pydantic import model_validator
from reallabel import (
    MAITERealLabel,
    RealLabelColumns,
    plot_reallabel_results,
)
from reallabel import RealLabelConfig as _NativeRealLabelConfig

from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, Number, OutputsBase, RunBase, TestStage
from jatic_ri.cached_tasks import predict
from jatic_ri.util._types import DataFrame, Image
from jatic_ri.util.utils import temp_image_file


class RealLabelImageOutput(OutputsBase):
    """Example image result for RealLabelTestStageResults.

    Attributes
    ----------
    image : Image
        The example image.
    id : dict[str, Any]
        The ID of the example image.
    """

    image: Image
    id: dict[str, Any]


class RealLabelOutputs(OutputsBase):
    """Results class for RealLabelTestStage.

    Attributes
    ----------
    results : DataFrame
        The main RealLabel results DataFrame.
    example_image : RealLabelImageOutput
        An example image output.
    classification_disagreements_df : DataFrame | None, optional
        DataFrame of classification disagreements. Defaults to None.
    verbose_df : DataFrame | None, optional
        Verbose DataFrame with detailed information. Defaults to None.
    sequence_priority_score_df : DataFrame | None, optional
        DataFrame of sequence priority scores. Defaults to None.
    sequence_priority_score_balanced_df : DataFrame | None, optional
        DataFrame of balanced sequence priority scores. Defaults to None.
    wanrs_df : DataFrame | None, optional
        DataFrame of WANRS scores. Defaults to None.
    aggregated_confidence_df : DataFrame | None, optional
        DataFrame of aggregated confidence scores. Defaults to None.
    """

    results: DataFrame
    example_image: RealLabelImageOutput
    classification_disagreements_df: DataFrame | None = None
    verbose_df: DataFrame | None = None
    sequence_priority_score_df: DataFrame | None = None
    sequence_priority_score_balanced_df: DataFrame | None = None
    wanrs_df: DataFrame | None = None
    aggregated_confidence_df: DataFrame | None = None


# reallabel already provides a pydantic model for its configuration so we just mix in our base
class RealLabelConfig(_NativeRealLabelConfig, ConfigBase):
    """Config class for RealLabelTestStage."""

    @model_validator(mode="after")
    def _ensure_aggregated_confidence(self) -> "RealLabelConfig":
        # Need AC for visualization. Add it to the config if it's not provided.
        # This is a RealLabel oversight that we need to fix.
        ac = RealLabelColumns.AGGREGATED_CONFIDENCE.value

        if ac not in self.additional_columns_clean_results:
            self.additional_columns_clean_results.append(ac)

        return self


class RealLabelRun(RunBase):
    """Run class for RealLabelTestStage.

    Attributes
    ----------
    config : RealLabelConfig
        The configuration for the RealLabel run.
    outputs : RealLabelOutputs
        The outputs of the RealLabel run.
    """

    config: RealLabelConfig
    outputs: RealLabelOutputs

    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:  # noqa: ARG002
        """Collect all report consumables.

        Parameters
        ----------
        threshold : float
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, where each dictionary represents a slide
            for the report.
        """

        reallabel_results = self.outputs
        default_results_df = reallabel_results.results

        # Find RealLabel statistics
        num_false_positives = (default_results_df["reallabel_type"] == "Likely Wrong").sum()
        num_false_negatives = (default_results_df["reallabel_type"] == "Likely Missed").sum()
        num_true_positives = (default_results_df["reallabel_type"] == "Likely Correct").sum()

        sentence_formatting: dict[str, Any] = {
            "fontsize": 18,
            "bold": True,
        }

        bullet_formatting: dict[str, Any] = {
            "fontsize": 16,
        }

        spaces_formatting: dict[str, Any] = {
            "fontsize": 12,
        }

        def bullet_point(text: str, **text_formatting_kwargs: Any) -> tuple[SubText, SubText]:
            return (
                SubText("• ", fontsize=22),
                SubText(text, **{**bullet_formatting, **text_formatting_kwargs}),
            )

        slides = [
            {
                "deck": self.test_stage_id,
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
                            SubText("\n" * 1, **spaces_formatting),
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
                                **bullet_formatting,
                                bold=True,
                            ),
                            *bullet_point(
                                "False Positive: ",
                                color=RGBColor(72, 127, 199),
                                bold=True,
                            ),
                            SubText(
                                f"{num_false_positives}\n",
                                **bullet_formatting,
                                bold=True,
                            ),
                            *bullet_point(
                                "False Negative: ",
                                color=RGBColor(221, 0, 0),
                                bold=True,
                            ),
                            SubText(
                                f"{num_false_negatives}\n",
                                **bullet_formatting,
                                bold=True,
                            ),
                            SubText("\n" * 1, **spaces_formatting),
                            SubText(
                                "An example image "
                                f"({','.join({f'{k}: {v}' for k, v in reallabel_results.example_image.id.items()})})"
                                " is shown to the right",
                                fontsize=14,
                            ),
                        ]
                    ),
                    "right_item": temp_image_file(reallabel_results.example_image.image),
                },
            },
        ]

        if reallabel_results.wanrs_df is not None:
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
                    "deck": self.test_stage_id,
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
                                        content="RealLabel documentation",
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
                        "item_section_body": reallabel_results.wanrs_df.loc[:, ["id", "WANRS"]]
                        .head(10)
                        .reset_index()
                        .assign(index=lambda df: df["index"] + 1)
                        .assign(WANRS=lambda df: df.WANRS.apply(lambda x: f"{x:.3f}"))  # format with 3 decimal places
                        .rename(columns={"index": "Priority", "id": "Image ID", "WANRS": "WANRS Score"}),
                    },
                }
            )

        return slides


class _RealLabelDatasetWrapper(od.Dataset):
    """A wrapper for the MAITE Dataset for use with RealLabel.

    This wrapper allows us to copy a generic MAITE Dataset and easily update the attributes
    (e.g., targets) which are not part of the MAITE Dataset protocol.
    """

    def __init__(self, dataset: od.Dataset) -> None:
        """Initialize the RealLabelDatasetWrapper.

        Parameters
        ----------
        dataset : od.Dataset
            The MAITE dataset to wrap.
        """
        self.metadata: DatasetMetadata = dataset.metadata
        self.images: list[ArrayLike] = []
        self.targets: list[od.ObjectDetectionTarget] = []
        self.datum_metadata: list[od.DatumMetadataType] = []

        for image, target, metadata in dataset:
            self.images.append(image)
            self.targets.append(target)
            self.datum_metadata.append(metadata)

    def __getitem__(self, ind: int) -> tuple[ArrayLike, od.ObjectDetectionTarget, DatumMetadata]:
        return (self.images[ind], self.targets[ind], self.datum_metadata[ind])

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The number of items in the dataset.
        """
        return len(self.images)


class RealLabelTestStage(
    TestStage[RealLabelOutputs, od.Dataset, od.Model, od.Metric, RealLabelConfig],
):
    """RealLabel test stage.

    RealLabel uses an ensemble of models and their inference results to provide insight into the potential
    correctness or incorrectness of ground truth labels, and which ones may need to be re-examined or re-labeled.

    For more info see our docs! https://jatic.pages.jatic.net/morse/reallabel/

    This test stage also uses MAITE-wrapped models and datasets, and MAITE itself,
    to produce the model inference results needed if they are not present in the
    cache before running RealLabel itself.
    """

    _RUN_TYPE = RealLabelRun

    @classmethod
    def _create_config(cls) -> RealLabelConfig:
        return RealLabelConfig()

    @property
    def supports_datasets(self) -> Number:
        """Number of datasets this test stage supports.

        Returns
        -------
        Number
            An enumeration value indicating dataset support.
        """
        return Number.ONE

    @property
    def supports_models(self) -> Number:
        """Number of models this test stage supports.

        Returns
        -------
        Number
            An enumeration value indicating model support.
        """
        return Number.MANY

    @property
    def supports_metrics(self) -> Number:
        """Number of models this test stage supports.

        Returns
        -------
        Number
            An enumeration value indicating metric support.
        """
        return Number.ZERO

    def _run(
        self,
        models: list[od.Model],
        datasets: list[od.Dataset],
        metrics: list[od.Metric],  # noqa: ARG002
        config: RealLabelConfig,
    ) -> RealLabelOutputs:
        """Run RealLabel test stage.

        Returns
        -------
        RealLabelOutputs
            The results of the RealLabel analysis.

        Raises
        ------
        RuntimeError
            If the RealLabel result pyspark DataFrame is empty.
        ValueError
            If the example image cannot be found in the dataset.
        RuntimeError
            If the image array type is not understood.
        """

        dataset = datasets[0]

        # compute the inference results from the models
        maite_inference_result = self._compute_maite_inference_result(models=models, dataset=dataset)

        # Create MAITERealLabel wrapper
        reallabel = MAITERealLabel(
            maite_inference_results=maite_inference_result,
            config=config,
            link_unique_identifier_columns_and_metadata=True,
            ground_truth_dataset=dataset,
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
        example_image_unique_id = (
            default_reallabel_results.groupBy(config.column_names.unique_identifier_columns)
            .count()
            .orderBy(sf.col("count").desc())
            .drop("count")
            .first()
            .asDict()
        )

        results_for_most_populous_image_df = default_reallabel_results.filter(
            *[sf.col(key) == value for key, value in example_image_unique_id.items()]
        )

        # Iterate through the dataset until we find the index with the matching UUID. Get the Image tensor from it.
        most_populous_image_array = None
        for image_tensor, _, image_metadata in dataset:
            if all(str(image_metadata[key]) == str(value) for key, value in example_image_unique_id.items()):
                most_populous_image_array = image_tensor
                break
        if most_populous_image_array is None:
            raise ValueError("Where's the image?!")

        # Since we aren't guaranteed an actual file name metadata field (not all datasets have them),
        # make a new file name with the raw UUID values
        most_populous_image_file_name = "_".join(value for value in example_image_unique_id.values()) + ".png"
        # Ensure we have a mapping of this portmanteau file name to the actual UUID values that
        # will be found in the dataframe.
        most_populous_image_name_to_uuid_value_map = {most_populous_image_file_name: example_image_unique_id}

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
                reallabel_config=config,
                image_name_map=most_populous_image_name_to_uuid_value_map,
            )

            return RealLabelOutputs(
                results=default_reallabel_results.toPandas(),
                example_image=RealLabelImageOutput(image=output_location, id=example_image_unique_id),  # pyright: ignore [reportArgumentType]
                classification_disagreements_df=reallabel_results.classification_disagreements_df.toPandas()
                if reallabel_results.classification_disagreements_df
                else None,
                verbose_df=reallabel_results.verbose_df.toPandas() if reallabel_results.verbose_df else None,
                sequence_priority_score_df=reallabel_results.sequence_priority_score_df.toPandas()
                if reallabel_results.sequence_priority_score_df
                else None,
                sequence_priority_score_balanced_df=reallabel_results.sequence_priority_score_balanced_df.toPandas()
                if reallabel_results.sequence_priority_score_balanced_df
                else None,
                wanrs_df=reallabel_results.wanrs_df.toPandas() if reallabel_results.wanrs_df else None,
                aggregated_confidence_df=reallabel_results.aggregated_confidence_df.toPandas()
                if reallabel_results.aggregated_confidence_df
                else None,
            )

    def _compute_maite_inference_result(self, models: list[od.Model], dataset: od.Dataset) -> dict[str, od.Dataset]:
        """Generate inference results from MAITE models.

        Returns
        -------
        A dictionary mapping model names to datasets containing their predictions
        as targets.
        """
        maite_inference_result = {}
        # Run all the models
        for model in models:
            predictions, _ = predict(
                model=model,
                dataset=dataset,
                dataset_id=dataset.metadata["id"],
                return_augmented_data=True,
            )
            clean_predictions = [x[0] for x in predictions]

            # We need to construct a new dataset with the ground truth targets replaced with the predictions.
            # MAITE Dataset does not have a way to set the targets directly, so we use the wrapper instance.
            copied_dataset = _RealLabelDatasetWrapper(dataset)

            # there is some pydantic shenanigans going on here which messes with the type-checker, but
            # we are confident that it's correct and so we override with a cast
            copied_dataset.targets = cast(list[od.TargetType], clean_predictions)

            # RealLabel will only use certain metadata passed in here (e.g. for the confidence calibration) which is not
            # supported through Reference Implementation currently. Additionally, metadata specific to the ground
            # truth data (e.g. ground truth bounding boxes) can cause errors in RealLabel so we remove all
            # non-mandatory datum_metadata keys.
            copied_dataset.datum_metadata = [{"id": dm["id"]} for dm in copied_dataset.datum_metadata]

            maite_inference_result[model.metadata["id"]] = copied_dataset

        return maite_inference_result
