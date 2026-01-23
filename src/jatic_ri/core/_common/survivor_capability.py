import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pydantic
import pyspark.sql
from maite import protocols as pr
from matplotlib.figure import Figure
from pydantic import Field
from pyspark.sql import SparkSession
from survivor import HeatmapPlot
from survivor.analysis import HistogramBarPlot
from survivor.config import SurvivorConfig as _NativeSurvivorConfig
from survivor.maite_survivor import MAITESurvivor

from jatic_ri.core._types import DataFrame, Image
from jatic_ri.core._utils import deprecated, requires_optional_dependency
from jatic_ri.core.cached_tasks import evaluate_from_predictions, predict
from jatic_ri.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
    TDataset,
    TMetric,
    TModel,
)
from jatic_ri.core.report import _gradient as gd
from jatic_ri.core.report._markdown import MarkdownOutput
from jatic_ri.core.report._plotting_utils import temp_image_file


# survivor already provides a pydantic model for its configuration so we just mix in our base
class SurvivorConfig(_NativeSurvivorConfig, CapabilityConfigBase):
    heatmap_plot_columns: list[str] | None = pydantic.Field(
        description="Metadata columns for which to create resulting heatmaps for each label",
        default=[],
    )

    batch_size: int = Field(default=1, description="Batch size used for model inference.")

    @pydantic.field_validator("heatmap_plot_columns")
    @classmethod
    def validate_heatmap_plot_columns(cls, value: list[str] | None) -> list[str]:
        """Validate heatmap plot columns.

        Parameters
        ----------
        value
            The input list of heatmap plot columns.

        Returns
        -------
            The validated list of heatmap plot columns, defaulting to an empty list if None.
        """
        if value is None:
            return []

        return value


class SurvivorOutputs(CapabilityOutputsBase):
    raw_output_df: DataFrame
    metrics_with_survivor_label_df: DataFrame
    label_count_plot: Image
    heatmap_plots: list[Image]


class SurvivorRun(CapabilityRunBase[SurvivorConfig, SurvivorOutputs]):
    config: SurvivorConfig
    outputs: SurvivorOutputs

    # The order is important
    @requires_optional_dependency("gradient", install_hint="pip install '.[unsupported]'")
    @deprecated(replacement="collect_md_report")
    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:  # noqa: ARG002 # pragma: no cover
        """Collect report consumables for Survivor.

        This method formats the Survivor analysis results into a list of
        dictionaries suitable for generating a Gradient report.

        Parameters
        ----------
        threshold
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
            A list of dictionaries, where each dictionary represents a slide
            for the Gradient report.
        """

        outputs = self.outputs

        survivor_results = outputs.raw_output_df

        label_counts = survivor_results.suid_label.value_counts()
        total = label_counts.sum()

        hard_proportion = label_counts.get("Hard", 0) / total
        easy_proportion = label_counts.get("Easy", 0) / total
        otb_proportion = label_counts.get("On the Bubble", 0) / total

        # Simply return the title of the section and the data to be plotted
        definition_text = gd.Text(
            [
                gd.SubText("Types of Data\n", bold=True),
                f"• Easy: Models score the same and perform well.\n"
                f"• Hard: Models score the same and perform poorly.\n"
                f"• On the Bubble: Models score differently.\n\n"
                f"• Ideally, a dataset would be primarily On the Bubble, "
                "so all data is helping distinguish between model performance.\n\n"
                f"• This dataset had {easy_proportion * 100:.1f}% Easy, {hard_proportion * 100:.1f}% Hard, and "
                f"{otb_proportion * 100:.1f}% On the Bubble data.",
            ],
            fontsize=22,
        )

        heatmap_slides = [
            {
                "deck": self.capability_id,
                "layout_name": "ItemByNarrowText",
                "title": "Survivor Metadata Heatmap",
                "layout_arguments": {
                    "title": "Survivor Metadata Heatmap",
                    "item": temp_image_file(plot),
                    "text": "",
                },
            }
            for plot in outputs.heatmap_plots
        ]

        return [
            {
                "deck": self.capability_id,
                "layout_name": "TwoItem",
                "layout_arguments": {
                    "title": "Survivor Dataset Breakdown",
                    "left_item": definition_text,
                    "right_item": temp_image_file(outputs.label_count_plot),
                },
            },
            *heatmap_slides,
        ]

    def collect_md_report(self, threshold: float) -> str:  # noqa: ARG002
        """Generate Markdown-formatted report for Survivor analysis.

        Parameters
        ----------
        threshold : float
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
        str
            Markdown-formatted report content.
        """
        outputs = self.outputs
        survivor_results = outputs.raw_output_df

        label_counts = survivor_results.suid_label.value_counts()
        total = label_counts.sum()

        hard_proportion = label_counts.get("Hard", 0) / total
        easy_proportion = label_counts.get("Easy", 0) / total
        otb_proportion = label_counts.get("On the Bubble", 0) / total

        md = MarkdownOutput("Survivor Dataset Analysis")

        md.add_section(heading="Dataset Breakdown")
        md.add_subsection(heading="Types of Data")
        md.add_text("**Easy**: Models score the same and perform well.")
        md.add_text("**Hard**: Models score the same and perform poorly.")
        md.add_text("**On the Bubble**: Models score differently.")
        md.add_blank_line()
        md.add_text(
            'Ideally, a dataset would be primarily "On the Bubble", '
            "so all data is helping distinguish between model performance."
        )

        md.add_subsection(heading="Results")
        md.add_table(
            headers=["Category", "Count", "Percentage"],
            rows=[
                [
                    "Easy",
                    str(label_counts.get("Easy", 0)),
                    f"{easy_proportion * 100:.1f}%",
                ],
                [
                    "Hard",
                    str(label_counts.get("Hard", 0)),
                    f"{hard_proportion * 100:.1f}%",
                ],
                [
                    "On the Bubble",
                    str(label_counts.get("On the Bubble", 0)),
                    f"{otb_proportion * 100:.1f}%",
                ],
                ["**Total**", f"**{total}**", "**100%**"],
            ],
        )
        md.add_image(temp_image_file(outputs.label_count_plot), alt_text="Dataset Breakdown")

        # Add heatmap plots
        if outputs.heatmap_plots:
            md.add_section(heading="Metadata Heatmaps")
            for idx, plot in enumerate(outputs.heatmap_plots, 1):
                md.add_subsection(f"Heatmap {idx}")
                md.add_image(temp_image_file(plot), alt_text=f"Metadata Heatmap {idx}")

        return md.render()


class SurvivorBase(
    Capability[SurvivorOutputs, TDataset, TModel, TMetric, SurvivorConfig],
):
    _RUN_TYPE = SurvivorRun

    @classmethod
    def _create_config(cls) -> SurvivorConfig:
        return SurvivorConfig(metric_column="")  # pyright: ignore[reportCallIssue]

    @property
    def supports_datasets(self) -> Number:
        """Number of datasets this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating dataset support.
        """
        return Number.ONE

    @property
    def supports_models(self) -> Number:
        """Number of models this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating model support.
        """
        return Number.MANY

    @property
    def supports_metrics(self) -> Number:
        """Number of metrics this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating metric support.
        """
        return Number.ONE

    def _run(
        self,
        models: list[TModel],
        datasets: list[TDataset],
        metrics: list[TMetric],
        config: SurvivorConfig,
        use_prediction_and_evaluation_cache: bool,
    ) -> SurvivorOutputs:
        """Run Survivor analysis.

        This method executes the MAITESurvivor analysis, generating
        raw output, metrics with survivor labels, a label count plot,
        and heatmap plots based on the configuration.

        Returns
        -------
        SurvivorOutputs
            An object containing the results of the Survivor analysis.
        """

        dataset = datasets[0]
        metric = metrics[0]

        survivor_metrics = self._run_survivor_metrics(
            models=models,
            dataset=dataset,
            metric=metric,
            config=config,
            use_prediction_and_evaluation_cache=use_prediction_and_evaluation_cache,
        )

        survivor = MAITESurvivor(
            maite_dataset=dataset,
            config=config,
            metrics=survivor_metrics,
            spark_session=SparkSession.builder.getOrCreate(),  # pyright: ignore[reportAttributeAccessIssue]
        )

        output_data = survivor.run()

        label_count_plot = self._label_count_plot(output_data.raw_output_df)
        heatmap_plots = []
        for metadata_column in config.heatmap_plot_columns:  # pyright: ignore[reportOptionalIterable]
            heatmap_plot = self._heatmap_plot(output_data.metrics_with_survivor_label_df, metadata_column)
            heatmap_plots.append(heatmap_plot)

        return SurvivorOutputs(
            raw_output_df=output_data.raw_output_df.toPandas(),
            metrics_with_survivor_label_df=output_data.metrics_with_survivor_label_df.toPandas(),
            label_count_plot=label_count_plot,  # pyright: ignore[reportArgumentType]
            heatmap_plots=heatmap_plots,
        )

    def _run_survivor_metrics(
        self,
        models: list[TModel],
        dataset: TDataset,
        metric: TMetric,
        config: SurvivorConfig,
        use_prediction_and_evaluation_cache: bool,
    ) -> dict[str, pr.ArrayLike]:
        """Create metrics by model for use in MAITESurvivor.

        This method calculates metrics for each model on a per-datum basis.
        It leverages cached predictions if available.

        Returns
        -------
        dict[str, pr.ArrayLike]
            A dictionary where keys are model IDs and values are numpy arrays
            of metric results for each datum.
        """
        all_model_metrics_per_datum: dict[str, pr.ArrayLike] = {}
        for model in models:
            model_metrics_per_datum = []
            # Since Survivor's implementation is unique in that it runs metric calculations on each individual in
            # a dataset, we first call the `predict()` method using the model against the entire dataset.
            # Otherwise, we could not leverage a cached result from a prior capability's prediction generation.
            predictions, targets = predict(
                model=model,
                dataset=dataset,
                batch_size=config.batch_size,
                return_augmented_data=True,
                use_cache=use_prediction_and_evaluation_cache,
            )
            # With the predictions for each target now in memory, we run only `compute_metric()` against each item.
            # Calling `evaluate()` to calculate the metrics would require doing model predictions again with each
            # target as its own dataset.
            for i, datum_prediction in enumerate(predictions):
                results: dict[str, Any] = evaluate_from_predictions(
                    metric=metric,
                    predictions=[datum_prediction],
                    targets=[targets[i][1]],
                    use_cache=use_prediction_and_evaluation_cache,
                )
                model_metrics_per_datum.append(results[config.metric_column].numpy())

                # Reset the metric for the next evaluate() call so results don't bleed together.
                metric.reset()

            # Take the aggregated datum results from this model and turn it into a 1-d colum numpy array for
            # later consumption.
            all_model_metrics_per_datum[model.metadata["id"]] = np.hstack(model_metrics_per_datum)

        return all_model_metrics_per_datum

    def _label_count_plot(self, raw_output_df: pyspark.sql.DataFrame) -> Figure:
        """Create a histogram plot of the label counts.

        Parameters
        ----------
        raw_output_df : pyspark.sql.DataFrame
            The raw output DataFrame from the MAITESurvivor run.

        Returns
        -------
        matplotlib.figure.Figure
            A matplotlib Figure object representing the histogram plot.
        """
        # Create a histogram plot of the label counts
        with tempfile.TemporaryDirectory() as output_dir:
            histogram_plot = HistogramBarPlot(
                title="Image count by type",
                col="suid_label",
                output_dir=Path(output_dir),
            )
            plot = histogram_plot.plot(raw_output_df)

        return plot[0][0][0].figure

    def _heatmap_plot(
        self,
        metrics_with_survivor_label_df: pyspark.sql.DataFrame,
        metadata_field: str,
    ) -> Figure:
        """Create a heatmap plot for a given metadata field.

        Parameters
        ----------
        metrics_with_survivor_label_df : pyspark.sql.DataFrame
            DataFrame containing metrics and survivor labels.
        metadata_field : str
            The metadata field for which to create the heatmap.

        Returns
        -------
        matplotlib.figure.Figure
            A matplotlib Figure object representing the heatmap plot.
        """
        with tempfile.TemporaryDirectory() as output_dir:
            heatmap_plot = HeatmapPlot(
                title=f"{metadata_field} Distribution",
                y_col=metadata_field,
                output_dir=Path(output_dir),
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message="The default of observed=False is deprecated and will be changed to True "
                    "in a future version of pandas.",
                )
                named_figure, _ = heatmap_plot.plot(metrics_with_survivor_label_df)[0]

        return named_figure.figure
