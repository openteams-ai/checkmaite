import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pydantic
import pyspark.sql
from gradient import SubText, Text
from maite import protocols as pr
from matplotlib.figure import Figure
from pyspark.sql import SparkSession
from survivor import HeatmapPlot
from survivor.analysis import HistogramBarPlot
from survivor.config import SurvivorConfig as _NativeSurvivorConfig
from survivor.maite_survivor import MAITESurvivor

from jatic_ri._common.test_stages.interfaces.test_stage import (
    ConfigBase,
    Number,
    OutputsBase,
    RunBase,
    TDataset,
    TestStage,
    TMetric,
    TModel,
)
from jatic_ri.cached_tasks import evaluate_from_predictions, predict
from jatic_ri.util._types import DataFrame, Image
from jatic_ri.util.utils import temp_image_file


# survivor already provides a pydantic model for its configuration so we just mix in our base
class SurvivorConfig(_NativeSurvivorConfig, ConfigBase):
    heatmap_plot_columns: list[str] | None = pydantic.Field(
        description="Metadata columns for which to create resulting heatmaps for each label",
        default=[],
    )

    @pydantic.field_validator("heatmap_plot_columns")
    @classmethod
    def validate_heatmap_plot_columns(cls, value: list[str] | None) -> list[str]:
        """Validate heatmap plot columns.

        Parameters
        ----------
        value : list[str] | None
            The input list of heatmap plot columns.

        Returns
        -------
        list[str]
            The validated list of heatmap plot columns, defaulting to an empty list if None.
        """
        if value is None:
            return []

        return value


class SurvivorOutputs(OutputsBase):
    raw_output_df: DataFrame
    metrics_with_survivor_label_df: DataFrame
    label_count_plot: Image
    heatmap_plots: list[Image]


class SurvivorRun(RunBase):
    config: SurvivorConfig
    outputs: SurvivorOutputs


class SurvivorTestStageBase(
    TestStage[SurvivorOutputs, TDataset, TModel, TMetric],
):
    _RUN_TYPE = SurvivorRun

    _deck: str
    _task: str

    def __init__(
        self,
        config: _NativeSurvivorConfig | dict[str, Any],
    ) -> None:
        """Create instance of SurvivorTestStage.

        Parameters
        ----------
        config : _NativeSurvivorConfig | dict[str, Any]
            Configuration for the survivor run. Can be a SurvivorConfig object
            or a dictionary.
        """
        super().__init__()

        if isinstance(config, pydantic.BaseModel):
            config = config.model_dump()
        self._config = SurvivorConfig.model_validate(config)

    def _create_config(self) -> SurvivorConfig:
        return self._config

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
        """Number of metrics this test stage supports.

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

        survivor_metrics = self._run_survivor_metrics(models=models, dataset=dataset, metric=metric)

        survivor = MAITESurvivor(
            maite_dataset=dataset,
            config=self._config,
            metrics=survivor_metrics,
            spark_session=SparkSession.builder.getOrCreate(),  # pyright: ignore[reportAttributeAccessIssue]
        )

        output_data = survivor.run()

        label_count_plot = self._label_count_plot(output_data.raw_output_df)
        heatmap_plots = []
        for metadata_column in self._config.heatmap_plot_columns:  # pyright: ignore[reportOptionalIterable]
            heatmap_plot = self._heatmap_plot(output_data.metrics_with_survivor_label_df, metadata_column)
            heatmap_plots.append(heatmap_plot)

        return SurvivorOutputs(
            raw_output_df=output_data.raw_output_df.toPandas(),
            metrics_with_survivor_label_df=output_data.metrics_with_survivor_label_df.toPandas(),
            label_count_plot=label_count_plot,  # pyright: ignore[reportArgumentType]
            heatmap_plots=heatmap_plots,
        )

    def _run_survivor_metrics(
        self, models: list[TModel], dataset: TDataset, metric: TMetric
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
            # Otherwise, we could not leverage a cached result from a prior test stage's prediction generation.
            predictions, targets = predict(
                model=model,
                dataset=dataset,
                batch_size=self._batch_size,
                return_augmented_data=True,
            )
            # With the predictions for each target now in memory, we run only `compute_metric()` against each item.
            # Calling `evaluate()` to calculate the metrics would require doing model predictions again with each
            # target as its own dataset.
            for i, datum_prediction in enumerate(predictions):
                results: dict[str, Any] = evaluate_from_predictions(
                    metric=metric,
                    predictions=[datum_prediction],
                    targets=[targets[i][1]],
                )
                model_metrics_per_datum.append(results[metric.metadata["id"]].numpy())

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
                named_figure, path = heatmap_plot.plot(metrics_with_survivor_label_df)[0]

        return named_figure.figure

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect report consumables for the SurvivorTestStage.

        This method formats the Survivor analysis results into a list of
        dictionaries suitable for generating a Gradient report.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, where each dictionary represents a slide
            for the Gradient report.

        Raises
        ------
        RuntimeError
            If the TestStage has not been run before calling this method.
        """
        if self._stored_run is None:
            raise RuntimeError("TestStage must be run before accessing outputs")
        outputs = self._stored_run.outputs

        survivor_results = outputs.raw_output_df

        label_counts = survivor_results.suid_label.value_counts()
        total = label_counts.sum()

        hard_proportion = label_counts.get("Hard", 0) / total
        easy_proportion = label_counts.get("Easy", 0) / total
        otb_proportion = label_counts.get("On the Bubble", 0) / total

        # Simply return the title of the section and the data to be plotted
        definition_text = Text(
            [
                SubText("Types of Data\n", bold=True),
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
                "deck": self._deck,
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
                "deck": self._deck,
                "layout_name": "TwoItem",
                "layout_arguments": {
                    "title": "Survivor Dataset Breakdown",
                    "left_item": definition_text,
                    "right_item": temp_image_file(outputs.label_count_plot),
                },
            },
            *heatmap_slides,
        ]
