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

from jatic_ri._common.test_stages.interfaces.plugins import (
    EvalToolPlugin,
    MetricPlugin,
    MultiModelPlugin,
    SingleDatasetPlugin,
)
from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, OutputsBase, RunBase, TestStage
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
        """Validate heatmap plot columns."""
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
    TestStage[SurvivorOutputs],
    MultiModelPlugin,
    SingleDatasetPlugin,
    MetricPlugin,
    EvalToolPlugin,
):
    _RUN_TYPE = SurvivorRun

    _deck: str
    _task: str

    def __init__(
        self,
        config: _NativeSurvivorConfig | dict[str, Any],
    ) -> None:
        """Create instance of SurvivorTestStage

        Args:
            config: config for survivor run.
        """

        super().__init__()

        if isinstance(config, pydantic.BaseModel):
            config = config.model_dump()
        self._config = SurvivorConfig.model_validate(config)

    def _create_config(self) -> SurvivorConfig:
        return self._config

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

    def _label_count_plot(self, raw_output_df: pyspark.sql.DataFrame) -> Figure:
        """Create a histogram plot of the label counts."""
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

    def _run(self) -> SurvivorOutputs:
        """Run Survivor if no cached results are found."""
        self.validate_plugins()

        # run metrics
        metrics = self.__run_metrics()

        # create maite wrapper and run Survivor
        survivor = MAITESurvivor(
            maite_dataset=self.dataset,
            config=self._config,
            metrics=metrics,
            spark_session=SparkSession.builder.getOrCreate(),  # type: ignore
        )

        output_data = survivor.run()
        df = output_data.raw_output_df  # noqa: PD901

        label_count_plot = self._label_count_plot(df)
        heatmap_plots = []
        for metadata_column in self._config.heatmap_plot_columns:  # type: ignore[reportOptionalIterable]
            heatmap_plot = self._heatmap_plot(output_data.metrics_with_survivor_label_df, metadata_column)
            heatmap_plots.append(heatmap_plot)

        return SurvivorOutputs(
            raw_output_df=output_data.raw_output_df,  # type: ignore[reportArgumentType]
            metrics_with_survivor_label_df=output_data.metrics_with_survivor_label_df,  # type: ignore[reportArgumentType]
            label_count_plot=label_count_plot,  # type: ignore[reportArgumentType]
            heatmap_plots=heatmap_plots,
        )

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect report consumables for the SurvivorTestStage."""
        survivor_results = self.outputs.raw_output_df

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
            for plot in self.outputs.heatmap_plots
        ]

        return [
            {
                "deck": self._deck,
                "layout_name": "TwoItem",
                "layout_arguments": {
                    "title": "Survivor Dataset Breakdown",
                    "left_item": definition_text,
                    "right_item": temp_image_file(self.outputs.label_count_plot),
                },
            },
            *heatmap_slides,
        ]
