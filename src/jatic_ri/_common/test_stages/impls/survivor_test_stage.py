from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Union

import numpy as np
import pydantic
import pyspark.sql
import pyspark.sql.functions as sf
from gradient import SubText, Text
from maite import protocols as pr
from pyspark.sql import SparkSession
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
    pass


class SurvivorOutputs(OutputsBase):
    results: DataFrame
    image: Image

    # TODO: remove this if all consumers are adapted to the new outputs type
    def __getitem__(self, index: int) -> Any:
        if index == 0:
            return self.results
        if index == 1:
            return self.image
        raise IndexError("index out of range")

    # TODO: remove this if all consumers are adapted to the new outputs type
    def __iter__(self) -> Iterator[Any]:  # type: ignore
        yield self.results
        yield self.image


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
        config: Union[_NativeSurvivorConfig, dict[str, Any]],
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
        df = survivor.run().raw_output_df  # noqa: PD901

        with tempfile.TemporaryDirectory() as output_dir:
            histogram_plot = HistogramBarPlot(
                title="Image count by type",
                col="suid_label",
                output_dir=Path(output_dir),
            )
            plot = histogram_plot.plot(df)
            image = plot[0][0][0].figure

            return SurvivorOutputs(results=df, image=image)  # type: ignore[reportArgumentType]

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect report consumables for the SurvivorTestStage."""
        survivor_results_pd, output_image = self.outputs

        survivor_results = pyspark.sql.SparkSession.active().createDataFrame(survivor_results_pd)

        total = survivor_results.count()
        hard_proportion = survivor_results.where(sf.col("suid_label") == "Hard").count() / total
        easy_proportion = survivor_results.where(sf.col("suid_label") == "Easy").count() / total
        otb_proportion = survivor_results.where(sf.col("suid_label") == "On the Bubble").count() / total

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
        return [
            {
                "deck": self._deck,
                "layout_name": "TwoItem",
                "layout_arguments": {
                    "title": "Survivor Dataset Breakdown",
                    "left_item": definition_text,
                    "right_item": temp_image_file(output_image),
                },
            },
        ]
