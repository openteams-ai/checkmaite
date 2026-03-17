import dataclasses
import io
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, runtime_checkable

import maite.protocols.generic as gen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pydantic
import torch
import torch.nn as nn
from dataeval import Metadata
from dataeval.performance import Sufficiency
from dataeval.protocols import AnnotatedDataset, ArrayLike, DatumMetadata, EvaluationStrategy, TrainingStrategy
from dataeval.protocols import Dataset as DatasetType
from dataeval.selection import Indices, Select
from dataeval.utils.data import split_dataset
from PIL import Image as PILImage
from pydantic import Field
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from checkmaite.core._types import Device, Image
from checkmaite.core._utils import deprecated, requires_optional_dependency, set_device
from checkmaite.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
    TDataset,
    TModel,
)
from checkmaite.core.report import _gradient as gd
from checkmaite.core.report._markdown import MarkdownOutput
from checkmaite.core.report._plotting_utils import temp_image_file

logger = logging.getLogger(__name__)

TDatum = TypeVar("TDatum")

SufficiencyDatum: TypeAlias = tuple[ArrayLike, ArrayLike, DatumMetadata]
SufficiencyTransform: TypeAlias = Callable[[SufficiencyDatum], SufficiencyDatum]
SufficiencyDataset: TypeAlias = AnnotatedDataset[SufficiencyDatum]
SufficiencyMetric: TypeAlias = gen.Metric[ArrayLike, DatumMetadata]


@dataclasses.dataclass(frozen=True)
class _SufficiencyLimits:
    min_dataset_size: int = 500
    min_samples_per_class: int = 50
    min_metric_abs_diff_ratio: float = 0.45


_DEFAULT_SUFFICIENCY_LIMITS = _SufficiencyLimits()


@runtime_checkable
class _SupportsReturnKey(Protocol):
    return_key: str


class DataevalSufficiencyConfig(CapabilityConfigBase):
    target_metric_value: float = Field(default=0.95, description="Target metric value on validation set.")
    target_metric_name: str | None = Field(
        default=None, description="Target metric name if not specified in the metric object"
    )
    extra_metrics: list[SufficiencyMetric] = Field(
        default_factory=lambda: [], description="Extra metrics to compute on validation set"
    )

    num_epochs: int | None = Field(
        default=None, description="Number epochs to use in the training strategy for sufficiency estimation."
    )
    num_iters: int | None = Field(
        default=None, description="Number iterations to use in the training strategy for sufficiency estimation."
    )
    batch_size: int = Field(
        default=32, description="Batch size to use in training strategy for sufficiency estimation."
    )
    device: Device = Field(default_factory=lambda: set_device(None))
    use_amp: bool = Field(default=True, description="If True use Automatic Mixed Precision in the training strategy.")

    sufficiency_num_runs: int = Field(default=1, description="Number of independent training runs")
    sufficiency_schedule: list[int] | None = Field(
        default=None, description="Specify this to collect metrics over a specific set of dataset lengths."
    )

    verbose: bool = Field(default=True, description="Use logger to display intermediate logging info.")

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.model_validator(mode="after")
    def _check_epochs_or_iters(self) -> "DataevalSufficiencyConfig":
        """Validate that exactly one of num_epochs or num_iters is provided.

        This ensures the training strategy has a clear termination condition.
        """
        has_epochs = self.num_epochs is not None
        has_iters = self.num_iters is not None

        if has_epochs and has_iters:
            raise ValueError(
                "Cannot specify both 'num_epochs' and 'num_iters'. " "Please provide exactly one of these parameters."
            )

        if not has_epochs and not has_iters:
            raise ValueError(
                "Must specify either 'num_epochs' or 'num_iters'. "
                "One of these parameters is required for the training strategy."
            )

        return self


class DataevalSufficiencyOutputs(CapabilityOutputsBase):
    target_metric_name: str
    target_dataset_size: int | None
    sufficiency_table: pl.DataFrame
    sufficiency_plot: Image


class DataevalSufficiencyRun(CapabilityRunBase[DataevalSufficiencyConfig, DataevalSufficiencyOutputs]):
    config: DataevalSufficiencyConfig
    outputs: DataevalSufficiencyOutputs

    # The order is important
    @requires_optional_dependency("gradient", install_hint="pip install '.[unsupported]'")
    @deprecated(replacement="collect_md_report")
    def collect_report_consumables(self, threshold: float) -> list[dict[str, Any]]:  # noqa: ARG002  # pragma: no cover
        """Collect consumables for the report.

        Gathers the sufficiency results and formats them
        for inclusion in a Gradient report.

        Parameters
        ----------
        threshold : float
            Minimum acceptable score. Results meeting or exceeding `threshold` are considered acceptable.
            Results below `threshold` require further inspection or are treated as failures.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, each representing a slide or section
            in the Gradient report.
        """
        outputs: DataevalSufficiencyOutputs = self.outputs

        deck = self.capability_id

        report_list = [generate_table_of_contents(deck)]

        report_list.append(report_sufficiency(deck, self.config.target_metric_value, outputs))

        # TODO: reactivate this when we have a next steps section from Team Aria
        # report_list.append(report_next_steps(deck))

        return report_list

    def collect_md_report(self, threshold: float) -> str:  # noqa: ARG002
        """Collect Markdown-formatted report content.

        Gathers the results from the dataset sufficiency analysis run and formats them
        as Markdown text, without dependencies on Gradient.

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
        outputs: DataevalSufficiencyOutputs = self.outputs

        md = MarkdownOutput(title="Sufficiency Analysis Report")

        generate_table_of_contents_md(md)

        md.add_section_divider()

        report_sufficiency_md(md, self.config.target_metric_value, outputs)

        return md.render()


class _DatasetToTorchDatasetAdapter(TorchDataset[TDatum], Generic[TDatum]):
    def __init__(self, dataset: DatasetType[TDatum]) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> TDatum:
        return self._dataset[index]


def _as_torch_dataset(dataset: DatasetType[TDatum]) -> TorchDataset[TDatum]:
    if isinstance(dataset, TorchDataset):
        return dataset
    return _DatasetToTorchDatasetAdapter(dataset)


class _TransformedDataset(TorchDataset[TDatum], Generic[TDatum]):
    def __init__(self, dataset: DatasetType[TDatum], transform: Callable[[TDatum], TDatum]) -> None:
        self._dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> TDatum:
        return self._transform(self._dataset[index])


class _DefaultEvaluationStrategy(Generic[TDatum]):
    # We can't use python dataclass due to dataeval config serialization internal implementation
    # dataclass becomes a dict internally and train method is discarded
    def __init__(
        self,
        metrics: list[SufficiencyMetric],
        batch_size: int,
        device: torch.device,
        num_workers: int = 4,
        verbose: bool = True,
    ) -> None:
        self.metrics = metrics
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.verbose = verbose

    def evaluate(self, model: nn.Module, dataset: DatasetType[TDatum]) -> Mapping[str, float | ArrayLike]:
        result: dict[str, float | ArrayLike] = {}
        dataloader = DataLoader(
            _as_torch_dataset(dataset),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory="cuda" in self.device.type,
        )

        model.eval()
        for metric in self.metrics:
            metric.reset()

        with torch.inference_mode():
            for batch in dataloader:
                x = batch[0].to(device=self.device)
                y = batch[1]
                preds = model(x).cpu()

                for metric in self.metrics:
                    metric.update(preds, y, ())

            for metric in self.metrics:
                result |= metric.compute()

        if self.verbose:
            logger.info(f"Evaluation results: {result}")
        return result


class _DefaultModelResetStrategy:
    def __init__(self, init_model: nn.Module) -> None:
        state_dict = init_model.state_dict()
        self.state_dict = {
            key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in state_dict.items()
        }

    def __call__(self, model: nn.Module) -> nn.Module:
        model.load_state_dict(self.state_dict)
        return model


@dataclasses.dataclass
class _MetricCurveLogModel:
    epsilon: float = 1e-6
    params: tuple[float, float, float] | None = None

    def _log_model(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.log(x + c) + b

    def _inv_log_model(self, y: float, a: float, b: float, c: float) -> float:
        return np.exp((y - b) / a) - c

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        opt_params, _ = curve_fit(self._log_model, x, y)
        self.params = tuple(opt_params.tolist())

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Please run fit method first")
        return self._log_model(x, *self.params)

    def inverse(self, y: float) -> int:
        if self.params is None:
            raise RuntimeError("Please run fit method first")
        return int(self._inv_log_model(y, *self.params))


class DataevalSufficiencyBase(
    Capability[DataevalSufficiencyOutputs, TDataset, TModel, SufficiencyMetric, DataevalSufficiencyConfig]
):
    """Estimates dataset sufficiency for a single dataset and a metric. This capability
    gives an estimate of the dataset size required at most to reach a target metric value using
    a baseline model. Metric values are computed on an evaluation subset of the given dataset.

    Note: It is possible that the target metric can be reached by more powerful models
    on smaller datasets, i.e. the size less than the estimated one by this capability.

    Generates a Gradient report with dataset sufficiency estimation and measurements,

    Attributes
    ----------
    _RUN_TYPE
        The type of the run object associated with this capability.
    """

    _RUN_TYPE = DataevalSufficiencyRun

    @classmethod
    def _create_config(cls) -> DataevalSufficiencyConfig:
        return DataevalSufficiencyConfig()

    @classmethod
    def _limits(cls) -> _SufficiencyLimits:
        return _DEFAULT_SUFFICIENCY_LIMITS

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
        return Number.ZERO

    @property
    def supports_metrics(self) -> Number:
        """Number of metrics this capability supports.

        Returns
        -------
        Number
            An enumeration value indicating metric support.
        """
        return Number.ONE

    def _get_model_and_preprocess_fns(
        self,
        num_classes: int,
        image_size: int,
    ) -> tuple[nn.Module, SufficiencyTransform, SufficiencyTransform]:
        raise NotImplementedError

    def _get_training_strategy(self, config: DataevalSufficiencyConfig) -> TrainingStrategy[SufficiencyDatum]:
        raise NotImplementedError

    def _get_evaluation_strategy(
        self,
        metric: SufficiencyMetric,
        config: DataevalSufficiencyConfig,
    ) -> EvaluationStrategy[SufficiencyDatum]:
        metrics = [metric, *config.extra_metrics]
        return _DefaultEvaluationStrategy[SufficiencyDatum](
            metrics=metrics,
            batch_size=config.batch_size,
            device=config.device,
            verbose=config.verbose,
        )

    def _split_dataset(self, dataset: SufficiencyDataset) -> tuple[SufficiencyDataset, SufficiencyDataset]:
        # If dataset is small, <1000 we split it by half into train/eval sets
        # such that evaluation set has a signicant size to produce usable
        # metric values
        num_folds = 4 if len(dataset) > 1_000 else 2
        split_defs = split_dataset(dataset, num_folds=num_folds, stratify=True)
        train_indices = split_defs.folds[0].train.tolist()
        eval_indices = split_defs.folds[0].val.tolist()
        train_dataset = Select(dataset, selections=Indices(train_indices))
        eval_dataset = Select(dataset, selections=Indices(eval_indices))
        return train_dataset, eval_dataset

    def _run(
        self,
        models: list[TModel],  # noqa: ARG002
        datasets: list[TDataset],
        metrics: list[SufficiencyMetric],
        config: DataevalSufficiencyConfig,
        use_prediction_and_evaluation_cache: bool,  # noqa: ARG002
    ) -> DataevalSufficiencyOutputs:
        """Performs dataset sufficiency analysis.

        Returns
        -------
            The outputs of the dataset sufficiency analysis.
        """
        raw_dataset = datasets[0]
        metric = metrics[0]

        if not isinstance(raw_dataset, AnnotatedDataset):
            raise TypeError("Input dataset must satisfy dataeval.protocols.AnnotatedDataset")
        dataset: SufficiencyDataset = raw_dataset

        limits = self._limits()

        min_dataset_size = limits.min_dataset_size
        if len(dataset) < min_dataset_size:
            raise ValueError(f"Input dataset is too small. Minimal dataset size is {min_dataset_size}")

        metadata = Metadata(dataset)
        class_names = list(metadata.index2label.values())
        num_classes = len(class_names)

        avg_samples_per_class = len(dataset) / num_classes
        if avg_samples_per_class < limits.min_samples_per_class:
            logger.warning(
                "Input dataset has small average number of samples per class. "
                "This may effect negatively on the result of this capability."
            )

        first_input = dataset[0][0]
        image_size = max(first_input.shape[1:])

        model, train_preprocess, eval_preprocess = self._get_model_and_preprocess_fns(
            num_classes=num_classes,
            image_size=image_size,
        )
        model = model.to(config.device)

        train_dataset, eval_dataset = self._split_dataset(dataset)
        train_dataset = _TransformedDataset(train_dataset, transform=train_preprocess)
        eval_dataset = _TransformedDataset(eval_dataset, transform=eval_preprocess)

        train_strategy = self._get_training_strategy(config)
        eval_strategy = self._get_evaluation_strategy(metric, config)

        sufficiency_config = Sufficiency.Config(
            training_strategy=train_strategy,
            evaluation_strategy=eval_strategy,
            reset_strategy=_DefaultModelResetStrategy(model),
            runs=config.sufficiency_num_runs,
        )
        sufficiency_evaluator = Sufficiency(
            model=model,
            config=sufficiency_config,
        )
        schedule = config.sufficiency_schedule
        if schedule is None:
            train_data_size = len(train_dataset)
            schedule = np.linspace(train_data_size // 10, train_data_size, num=10, dtype=int).tolist()

        output = sufficiency_evaluator.evaluate(train_dataset, eval_dataset, schedule)
        if isinstance(metric, _SupportsReturnKey):
            metric_name = metric.return_key
        else:
            metric_name = config.target_metric_name if config.target_metric_name is not None else "target_metric"
        steps = output.steps
        measurements = output.averaged_measures[metric_name]
        last_metric_measurement = measurements[-1]

        target_value = config.target_metric_value
        if abs(last_metric_measurement - target_value) / target_value > limits.min_metric_abs_diff_ratio:
            # We can't estimate data sufficiency
            logger.warning(
                "We can not estimate dataset size sufficiency to achieve target metric value. "
                "We have too small validation metric values compared to the target value. "
                "Please provide larger dataset and train for longer number of epochs."
            )
            return DataevalSufficiencyOutputs.model_validate(
                {
                    "target_metric_name": metric_name,
                    "target_dataset_size": None,
                    "sufficiency_table": output.to_dataframe(),
                    "sufficiency_plot": self._create_plot_image(metric_name, steps, measurements),
                }
            )

        curve_predictor = _MetricCurveLogModel()
        curve_predictor.fit(steps, measurements)

        target_dataset_size = curve_predictor.inverse(config.target_metric_value)
        proj_steps = np.linspace(steps[len(steps) // 2], target_dataset_size, num=15)
        proj_values = curve_predictor.transform(proj_steps)

        if target_dataset_size > 1_000_000_000:
            logger.warning(
                "Dataset size sufficiency estimation to achieve target metric value has failed. "
                "Estimated dataset size is larger than 1B samples. "
                "Please provide larger dataset and train for longer number of epochs"
            )
            target_dataset_size = None

        return DataevalSufficiencyOutputs.model_validate(
            {
                "target_metric_name": metric_name,
                "target_dataset_size": target_dataset_size,
                "sufficiency_table": output.to_dataframe(),
                "sufficiency_plot": self._create_plot_image(metric_name, steps, measurements, proj_steps, proj_values),
            }
        )

    def _create_plot_image(
        self,
        metric_name: str,
        steps: np.ndarray,
        measurements: np.ndarray,
        proj_steps: np.ndarray | None = None,
        proj_values: np.ndarray | None = None,
    ) -> Image:
        fig, ax = plt.subplots()

        ax.plot(steps, measurements, color="orange", marker="o", label="Measurements")
        if proj_steps is not None and proj_values is not None:
            ax.plot(proj_steps, proj_values, color="red", marker="*", label="Predictions")

        ax.set_xlabel("Dataset size")
        ax.set_ylabel(f"{metric_name}")
        ax.legend()
        ax.grid()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image = PILImage.open(buf)
        image.load()
        plt.close(fig)
        return image


def generate_table_of_contents(deck: str) -> dict[str, Any]:  # pragma: no cover
    """Generates a table of contents for the report.

    Returns
    -------
    dict[str, Any]
        A dictionary representing the table of contents slide.
    """

    right_item = [
        "\n",
        "* Dataset Sufficiency Analysis",
        "* Next Steps",
    ]

    left_item = gd.GradientImage(
        src=Path(__file__).parents[2] / "assets/toc.png",
        width=100,
        height=100,
        top=0.5,
        left=0.5,
    )
    return gd.create_two_item_text_slide(
        deck=deck, title="Sufficiency Table of Contents", left_item=left_item, right_item=right_item
    )


def report_sufficiency(
    deck: str, target_metric_value: float, output: DataevalSufficiencyOutputs
) -> dict[str, Any]:  # pragma: no cover
    """Format sufficiency result for Gradient consumption.

    Parameters
    ----------
    output
        The sufficiency results.

    Returns
    -------
    dict[str, Any]
        A dictionary representing the sufficiency target dataset size slide.
    """
    title = "Dataset Sufficiency Analysis"

    data_df = pd.DataFrame(
        {
            "Target dataset size": [f"{output.target_dataset_size}"],
            "Target metric name": [f"{output.target_metric_name}"],
            "Target metric value": [f"{target_metric_value}"],
        },
    )

    content = gd.Text(
        [
            gd.SubText("Description: ", bold=True),
            gd.SubText(
                "Dataset Sufficiency estimates the size of the training dataset required "
                "to achieve target metric value on an evaluation set.\n"
            ),
        ],
        fontsize=22,
    )

    return gd.create_table_text_slide(deck=deck, title=title, text=content, data=data_df)


# ============================================================================
# Markdown Report Generation Functions
# ============================================================================


def generate_table_of_contents_md(md: MarkdownOutput) -> None:
    """Generate Markdown table of contents for the dataset sufficiency report.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    """
    md.add_section(heading="Table of Contents")
    md.add_bulleted_list(
        [
            "[Sufficiency Analysis](#sufficiency-analysis)",
        ]
    )


def report_sufficiency_md(md: MarkdownOutput, target_metric_value: float, output: DataevalSufficiencyOutputs) -> None:
    """Format sufficiency results as Markdown.

    Parameters
    ----------
    md : MarkdownOutput
        The MarkdownOutput instance to add content to.
    target_metric_value : float
        Target metric value
    output : DataevalSufficiencyOutputs
        The dataset sufficiency analysis outputs.
    """
    md.add_section(heading="Dataset Sufficiency Analysis")
    md.add_text(
        "**Description:** Dataset Sufficiency estimates the size of the training dataset required "
        "to achieve target metric value on an evaluation set.\n"
    )

    md.add_subsection(heading="Summary")
    md.add_table(
        headers=["Description", "Value"],
        rows=[
            ["Target dataset size", f"{output.target_dataset_size}"],
            ["Target metric name", f"{output.target_metric_name}"],
            ["Target metric value", f"{target_metric_value}"],
        ],
    )

    md.add_subsection(heading="Sufficiency estimation details")

    md.add_table(dataframe=output.sufficiency_table)

    img_path = temp_image_file(output.sufficiency_plot)
    md.add_image(img_path, alt_text="Sufficiency Visualization")
