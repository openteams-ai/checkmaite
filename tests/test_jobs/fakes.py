from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from checkmaite.core.analytics_store import BaseRecord
from checkmaite.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
)
from checkmaite.core.report import MAX_INLINE_REPORT_BYTES, InlineTextReport


class TinyRecord(BaseRecord, table_name="tiny_jobs"):
    payload: str


class TinyConfig(CapabilityConfigBase):
    text: str = "ok"
    sleep_s: float = 0.0
    fail: bool = False
    env_key: str | None = None
    num_cpus: int | None = None
    num_gpus: float | None = None
    start_marker_path: str | None = None
    finish_marker_path: str | None = None


class TinyOutputs(CapabilityOutputsBase):
    text: str


class TinyRun(CapabilityRunBase[TinyConfig, TinyOutputs]):
    config: TinyConfig
    outputs: TinyOutputs

    def collect_md_report(self, threshold: float) -> InlineTextReport:
        return InlineTextReport(
            media_type="text/markdown",
            content=f"{self.outputs.text}:{threshold}",
            filename="tiny-report.md",
        )

    def extract(self) -> Sequence[BaseRecord]:
        return [TinyRecord(run_uid=self.run_uid, payload=self.outputs.text)]


class EmptyTinyRun(TinyRun):
    """Valid run that intentionally produces no analytics records."""

    def extract(self) -> Sequence[BaseRecord]:
        return []


class ReportlessTinyRun(CapabilityRunBase[TinyConfig, TinyOutputs]):
    """Run that intentionally does not implement Markdown reporting."""

    config: TinyConfig
    outputs: TinyOutputs

    def extract(self) -> Sequence[BaseRecord]:
        return [TinyRecord(run_uid=self.run_uid, payload=self.outputs.text)]


class OversizedReportTinyRun(TinyRun):
    """Run whose generated inline report exceeds the job metadata limit."""

    def collect_md_report(self, threshold: float) -> InlineTextReport:
        _ = threshold
        return InlineTextReport(
            media_type="text/markdown",
            content="x" * (MAX_INLINE_REPORT_BYTES + 1),
            filename="oversized.md",
        )


class TinyCapability(Capability[TinyOutputs, Any, Any, Any, TinyConfig]):
    _RUN_TYPE = TinyRun

    default_num_cpus = 1
    default_num_gpus = 0

    @classmethod
    def _create_config(cls) -> TinyConfig:
        return TinyConfig()

    @property
    def supports_datasets(self) -> Number:
        return Number.ZERO

    @property
    def supports_models(self) -> Number:
        return Number.ZERO

    @property
    def supports_metrics(self) -> Number:
        return Number.ZERO

    def _run(
        self,
        models: list[Any],
        datasets: list[Any],
        metrics: list[Any],
        config: TinyConfig,
        use_prediction_and_evaluation_cache: bool,
    ) -> TinyOutputs:
        del models, datasets, metrics, use_prediction_and_evaluation_cache

        if config.start_marker_path is not None:
            start_marker = Path(config.start_marker_path)
            start_marker.parent.mkdir(parents=True, exist_ok=True)
            start_marker.write_text("started")

        if config.sleep_s:
            time.sleep(config.sleep_s)

        if config.fail:
            raise RuntimeError("tiny capability failure")

        if config.env_key is not None:
            output_text = os.environ.get(config.env_key, "")
        else:
            output_text = config.text

        if config.finish_marker_path is not None:
            finish_marker = Path(config.finish_marker_path)
            finish_marker.parent.mkdir(parents=True, exist_ok=True)
            finish_marker.write_text("finished")

        return TinyOutputs(text=output_text)


class EmptyTinyCapability(TinyCapability):
    """Tiny capability used to exercise successful empty analytics results."""

    _RUN_TYPE = EmptyTinyRun


class ReportlessTinyCapability(TinyCapability):
    """Tiny capability used to exercise runs without reports."""

    _RUN_TYPE = ReportlessTinyRun


class OversizedReportTinyCapability(TinyCapability):
    """Tiny capability used to exercise oversized report handling."""

    _RUN_TYPE = OversizedReportTinyRun


class AppendMarkerCapability(TinyCapability):
    """Tiny capability variant that appends to ``start_marker_path`` on every execution.

    Used by Ray worker tests that need to distinguish a cached local run from a
    fresh worker execution with the same run identity.
    """

    def _run(
        self,
        models: list[Any],
        datasets: list[Any],
        metrics: list[Any],
        config: TinyConfig,
        use_prediction_and_evaluation_cache: bool,
    ) -> TinyOutputs:
        if config.start_marker_path is not None:
            marker = Path(config.start_marker_path)
            marker.parent.mkdir(parents=True, exist_ok=True)
            with marker.open("a") as file:
                file.write("run\n")
            config = config.model_copy(update={"start_marker_path": None})
        return super()._run(models, datasets, metrics, config, use_prediction_and_evaluation_cache)


class TinyDatasetCapability(TinyCapability):
    """Tiny capability variant that requires one dataset.

    Used by tests that need rows in the ``runs`` table.
    """

    @property
    def supports_datasets(self) -> Number:
        return Number.ONE
