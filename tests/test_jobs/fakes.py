from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

from checkmaite.core.analytics_store import BaseRecord
from checkmaite.core.capability_core import (
    Capability,
    CapabilityConfigBase,
    CapabilityOutputsBase,
    CapabilityRunBase,
    Number,
)


class TinyRecord(BaseRecord, table_name="tiny_jobs"):
    payload: str


class TinyConfig(CapabilityConfigBase):
    text: str = "ok"
    sleep_s: float = 0.0
    fail: bool = False
    env_key: str | None = None
    num_cpus: int | None = None
    num_gpus: float | None = None


class TinyOutputs(CapabilityOutputsBase):
    text: str


class TinyRun(CapabilityRunBase[TinyConfig, TinyOutputs]):
    config: TinyConfig
    outputs: TinyOutputs

    def collect_md_report(self, threshold: float) -> str:
        return f"{self.outputs.text}:{threshold}"

    def extract(self) -> Sequence[BaseRecord]:
        return [TinyRecord(run_uid=self.run_uid, payload=self.outputs.text)]


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

        if config.sleep_s:
            time.sleep(config.sleep_s)

        if config.fail:
            raise RuntimeError("tiny capability failure")

        if config.env_key is not None:
            return TinyOutputs(text=os.environ.get(config.env_key, ""))

        return TinyOutputs(text=config.text)


class TinyDatasetCapability(TinyCapability):
    """Tiny capability variant that requires one dataset.

    Used by tests that need rows in the ``runs`` table.
    """

    @property
    def supports_datasets(self) -> Number:
        return Number.ONE
