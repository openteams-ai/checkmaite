import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Literal, TypeVar

import maite.protocols.generic as gen
from maite.protocols import MetricMetadata

MetricFailureStage = Literal["reset", "update", "compute", "normalize"]
T_Target = TypeVar("T_Target")
T_Metadata = TypeVar("T_Metadata")
T_Metric = TypeVar("T_Metric", bound=gen.Metric[Any, Any])


class MaiteEvaluationMetricError(RuntimeError):
    """A failure attributed to one metric in a MAITE evaluation."""

    def __init__(
        self,
        metric_id: str,
        stage: MetricFailureStage,
        error_type: str,
        message: str,
    ) -> None:
        self.metric_id: str = metric_id
        self.stage: MetricFailureStage = stage
        self.error_type: str = error_type
        self.message: str = message
        super().__init__(f"Metric {metric_id!r} failed during {stage}: {error_type}: {message}")

    def __reduce__(self) -> tuple[type["MaiteEvaluationMetricError"], tuple[str, MetricFailureStage, str, str]]:
        return type(self), (self.metric_id, self.stage, self.error_type, self.message)


def _canonicalize_metrics(metrics: Sequence[T_Metric]) -> list[T_Metric]:
    """Validate metrics and return a new list ordered by stable metadata ID."""
    identified_metrics: list[tuple[str, T_Metric]] = []
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for metric in metrics:
        try:
            metric_id = metric.metadata["id"]
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError("Every metric must define metadata with a non-empty string 'id'.") from error
        if type(metric_id) is not str or not metric_id.strip():
            raise ValueError("Every metric metadata 'id' must be a non-empty string.")
        if metric_id in seen_ids:
            duplicate_ids.add(metric_id)
        seen_ids.add(metric_id)
        identified_metrics.append((metric_id, metric))

    if duplicate_ids:
        duplicate_list = ", ".join(sorted(duplicate_ids))
        raise ValueError(f"Metric metadata IDs must be unique, but found duplicates: {duplicate_list}.")

    return [metric for _, metric in sorted(identified_metrics, key=lambda item: item[0])]


def _member_failure(metric_id: str, stage: MetricFailureStage, error: Exception) -> MaiteEvaluationMetricError:
    return MaiteEvaluationMetricError(
        metric_id=metric_id,
        stage=stage,
        error_type=type(error).__name__,
        message=str(error),
    )


class _MetricFanout(Generic[T_Target, T_Metadata]):
    """Present an ordered collection of metrics as one MAITE metric."""

    def __init__(self, metrics: Sequence[gen.Metric[T_Target, T_Metadata]]) -> None:
        self._metrics = tuple(_canonicalize_metrics(metrics))
        payload = {
            "schema": 1,
            "metric_ids": [metric.metadata["id"] for metric in self._metrics],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.metadata: MetricMetadata = {"id": f"checkmaite.metric-fanout.v1:{digest}"}

    def reset(self) -> None:
        for metric in self._metrics:
            metric_id = metric.metadata["id"]
            try:
                metric.reset()
            except Exception as error:
                raise _member_failure(metric_id, "reset", error) from error

    def update(
        self,
        predictions: Sequence[T_Target],
        targets: Sequence[T_Target],
        metadata: Sequence[T_Metadata],
        /,
    ) -> None:
        for metric in self._metrics:
            metric_id = metric.metadata["id"]
            try:
                metric.update(predictions, targets, metadata)
            except Exception as error:
                raise _member_failure(metric_id, "update", error) from error

    def compute(self) -> Mapping[str, Any]:
        results: dict[str, dict[str, Any]] = {}
        for metric in self._metrics:
            metric_id = metric.metadata["id"]
            try:
                result = metric.compute()
                if not isinstance(result, Mapping):
                    raise TypeError(f"Metric.compute() must return a mapping, but returned {type(result).__name__}.")
                results[metric_id] = dict(result)
            except Exception as error:
                raise _member_failure(metric_id, "compute", error) from error
        return results
