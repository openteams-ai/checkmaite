from collections.abc import Callable, Sequence

import maite.protocols.object_detection as od
import numpy as np
from pydantic import Field, model_validator

from checkmaite.core._common.maite_evaluation_capability import (
    MaiteEvaluationBase,
)
from checkmaite.core._common.maite_evaluation_capability import (
    MaiteEvaluationConfig as BaseMaiteEvaluationConfig,
)
from checkmaite.core._common.maite_evaluation_capability import (
    MaiteEvaluationRun as BaseMaiteEvaluationRun,
)
from checkmaite.core.object_detection.dataset_loaders import DetectionTarget

ODPredictionBatches = Sequence[Sequence[od.TargetType]]
ODCpuPredictionPostprocessor = Callable[[ODPredictionBatches], Sequence[Sequence[od.TargetType]]]


class MaiteEvaluationConfig(BaseMaiteEvaluationConfig):
    """Inference and CPU postprocessing settings for object-detection evaluation.

    The default values disable postprocessing completely, passing model
    predictions unchanged to the metric. When configured, the stock
    postprocessor consumes MAITE ``ArrayLike`` predictions on CPU after raw
    prediction-cache retrieval, so its settings do not invalidate that cache.

    Power users who need accelerator-resident postprocessing should implement it
    inside their model wrapper and leave these postprocessing settings unset,
    avoiding an unnecessary device-to-host transfer. The wrapper's
    ``metadata["id"]`` must incorporate its postprocessing configuration so
    prediction and run caches cannot reuse stale outputs when those semantics
    change.
    """

    confidence_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Minimum confidence required to keep a detection; for class-score matrices, "
            "the maximum class score is used."
        ),
    )
    nms_iou_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="IoU threshold for non-maximum suppression; disabled when unset.",
    )
    class_agnostic_nms: bool = Field(
        default=False,
        description=(
            "Whether non-maximum suppression should suppress overlapping boxes across different labels; "
            "requires nms_iou_threshold."
        ),
    )
    max_detections: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of highest-confidence detections to keep per image after other postprocessing; "
            "for class-score matrices, the maximum class score is used."
        ),
    )

    @model_validator(mode="after")
    def _validate_nms_settings(self) -> "MaiteEvaluationConfig":
        if self.class_agnostic_nms and self.nms_iou_threshold is None:
            raise ValueError("class_agnostic_nms=True requires nms_iou_threshold.")
        return self


class MaiteEvaluationRun(BaseMaiteEvaluationRun[MaiteEvaluationConfig]):
    """Object-detection MAITE evaluation run with OD-specific configuration."""


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    class_agnostic: bool,
) -> np.ndarray:
    """Run greedy NMS for MAITE ArrayLike prediction targets."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        intersection_x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        intersection_y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        intersection_x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        intersection_y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0, intersection_x2 - intersection_x1) * np.maximum(
            0, intersection_y2 - intersection_y1
        )
        iou = intersection / np.maximum(area[current] + area[rest] - intersection, np.finfo(float).eps)
        suppressed = iou > threshold
        if not class_agnostic:
            suppressed &= labels[rest] == labels[current]
        order = rest[~suppressed]
    return np.asarray(keep, dtype=np.int64)


def _confidence_scores(scores: np.ndarray) -> np.ndarray:
    if scores.ndim == 1:
        return scores
    if scores.ndim == 2:
        return np.max(scores, axis=1)
    raise ValueError(
        "Object-detection scores must have shape (n_boxes,) or (n_boxes, n_classes), "
        f"but received shape {tuple(scores.shape)}."
    )


def _postprocess_target(
    target: od.TargetType,
    config: MaiteEvaluationConfig,
) -> DetectionTarget:
    boxes = np.asarray(target.boxes)
    labels = np.asarray(target.labels)
    scores = np.asarray(target.scores)
    confidence_scores = _confidence_scores(scores)

    indices = np.arange(scores.shape[0])
    if config.confidence_threshold is not None:
        indices = indices[confidence_scores >= config.confidence_threshold]

    if config.nms_iou_threshold is not None and indices.size:
        selected = _nms(
            boxes[indices],
            confidence_scores[indices],
            labels[indices],
            config.nms_iou_threshold,
            config.class_agnostic_nms,
        )
        indices = indices[selected]

    if config.max_detections is not None:
        score_order = np.argsort(confidence_scores[indices])[::-1]
        indices = indices[score_order[: config.max_detections]]

    return DetectionTarget(
        boxes=boxes[indices],
        labels=labels[indices],
        scores=scores[indices],
    )


class MaiteEvaluation(MaiteEvaluationBase[od.Dataset, od.Model, od.Metric, MaiteEvaluationConfig]):
    """Evaluation of a single object-detection model, dataset and metric."""

    _RUN_TYPE = MaiteEvaluationRun

    @classmethod
    def _create_config(cls) -> MaiteEvaluationConfig:
        return MaiteEvaluationConfig()

    def _cpu_prediction_postprocessor(
        self, config: MaiteEvaluationConfig
    ) -> tuple[ODCpuPredictionPostprocessor | None, str | None]:
        """Build configured CPU postprocessing and its evaluation-cache identity.

        Returns no postprocessor for the default configuration, leaving model
        predictions and their normal cache behavior unchanged.
        """
        if not isinstance(config, MaiteEvaluationConfig):
            raise TypeError("Object-detection MaiteEvaluation requires an object-detection MaiteEvaluationConfig.")
        if config.confidence_threshold is None and config.nms_iou_threshold is None and config.max_detections is None:
            return None, None

        # Derive the postprocessor identity from every OD-specific field. Future
        # postprocessing fields therefore invalidate stale evaluations by default,
        # while base execution settings such as batch_size remain excluded.
        base_config_fields = set(BaseMaiteEvaluationConfig.model_fields)
        cpu_prediction_postprocessor_id = config.model_dump_json(exclude=base_config_fields)
        return (
            lambda predictions: self._cpu_postprocess_predictions(predictions, config),
            cpu_prediction_postprocessor_id,
        )

    def _cpu_postprocess_predictions(
        self,
        predictions: ODPredictionBatches,
        config: MaiteEvaluationConfig,
    ) -> list[list[DetectionTarget]]:
        """Apply postprocessing to MAITE ArrayLike detection targets.

        The returned canonical targets contain NumPy ``boxes``, ``labels``,
        and ``scores`` arrays. Additional attributes on custom target
        implementations are intentionally not copied.
        """
        return [[_postprocess_target(target, config) for target in batch] for batch in predictions]
