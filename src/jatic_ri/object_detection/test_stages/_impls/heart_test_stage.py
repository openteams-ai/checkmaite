"""Heart Test Stage that supports evaluation model robustness against PGD and Patch Attacks for Object Detection"""

import itertools
from typing import Any, Literal

import cv2
import numpy as np
import pydantic
from art.attacks.evasion import AdversarialPatchPyTorch, ProjectedGradientDescent
from gradient import parse_lines
from heart_library.attacks.attack import JaticAttack
from heart_library.estimators.object_detection.pytorch import (
    JaticPyTorchObjectDetectionOutput,
    JaticPyTorchObjectDetector,
)
from maite.protocols import ArrayLike
from numpy.typing import NDArray

from jatic_ri._common.test_stages.interfaces.plugins import (
    MetricPlugin,
    SingleDatasetPlugin,
    SingleModelPlugin,
    ThresholdPlugin,
)
from jatic_ri._common.test_stages.interfaces.test_stage import ConfigBase, OutputsBase, RunBase, TestStage
from jatic_ri.cached_tasks import evaluate
from jatic_ri.util._types import Image
from jatic_ri.util.utils import temp_image_file

AttackName = Literal["PGD", "Patch"]
AttackStrength = Literal["weak", "strong"]

_DEFAULT_HEART_ATTACK_PARAMETERS: dict[tuple[AttackName, AttackStrength], dict[str, Any]] = {
    ("PGD", "weak"): {
        "max_iter": 10,  # Number of optimization steps (fewer = weaker attack)
        "eps": 1,  # Max pixel change allowance
        "eps_step": 0.2,  # Step size for each optimization step
    },
    ("PGD", "strong"): {"max_iter": 20, "eps": 2, "eps_step": 0.2},
    ("Patch", "weak"): {
        "rotation_max": 0.0,  # Maximum rotation in radians (0 = no rotation)
        "scale_min": 0.5,  # Minimum scaling factor for patch size
        "scale_max": 1.0,  # Maximum scaling factor for patch size
        "distortion_scale_max": 0.0,  # Maximum perspective distortion (0 = no distortion)
        "learning_rate": 1.99,  # Step size for patch updates
        "max_iter": 5,  # Number of optimization steps (how many times we update the patch)
        "batch_size": 16,  # Number of images processed simultaneously during optimization
        "patch_shape": (3, 100, 100),  # Patch dimensions (channels, height, width)
        "patch_location": (240, 100),  # (x,y) coordinates where patch is applied (relative to image)
        "patch_type": "square",  # Shape of the patch (options: 'square', 'circle', etc.)
        "optimizer": "Adam",  # Optimization algorithm used
    },
    ("Patch", "strong"): {
        "rotation_max": 0.0,
        "scale_min": 0.5,
        "scale_max": 1.0,
        "distortion_scale_max": 0.0,
        "learning_rate": 1.99,
        "max_iter": 10,
        "batch_size": 16,
        "patch_shape": (3, 100, 100),
        "patch_location": (440, 100),
        "patch_type": "square",
        "optimizer": "Adam",
    },
}

_ATTACK_MAP: dict[AttackName, type[Any]] = {
    "PGD": ProjectedGradientDescent,
    "Patch": AdversarialPatchPyTorch,
}


class HeartAttackConfig(pydantic.BaseModel):
    """Attack config for HEART test stage"""

    model_config = pydantic.ConfigDict(frozen=True)

    name: AttackName
    strength: AttackStrength
    parameters: dict[str, Any] = pydantic.Field(
        default_factory=lambda values: _DEFAULT_HEART_ATTACK_PARAMETERS[(values["name"], values["strength"])].copy()
    )


class HeartConfig(ConfigBase):
    """Config for HEART test stage"""

    attack_configs: list[HeartAttackConfig]


class BaselineOutput(OutputsBase):
    """Baseline evaluation output of HEART test stage"""

    result: dict[str, float]
    images: list[Image]


class AttackOutput(OutputsBase):
    """Attacked evaluation output of HEART test stage"""

    attack_config: HeartAttackConfig
    result: dict[str, float]
    images: list[Image]


class HeartOutputs(OutputsBase):
    """Output of HEART test stage"""

    baseline: BaselineOutput
    attacked: list[AttackOutput]


class HeartRun(RunBase):
    """Run for HEART test stage"""

    config: HeartConfig
    outputs: HeartOutputs


class HeartTestStage(
    TestStage[HeartOutputs],
    SingleDatasetPlugin[Any],
    SingleModelPlugin[Any],
    MetricPlugin[Any],
    ThresholdPlugin,
):
    """HEART Specific Implementation of a TestStage Class to Support
    Object Detection Adversarial Attacks, including:
    1. Projected Gradient Descent
    2. Adversarial Patch Attack
    """

    def __init__(self, *, attack_configs: list[HeartAttackConfig], threshold: float = 0.3) -> None:
        super().__init__()
        self._config = HeartConfig.model_validate({"attack_configs": attack_configs})
        self.load_threshold(threshold)

    _RUN_TYPE = HeartRun

    def _create_config(self) -> HeartConfig:
        return self._config

    def _run(self) -> HeartOutputs:
        """Runs Object Detection Adversarial Attacks and performs robustness
        evaluations for specific metrics.
        """
        # Note: __init__ ensures heart_library is available before this runs
        heart_detector: Any = JaticPyTorchObjectDetector(self.model.model, clip_values=(0, 1))

        result, preds, _ = evaluate(
            model=heart_detector,
            metric=self.metric,
            dataset=self.dataset,
            return_preds=True,
            return_augmented_data=False,
        )
        baseline = BaselineOutput.model_validate(
            {
                "result": result,
                "images": [
                    self._plot_image_with_boxes(
                        image,
                        preds,
                        threshold=self.threshold,
                        index2label=self.dataset.metadata["index2label"],
                    )
                    for image, preds in zip(
                        (image for image, *_ in self.dataset),
                        itertools.chain.from_iterable(preds),
                        strict=True,
                    )
                ],
            }
        )

        attacked: list[AttackOutput] = []
        for c in self._config.attack_configs:
            attack = _ATTACK_MAP[c.name](heart_detector, **c.parameters)
            result, preds, augmented_data = evaluate(
                model=heart_detector,
                metric=self.metric,
                dataset=self.dataset,
                # Note: __init__ ensures heart_library is available before this runs
                augmentation=JaticAttack(attack),
                return_preds=True,
                return_augmented_data=True,
            )
            attacked.append(
                AttackOutput.model_validate(
                    {
                        "attack_config": c,
                        "result": result,
                        "images": [
                            self._plot_image_with_boxes(
                                image,
                                preds,
                                threshold=self.threshold,
                                index2label=self.dataset.metadata["index2label"],
                            )
                            for image, preds in zip(
                                itertools.chain.from_iterable(b for b, *_ in augmented_data),
                                itertools.chain.from_iterable(preds),
                                strict=True,
                            )
                        ],
                    }
                )
            )

        return HeartOutputs(baseline=baseline, attacked=attacked)

    def _plot_image_with_boxes(
        self,
        image: ArrayLike,
        # Note: __init__ ensures heart_library is available before this runs
        preds: JaticPyTorchObjectDetectionOutput,
        *,
        threshold: float,
        index2label: dict[int, str],
    ) -> bytes:
        image = self._preprocess_image(image)
        extracted_preds = self._extract_predictions(preds, threshold=threshold, index2label=index2label)

        for label, box, _ in zip(*extracted_preds, strict=False):
            cv2.rectangle(
                image,
                pt1=list(map(int, box[0])),
                pt2=list(map(int, box[1])),
                color=(0, 255, 0),
                thickness=1,
            )
            cv2.putText(
                image,
                label,
                org=list(map(int, box[0])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=1,
            )

        _, data = cv2.imencode(".png", image)
        return bytes(data)

    def _preprocess_image(self, img: ArrayLike) -> NDArray:
        """Preprocess image for use with OpenCV"""
        img = np.asarray(img)

        if img.shape[0] in [1, 3]:  # If first dimension is channel, Convert CHW -> HWC
            img = (img).transpose(1, 2, 0)

        # Normalize to 0-255 range if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        if len(img.shape) == 2 or img.shape[2] == 1:  # Convert Grayscale OR Single Channel -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:  # Convert RGB -> BGR (OpenCV expects BGR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 4:  # Convert RGBA -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        return img

    def _extract_predictions(
        self,
        # Note: __init__ ensures heart_library is available before this runs
        predictions: JaticPyTorchObjectDetectionOutput,
        *,
        threshold: float,
        index2label: dict[int, str],
    ) -> tuple[list, list, list]:
        """Utility function to extract predictions within a threshold"""
        predictions_class = [index2label[i] for i in list(predictions.labels)]
        if len(predictions_class) < 1:
            return [], [], []
        # Get the predicted bounding boxes
        predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions.boxes)]
        # Get the predicted prediction score
        predictions_score = list(predictions.scores)

        # Get a list of index with score greater than the threshold
        predictions_t = np.argwhere(np.array(predictions_score) > threshold).ravel().tolist()
        if not predictions_t:
            return [], [], []
        # predictions in score order
        predictions_boxes = [predictions_boxes[i] for i in predictions_t]
        predictions_class = [predictions_class[i] for i in predictions_t]
        predictions_scores = [predictions_score[i] for i in predictions_t]
        return predictions_class, predictions_boxes, predictions_scores

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Creates all needed information to create meaningful heart gradient component"""

        if self._stored_run is None:
            raise RuntimeError("TestStage must be run before accessing outputs")
        outputs = self._stored_run.outputs

        def format_result(result: dict[str, float]) -> str:
            return ", ".join(f"{metric}={value:.3f}" for metric, value in result.items())

        slides: list[dict[str, Any]] = []
        for i, output in enumerate(outputs.attacked, 1):
            for baseline_image, attack_image in zip(outputs.baseline.images, output.images, strict=True):
                left_section_content = temp_image_file(baseline_image)
                mid_section_content = temp_image_file(attack_image)
                right_section_content = parse_lines(
                    [
                        f"**Original metric**: *{format_result(outputs.baseline.result)}*",
                        f"**Perturbed metric**: *{format_result(output.result)}*",
                        f"**Strength**: *{output.attack_config.strength}*",
                        "**Parameters**:",
                        *[
                            f"*{param.replace('_', ' ')}*: {value}"
                            for param, value in output.attack_config.parameters.items()
                        ],
                    ],
                    fontsize=12,
                )

                slides.append(
                    {
                        "deck": "object_detection_model_evaluation",
                        "layout_name": "ThreeSection",
                        "layout_arguments": {
                            "title": (
                                f"Adversarial Robustness Testing: {output.attack_config.name} - "
                                f"Image {i}/{len(outputs.baseline.images)}"
                            ),
                            "left_section_heading": "Original",
                            "left_section_content": left_section_content,
                            "mid_section_heading": "Perturbed",
                            "mid_section_content": mid_section_content,
                            "right_section_heading": "Metadata",
                            "right_section_content": right_section_content,
                        },
                    },
                )

        return slides
