"""Native modelmaite object-detection models for CheckMAITE.

The model wrappers (`TorchvisionODModel`, `VisdroneODModel`, `OnnxODModel`)
live in `modelmaite <https://pypi.org/project/modelmaite/>`_ and are re-exported
here; CheckMAITE does not maintain a second set of model classes. What stays in
CheckMAITE is the config-facing contract consumed by the checkmaite-frontend
api adapter: :class:`ModelSpecification`, :func:`load_models`, and the
``SUPPORTED_MODELS`` table.
"""

from typing import Any, Literal, TypedDict

from modelmaite.object_detection import OnnxODModel, TorchvisionODModel, VisdroneODModel
from modelmaite.object_detection.models import (
    SUPPORTED_ONNX_MODELS,
    SUPPORTED_TORCHVISION_MODELS,
    SUPPORTED_VISDRONE_MODELS,
)
from modelmaite.object_detection.models import load_models as modelmaite_load_models
from typing_extensions import NotRequired

__all__ = [
    "SUPPORTED_MODELS",
    "SUPPORTED_ONNX_MODELS",
    "SUPPORTED_TORCHVISION_MODELS",
    "SUPPORTED_VISDRONE_MODELS",
    "ModelSpecification",
    "OnnxODModel",
    "TorchvisionODModel",
    "VisdroneODModel",
    "load_models",
]

# list of all available model wrappers
SUPPORTED_MODELS = {**SUPPORTED_TORCHVISION_MODELS, **SUPPORTED_VISDRONE_MODELS, "jatic_onnx": "JATIC_ONNX"}


class ModelSpecification(TypedDict):
    """Model metadata required for loading models via CheckMAITE wrappers"""

    # full filepath to model weights file
    model_weights_path: NotRequired[str]
    # full filepath to model config file
    # NOTE Visdrone Models do not take a config path.
    model_config_path: NotRequired[str]
    # model type, keys map to model wrappers
    # TO DO hard-coded due to https://github.com/microsoft/pyright/issues/9194 and maite pyright<=1.1.320
    model_type: Literal[
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_resnet50_fpn_v2",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "maskrcnn_resnet50_fpn_v2",
        "maskrcnn_resnet50_fpn",
        "retinanet_resnet50_fpn_v2",
        "retinanet_resnet50_fpn",
        "fcos_resnet50_fpn",
        "keypointrcnn_resnet50_fpn",
        "ssd300_vgg16",
        "ssdlite320_mobilenet_v3_large",
        "res2net50",
        "resnet50",
        "resnet18",
        "jatic_onnx",
    ]


def load_models(
    models: dict[str, ModelSpecification],
    **kwargs: Any,
) -> dict[str, TorchvisionODModel | VisdroneODModel | OnnxODModel]:
    """Load object-detection models via modelmaite's native factory.

    Checkmaite's spec contract keeps the legacy VisDrone ``model_weights_path``
    key; it is translated to modelmaite's ``model_pickle_dir`` here, then
    everything — dispatch, validation, options, and errors — is
    ``modelmaite.object_detection.load_models``. Keyword arguments therefore
    now reach VisDrone wrappers too (the previous checkmaite dispatch dropped
    them for the VisDrone branch only).
    """
    translated: dict[str, Any] = {}
    for name, spec in models.items():
        if spec.get("model_type") in SUPPORTED_VISDRONE_MODELS:
            visdrone_spec: dict[str, Any] = {"model_type": spec["model_type"]}
            if "model_weights_path" in spec:
                visdrone_spec["model_pickle_dir"] = spec["model_weights_path"]
            translated[name] = visdrone_spec
        else:
            translated[name] = spec
    return modelmaite_load_models(translated, **kwargs)
