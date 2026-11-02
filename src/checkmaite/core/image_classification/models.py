"""Native modelmaite image-classification models for CheckMAITE.

The model wrappers (`TorchvisionICModel`, `OnnxICModel`) live in
`modelmaite <https://pypi.org/project/modelmaite/>`_ and are re-exported here;
CheckMAITE does not maintain a second set of model classes. What stays in
CheckMAITE is the config-facing contract consumed by the checkmaite-frontend
api adapter: :class:`ModelSpecification`, :func:`load_models`, and the
``SUPPORTED_MODELS`` table.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from modelmaite.image_classification import OnnxICModel, TorchvisionICModel
from modelmaite.image_classification.models import (
    SUPPORTED_ONNX_MODELS,
    SUPPORTED_TORCHVISION_MODELS,
)
from modelmaite.image_classification.models import load_models as modelmaite_load_models

from checkmaite.core._common.install_hints import checkmaite_install_hints

__all__ = [
    "SUPPORTED_MODELS",
    "SUPPORTED_ONNX_MODELS",
    "SUPPORTED_TORCHVISION_MODELS",
    "ModelSpecification",
    "OnnxICModel",
    "TorchvisionICModel",
    "load_models",
]

# list of all available model wrappers
SUPPORTED_MODELS = {**SUPPORTED_TORCHVISION_MODELS, "jatic_onnx": "JATIC_ONNX"}


class ModelSpecification(TypedDict):
    """Model metadata required for loading models via CheckMAITE wrappers"""

    # full filepath to model weights file
    model_weights_path: NotRequired[str | Path]
    # full filepath to model config file
    model_config_path: NotRequired[str | Path]
    # model type, keys map to model wrappers
    # TO DO hard-coded due to https://github.com/microsoft/pyright/issues/9194 and maite pyright<=1.1.320
    model_type: Literal[
        "alexnet",
        "resnext50_32x4d",
        "jatic_onnx",
    ]


def load_models(
    models: Mapping[str, ModelSpecification],
    **kwargs: Any,
) -> dict[str, TorchvisionICModel | OnnxICModel]:
    """Load image-classification models via modelmaite's native factory.

    checkmaite's IC spec keys are exactly modelmaite's, so dispatch, validation,
    options, and errors are ``modelmaite.image_classification.load_models``.
    The only change is that a missing ONNX dependency raises an ``ImportError``
    that names the ``checkmaite[onnx]`` extra.
    """
    # checkmaite's ModelSpecification is one TypedDict; modelmaite's is a union
    # of per-wrapper TypedDicts, so pyright cannot match them structurally.
    specs: dict[str, Any] = dict(models)
    with checkmaite_install_hints():
        return modelmaite_load_models(specs, **kwargs)
