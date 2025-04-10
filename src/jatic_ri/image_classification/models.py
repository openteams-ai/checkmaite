"""models"""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING, Any, Literal, Optional, TypedDict

import torch
from maite.protocols import image_classification as ic
from typing_extensions import NotRequired

from jatic_ri._common.models import (
    get_default_index2label,
    get_index2label_from_model_config,
    maybe_download_weights,
    set_device,
    to_torch_batch,
    validate_input_batch,
)

if TYPE_CHECKING:
    from torch import nn

SUPPORTED_TORCHVISION_MODELS = {
    "alexnet": "AlexNet_Weights",
    "resnext50_32x4d": "ResNeXt50_32X4D_Weights",
}

# list of all available model wrappers
SUPPORTED_MODELS = SUPPORTED_TORCHVISION_MODELS


DEFAULT_TORCHVISION_CONFIG_PATH = "config.json"


class TorchvisionICModel:
    """
    A MAITE-compliant wrapper for image classification models from the torchvision package.

    The primary purpose of this wrapper is to ensure that the model is called correctly.
    This includes:
        - Downloading and/or loading the model weights onto the specified device.
        - Providing access to a mapping between integer predictions made by the model
          and human-readable class names.
        - Ensuring that input images are correctly transformed before being passed to the model
          for inference, and similarly when predictions are returned by the model.

    Attributes:
        name: Name of the image classification model (e.g., 'alexnet').
        device: Device where the model will run ('cpu' or 'cuda').
        preprocess: Preprocessing transforms necessary for converting images to the format
          expected by the model.
        index2label: List of class labels corresponding to the model's output categories.
        model: Underlying torchvision model instance, ready for inference.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: Optional[str] = None,  # noqa: UP007
        weights_path: Optional[str] = None,  # noqa: UP007
        config_path: Optional[str] = None,  # noqa: UP007
        index2label_key: str = "index2label",
        model_id: str = "torchvisionIC",
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Args:
            model_name: Name of torchvision model to instantiate.
            device: Device to use (e.g., 'cpu', 'cuda').
            weights_path: Path to pickle file with pretrained weights.
            config_path: Path to config file with metadata for pretrained weights.
            index2label_key: Config key for mapping from class labels to output categories.
            model_id: Unique identifier for this model
            **kwargs: Additional parameters passed to `torchvision` base class. See
              `torchvision` for more details.
        """
        if model_name not in SUPPORTED_TORCHVISION_MODELS:
            raise ValueError(f"Model {model_name} is not currently supported by jatic_ri.")
        self._model_name = model_name

        if not config_path:
            config_path = DEFAULT_TORCHVISION_CONFIG_PATH

        try:
            model = getattr(importlib.import_module("torchvision.models"), model_name)
            torchvision_weights_constructor = getattr(
                importlib.import_module("torchvision.models"), SUPPORTED_TORCHVISION_MODELS[model_name]
            )
        except Exception as e:
            raise ImportError(f"There was an error importing {model_name} from torchvision.") from e

        self.device: torch.device = set_device(device)

        # warning: assumed that preprocessing transforms will be unchanged even when loading user-supplied weights
        self.preprocess: nn.Module = torchvision_weights_constructor.DEFAULT.transforms()

        self.model: nn.Module
        self.index2label: dict[int, str]
        if weights_path is not None:  # we are loading user weights into model
            try:
                with open(config_path) as f:
                    model_config = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Configuration file not found at path: {config_path}") from None

            self.index2label = get_index2label_from_model_config(config_path, model_config, index2label_key)
            num_classes = model_config.get("num_classes", len(self.index2label))

            # initializing model with random weights and then loading weights via state_dict
            # appears to be easiest method for allowing user to bring their own weights.
            self.model = model(weights=None, num_classes=num_classes, **kwargs).to(self.device)

            try:
                state_dict = torch.load(weights_path, map_location=torch.device(self.device), weights_only=True)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                raise RuntimeError(f"Error loading data from state_dict from {weights_path}") from e
        else:  # we are loading torchvision pre-trained weights into model
            self.index2label = get_default_index2label(torchvision_weights_constructor)
            self.model = maybe_download_weights(model, torchvision_weights_constructor, self.device, **kwargs)

        self.model.eval()  # sets model to inference mode rather than training mode
        self.metadata = {"id": model_id}

    def __call__(self, input_batch: ic.InputBatchType) -> ic.TargetBatchType:
        """
        Make a model prediction for inputs in input batch. Elements of input batch
        are expected in the shape `(C, H, W)`.
        """
        validate_input_batch(input_batch)
        torch_batch = to_torch_batch(input_batch, self.device)
        batch_on_device = self.preprocess(torch_batch).to(self.device)

        with torch.no_grad():
            logits_batch = self.model(batch_on_device)
            # transfer to CPU as downstream package may not support other devices
            return list(torch.nn.functional.softmax(input=logits_batch, dim=1).cpu().detach())

    @property
    def name(self) -> str:
        """Human-readable name for torchvision model"""
        return self._model_name


class ModelSpecification(TypedDict):
    """Model metadata required for loading models via the RI wrappers"""

    # full filepath to model weights file
    model_weights_path: NotRequired[str]
    # full filepath to model config file
    model_config_path: NotRequired[str]
    # model type, keys map to model wrappers
    # TO DO hard-coded due to https://github.com/microsoft/pyright/issues/9194 and maite pyright<=1.1.320
    model_type: Literal[
        "alexnet",
        "resnext50_32x4d",
    ]


def load_models(
    models: dict[str, ModelSpecification],
    **kwargs: Any,
) -> dict[str, TorchvisionICModel]:  # pragma: no cover
    """Simplified programmatic loading of models from a dictionary of
    ModelSpecifications."""
    loaded = {}
    for name, meta_dict in models.items():
        # if this is a torchvision model
        if meta_dict["model_type"] in SUPPORTED_TORCHVISION_MODELS:
            wrapper = TorchvisionICModel(
                model_name=meta_dict.get("model_type"),
                weights_path=meta_dict.get("model_weights_path"),
                config_path=meta_dict.get("model_config_path"),
                model_id=meta_dict.get("model_type"),
                **kwargs,
            )

            loaded[name] = wrapper
        else:
            raise RuntimeError(f"Model type not yet supported {meta_dict['model_type']}")

    return loaded
