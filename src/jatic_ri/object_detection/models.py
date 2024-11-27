"""models"""

from __future__ import annotations

import importlib
import json
import warnings
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from maite.protocols import object_detection as od

from jatic_ri.object_detection.datasets import DetectionTarget

if TYPE_CHECKING:
    from torch import nn

SUPPORTED_TORCHVISION_MODELS = {
    "fasterrcnn_resnet50_fpn": "FasterRCNN_ResNet50_FPN_Weights",
    "fasterrcnn_resnet50_fpn_v2": "FasterRCNN_ResNet50_FPN_V2_Weights",
    "fasterrcnn_mobilenet_v3_large_fpn": "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
    "maskrcnn_resnet50_fpn_v2": "MaskRCNN_ResNet50_FPN_V2_Weights",
    "maskrcnn_resnet50_fpn": "MaskRCNN_ResNet50_FPN_Weights",
    "retinanet_resnet50_fpn_v2": "RetinaNet_ResNet50_FPN_V2_Weights",
    "retinanet_resnet50_fpn": "RetinaNet_ResNet50_FPN_Weights",
    "fcos_resnet50_fpn": "FCOS_ResNet50_FPN_Weights",
    "keypointrcnn_resnet50_fpn": "KeypointRCNN_ResNet50_FPN_Weights",
    "ssd300_vgg16": "SSD300_VGG16_Weights",
    "ssdlite320_mobilenet_v3_large": "SSDLite320_MobileNet_V3_Large_Weights",
}


class TorchvisionWrapperError(Exception):
    """Base class for catching errors in torchvision wrapper"""

    pass


class InvalidModelNameError(TorchvisionWrapperError):
    """Model not currently supported by jatic_ri torchvision wrapper."""

    pass


class TorchvisionImportError(TorchvisionWrapperError):
    """Error when importing torchvision model."""

    pass


class MissingConfigFileError(TorchvisionWrapperError):
    """No config file supplied for user-supplied weights."""

    pass


class MissingIndex2LabelKeyError(TorchvisionWrapperError):
    """The config file supplied for user-supplied weights is missing an index2label key."""

    pass


class InvalidIndex2LabelError(TorchvisionWrapperError):
    """The index2label mapping has not been provided correctly."""

    pass


class StateDictLoadError(TorchvisionWrapperError):
    """Error when trying to load a user-supplied state dict."""

    pass


class InvalidInputBatchShapeError(TorchvisionWrapperError):
    """Input data is not in CHW format required by MAITE."""

    pass


class TorchvisionODModel:
    """
    A MAITE-compliant wrapper for object detection models from the torchvision package.

    The primary purpose of this wrapper is to ensure that the model is called correctly.
    This includes:
        - Downloading and/or loading the model weights onto the specified device.
        - Providing access to a mapping between integer predictions made by the model
          and human-readable class names.
        - Ensuring that input images are correctly transformed before being passed to the model
          for inference, and similarly when predictions are returned by the model.

    Attributes:
        name: Name of the object detection model (e.g., 'fasterrcnn_resnet50_fpn').
        device: Device where the model will run ('cpu' or 'cuda').
        preprocess: Preprocessing transforms necessary for converting images to the format
        expected by the model.
        index2label: List of class labels corresponding to the model's output categories.
        model: Underlying torchvision model instance, ready for inference.
    """

    def __init__(  # noqa: C901
        self,
        *,
        model_name: str,
        device: Optional[str] = None,  # noqa: UP007
        weights_path: Optional[str] = None,  # noqa: UP007
        config_path: str = "config.json",
        index2label_key: str = "index2label",
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Args:
            model_name (str): Name of torchvision model to instantiate.
            device (Optional[str]): Device to use (e.g., 'cpu', 'cuda').
            weights_path (Optional[str]): Path to pickle file with pretrained weights.
            config_path (str): Path to config file with metadata for pretrained weights.
            index2label_key (str): Config key for mapping from class labels to output categories.
            **kwargs (Any): Additional parameters passed to `torchvision` base class. See
            `torchvision` for more details.
        """
        if model_name not in SUPPORTED_TORCHVISION_MODELS:
            raise InvalidModelNameError(f"Model {model_name} is not currently supported by jatic_ri.")
        self._model_name = model_name

        try:
            model = getattr(importlib.import_module("torchvision.models.detection"), model_name)
            torchvision_weights_constructor = getattr(
                importlib.import_module("torchvision.models.detection"), SUPPORTED_TORCHVISION_MODELS[model_name]
            )
        except Exception as e:
            raise TorchvisionImportError(f"There was an error importing {model_name} from torchvision.") from e

        if device is None:
            if torch.cuda.is_available():
                self.device: str = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # warning: assumed that preprocessing transforms will be unchanged even when loading user-supplied weights
        self.preprocess: nn.Module = torchvision_weights_constructor.DEFAULT.transforms()

        self.model: nn.Module
        self.index2label: dict[int, str]
        if weights_path is not None:  # we are loading user weights into model
            try:
                with open(config_path) as f:
                    model_config = json.load(f)
            except FileNotFoundError:
                raise MissingConfigFileError(f"Configuration file not found at path: {config_path}") from None

            num_classes = model_config.get("num_classes", None)

            if index2label_key not in model_config:
                raise MissingIndex2LabelKeyError(
                    f"The config_file at {config_path} is missing a {index2label_key} key."
                )
            if isinstance(model_config[index2label_key], list):
                self.index2label = dict(enumerate(model_config[index2label_key]))
            elif isinstance(model_config[index2label_key], dict):
                self.index2label = {int(key): val for key, val in model_config[index2label_key].items()}
            else:
                raise InvalidIndex2LabelError(
                    f"index2label should be provided as a dict or list, not {type(model_config[index2label_key])}"
                )

            # initializing model with random weights and then loading weights via state_dict
            # appears to be easiest method for allowing user to bring their own weights.
            # gotcha: it's not sufficient to set weights to None, the weights for the backbone
            # model must also be set to None, otherwise torchvision will instantiate a subtly
            # different model architecture in some cases
            self.model = model(weights=None, weights_backbone=None, num_classes=num_classes, **kwargs).to(self.device)

            try:
                state_dict = torch.load(weights_path, map_location=torch.device(self.device), weights_only=True)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                raise StateDictLoadError(f"Error loading data from state_dict from {weights_path}") from e

        else:  # we are loading torchvision pre-trained weights into model
            self.index2label = dict(enumerate(torchvision_weights_constructor.DEFAULT.meta["categories"]))

            # if weights not already in cache, downloaded here
            self.model = model(weights=torchvision_weights_constructor.DEFAULT, **kwargs).to(self.device)

        self.model.eval()  # sets model to inference mode rather than training mode

    def __call__(self, input_batch: od.InputBatchType) -> od.TargetBatchType:
        """
        Make a model prediction for inputs in input batch. Elements of input batch
        are expected in the shape `(C, H, W)`.
        """
        total_channels, orig_img_height, orig_img_width = np.asarray(input_batch[0]).shape
        # channels can be used as a proxy to confirm CHW-ordering
        if not (1 <= total_channels <= 4):
            raise InvalidInputBatchShapeError(
                f"Input data must follow CHW-ordering, current shape: {total_channels, orig_img_height, orig_img_width}"
            )
        for val in input_batch:
            # required to convert to array to appease type-checker...
            npy_array = np.asarray(val)  # creates view, not copy
            if npy_array.shape != (total_channels, orig_img_height, orig_img_width):
                raise InvalidInputBatchShapeError(
                    f"All input images currently required to have identical shape, {npy_array.shape} "
                    f"not equal to {(total_channels, orig_img_height, orig_img_width)}. Please "
                    "contact the RI-team if your use-case requires unevenly shaped images."
                )

        # we are not writing to the underlying array in this method and hence we suppress this error
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors.",
                category=UserWarning,
            )
            torch_batch = torch.stack([torch.as_tensor(obj, device=self.device) for obj in input_batch])

        batch_on_device = self.preprocess(torch_batch).to(self.device)
        predictions_batch = self.model(batch_on_device)

        # gotcha: torchvision boxes do not need to be denormalised post inference!
        # see https://github.com/pytorch/vision/issues/2397
        # hence can just return predictions directly
        return [
            DetectionTarget(
                boxes=pred["boxes"]
                .detach()
                .cpu(),  # transfer to CPU as downstream package may not support other devices
                labels=pred["labels"].detach().cpu(),
                scores=pred["scores"].detach().cpu(),
            )
            for pred in predictions_batch
        ]

    @property
    def name(self) -> str:
        """Human-readable name for torchvision model"""
        return self._model_name
