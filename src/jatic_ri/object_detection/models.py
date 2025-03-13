"""Catalog of model wrappers and other model loading utilities"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, TypedDict

import httpx
import numpy as np
import torch
from maite.protocols import object_detection as od
from typing_extensions import NotRequired

from jatic_ri._common.models import (
    get_default_index2label,
    get_index2label_from_model_config,
    maybe_download_weights,
    set_device,
    to_torch_batch,
    validate_input_batch,
)
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

SUPPORTED_VISDRONE_MODELS = {
    "res2net50": "Res2Net50_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet18": "ResNet18_Weights",
}

# list of all available model wrappers
SUPPORTED_MODELS = {**SUPPORTED_TORCHVISION_MODELS, **SUPPORTED_VISDRONE_MODELS}

DEFAULT_TORCHVISION_CONFIG_PATH = "config.json"


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

    def __init__(
        self,
        *,
        model_name: str,
        device: Optional[str] = None,  # noqa: UP007
        weights_path: Optional[str] = None,  # noqa: UP007
        config_path: Optional[str] = None,  # noqa: UP007
        index2label_key: str = "index2label",
        model_id: str = "torchvisionOD",
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Args:
            model_name: Name of torchvision model to instantiate.
            device: Device to use (e.g., 'cpu', 'cuda').
            weights_path: Path to pickle file with pretrained weights.
            config_path: Path to config file with metadata for pretrained weights.
            index2label_key: Config key for mapping from class labels to output categories.
            model_id: unique identifier for this model
            **kwargs: Additional parameters passed to `torchvision` base class. See
              `torchvision` for more details.
        """
        if model_name not in SUPPORTED_TORCHVISION_MODELS:
            raise ValueError(f"Model {model_name} is not currently supported by jatic_ri.")
        self._model_name = model_name

        if not config_path:
            config_path = DEFAULT_TORCHVISION_CONFIG_PATH

        try:
            model = getattr(importlib.import_module("torchvision.models.detection"), model_name)
            torchvision_weights_constructor = getattr(
                importlib.import_module("torchvision.models.detection"),
                SUPPORTED_TORCHVISION_MODELS[model_name],
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

            num_classes = model_config.get("num_classes", None)

            self.index2label = get_index2label_from_model_config(config_path, model_config, index2label_key)

            # initializing model with random weights and then loading weights via state_dict
            # appears to be easiest method for allowing user to bring their own weights.
            # gotcha: it's not sufficient to set weights to None, the weights for the backbone
            # model must also be set to None, otherwise torchvision will instantiate a subtly
            # different model architecture in some cases
            self.model = model(weights=None, weights_backbone=None, num_classes=num_classes, **kwargs).to(self.device)

            try:
                state_dict = torch.load(
                    weights_path,
                    map_location=torch.device(self.device),
                    weights_only=True,
                )
                self.model.load_state_dict(state_dict)
            except Exception as e:
                raise RuntimeError(f"Error loading data from state_dict from {weights_path}") from e

        else:  # we are loading torchvision pre-trained weights into model
            self.index2label = get_default_index2label(torchvision_weights_constructor)
            self.model = maybe_download_weights(model, torchvision_weights_constructor, self.device, **kwargs)

        self.metadata = {"id": model_id}
        self.model.eval()  # sets model to inference mode rather than training mode

    def __call__(self, input_batch: od.InputBatchType) -> od.TargetBatchType:
        """
        Make a model prediction for inputs in input batch. Elements of input batch
        are expected in the shape `(C, H, W)`.
        """
        validate_input_batch(input_batch)
        torch_batch = to_torch_batch(input_batch, self.device)
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


class VisdroneODModel:
    """
    Minimal MAITE-complaint wrapper of visdone wrapper packaged by Kitware.

    See https://github.com/Kitware/SMQTK-Detection/blob/master/smqtk_detection/impls/detect_image_objects/centernet.py
    for details.
    """

    _weights_urls = {
        "Res2Net50_Weights": "https://data.kitware.com/api/v1/item/623e18464acac99f42f40a4e/download",
        "ResNet50_Weights": "https://data.kitware.com/api/v1/item/623259f64acac99f426f21db/download",
        "ResNet18_Weights": "https://data.kitware.com/api/v1/item/623de4744acac99f42f05fb1/download",
    }

    def __init__(
        self,
        *,
        arch: str,
        model_pickle_dir: Optional[str],  # noqa: UP007
        device: None | str | torch.device = None,
        batch_size: int = 3,
        num_workers: int = 0,  # default 0 easiest solution to https://github.com/pytorch/pytorch/issues/87688
        max_dets: int = 500,
        model_id: str = "visdrone",
    ) -> None:
        """
        Args:
            arch: Model backbone to be used (allowed values are "res2net50", "resnet50", "resnet18")
            model_pickle_dir: Directory where model pickle will be downloaded to.  Defaults to CWD.
            device: Device to use (e.g., 'cpu', 'cuda'). If None, attempts to locate a GPU. If no GPU
                is present, falls back to CPU.
            batch_size: How many samples per batch to load.
            num_workers: How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.
            max_dets: Maximum number of detections returned.
            model_id: unique identifier for this model
        """
        from smqtk_detection.impls.detect_image_objects.centernet import (
            CenterNetVisdrone,
        )

        if not model_pickle_dir:
            model_pickle_dir = os.getcwd()

        if arch not in SUPPORTED_VISDRONE_MODELS:
            raise ValueError(f"Model with backbone {arch} is not currently supported by the visdrone model.")
        self._model_name = f"centernet-{arch}"
        model_file = os.path.join(model_pickle_dir, f"{self._model_name}.pth")
        if not os.path.isfile(model_file):
            Path(model_pickle_dir).mkdir(parents=True, exist_ok=True)
            with httpx.stream(
                "GET", self._weights_urls[SUPPORTED_VISDRONE_MODELS[arch]], timeout=10, follow_redirects=True
            ) as response:
                response.raise_for_status()
                with open(model_file, "wb") as file:
                    for chunk in response.iter_bytes():
                        file.write(chunk)

        self.model = CenterNetVisdrone(
            arch=arch,
            model_file=model_file,
            device=str(set_device(device)),
            max_dets=max_dets,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.index2label = {
            0: "ignored regions",
            1: "pedestrian",
            2: "people",
            3: "bicycle",
            4: "car",
            5: "van",
            6: "truck",
            7: "tricycle",
            8: "awning-tricycle",
            9: "bus",
            10: "motor",
            11: "others",
        }
        self.metadata = {"id": model_id}

    def __call__(self, input_batch: od.InputBatchType) -> od.TargetBatchType:
        """
        Make a model prediction for inputs in input batch. Elements of input batch
        are expected in the shape `(C, H, W)`.
        """
        # required to convert to numpy array before passing to detect_objects ...
        array_batch = [np.array(inp).transpose(1, 2, 0) for inp in input_batch]

        predictions_batch = self.model.detect_objects(array_batch)

        label2index = {v: k for k, v in self.index2label.items()}

        output_batch = []
        for pred in predictions_batch:
            num_bboxes = len(pred)  # type: ignore

            bboxes = np.empty((num_bboxes, 4), dtype=np.float32)
            labels = np.empty(num_bboxes, dtype=np.int32)
            scores = np.empty(num_bboxes, dtype=np.float32)

            for i, (bbox, label_map) in enumerate(pred):
                (min_x, min_y), (max_x, max_y) = (
                    bbox.min_vertex,
                    bbox.max_vertex,
                )
                bboxes[i] = [min_x, min_y, max_x, max_y]
                label, score = max(label_map.items(), key=lambda x: x[1])
                labels[i] = label2index[label]  # type: ignore
                scores[i] = score

            output_batch.append(DetectionTarget(boxes=bboxes, labels=labels, scores=scores))

        return output_batch

    @property
    def name(self) -> str:
        """Human-readable name for visdrone model"""
        return f"visdrone-centernet-{self._model_name}"


class ModelSpecification(TypedDict):
    """Model metadata required for loading models via the RI wrappers"""

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
    ]


def load_models(
    models: dict[str, ModelSpecification],
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, TorchvisionODModel]:  # pragma: no cover
    """Simplified programmatic loading of models from a dictionary of
    ModelSpecifications."""
    loaded = {}
    for name, meta_dict in models.items():
        if meta_dict["model_type"] in SUPPORTED_TORCHVISION_MODELS:
            wrapper = TorchvisionODModel(
                model_name=meta_dict["model_type"],
                weights_path=meta_dict.get("model_weights_path"),
                config_path=meta_dict.get("model_config_path"),
                model_id=meta_dict["model_type"],
                **kwargs,
            )

            loaded[name] = wrapper

        elif meta_dict["model_type"] in SUPPORTED_VISDRONE_MODELS:
            wrapper = VisdroneODModel(
                arch=meta_dict["model_type"],
                model_pickle_dir=meta_dict.get("model_weights_path"),
                model_id=meta_dict["model_type"],
            )
            loaded[name] = wrapper

        else:
            raise RuntimeError(f"Model type not yet supported {meta_dict['model_type']}")

    return loaded
