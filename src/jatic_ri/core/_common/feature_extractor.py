import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torchvision.transforms import Compose, Lambda

from jatic_ri.core._utils import maybe_download_weights

if TYPE_CHECKING:
    from PIL import Image

    from jatic_ri.core._types import ModelSpec


SUPPORTED_TORCHVISION_CLASSIFICATION_MODELS: dict[str, str] = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
    "efficientnet_b1": "EfficientNet_B1_Weights",
    "efficientnet_b2": "EfficientNet_B2_Weights",
    "efficientnet_b3": "EfficientNet_B3_Weights",
    "convnext_tiny": "ConvNeXt_Tiny_Weights",
    "convnext_small": "ConvNeXt_Small_Weights",
    "convnext_base": "ConvNeXt_Base_Weights",
}

# Feature Extractor/Embedding high-level plan:
# Use *image-classification* model (not object-detection) to turn each image into stable feature vector.
#
# Details:
# Apply image classification model's matching preprocessing, remove final classification head, then treat
# remaining penultimate feature vector as embedding (one vector per image). Optionally apply PCA to
# compress embedding down to user-requested number of components (only when requested dimension smaller
# than feature vector dimension).


@dataclass(frozen=True)
class FeatureExtractor:
    model: nn.Module
    transforms: Any
    name: str
    out_dim: int  # output dimension of feature vectors


def _ensure_rgb(img: "Image.Image | torch.Tensor") -> "Image.Image | torch.Tensor":
    """Convert image to 3-channel RGB if needed."""

    if isinstance(img, torch.Tensor):
        # Tensor: assume shape (C, H, W)
        if img.shape[0] == 1:
            # Grayscale -> RGB by repeating
            return img.repeat(3, 1, 1)
        if img.shape[0] == 4:
            # RGBA -> RGB by dropping alpha
            return img[:3]
        return img
    # PIL Image
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _load_torchvision_classification_model_and_preprocess(
    name: str,
    device: torch.device,
) -> tuple[nn.Module, Any]:
    mod = importlib.import_module("torchvision.models")

    if name not in SUPPORTED_TORCHVISION_CLASSIFICATION_MODELS:
        supported = ", ".join(sorted(SUPPORTED_TORCHVISION_CLASSIFICATION_MODELS))
        raise ValueError(f"Unsupported torchvision classification model: '{name}'. " f"Supported: {supported}")

    weights_enum_name = SUPPORTED_TORCHVISION_CLASSIFICATION_MODELS[name]

    try:
        model_constructor = getattr(mod, name)
        weights_constructor = getattr(mod, weights_enum_name)
    except AttributeError as e:
        raise ImportError(
            f"torchvision.models is missing '{name}' or its weights enum '{weights_enum_name}'. "
            "This usually means your torchvision version is too old/new for the mapping."
        ) from e

    base_preprocess = weights_constructor.DEFAULT.transforms()
    preprocess = Compose([Lambda(_ensure_rgb), base_preprocess])

    model = maybe_download_weights(model_constructor, weights_constructor, device)

    return model, preprocess


def _make_torchvision_feature_extractor(name: str, device: torch.device) -> FeatureExtractor:
    m, transforms = _load_torchvision_classification_model_and_preprocess(name, device)
    m.eval()

    # ResNet
    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        out_dim = m.fc.in_features
        m.fc = nn.Identity()

    # EfficientNet / ConvNeXt
    elif hasattr(m, "classifier") and isinstance(m.classifier, nn.Module):
        last_linear: nn.Linear | None = None
        for mod in m.classifier.modules():
            if isinstance(mod, nn.Linear):
                last_linear = mod
        if last_linear is None:
            raise ValueError(
                f"Cannot infer out_dim for torchvision model '{name}': model.classifier contains no nn.Linear layer."
            )
        out_dim = int(last_linear.in_features)
        m.classifier = nn.Identity()

    else:
        raise ValueError(
            f"Cannot infer out_dim for torchvision model '{name}': unsupported head structure "
            f"(expected .fc or .classifier)."
        )

    # Ensure output is (B, D) even if model returns (B, D, 1, 1)
    model = nn.Sequential(m, nn.Flatten(1))

    return FeatureExtractor(model=model, transforms=transforms, name=name, out_dim=out_dim)


# this is location for adding logic related to loading custom feature extractors in future
# e.g. user-defined models, other model zoos, etc.
# user provides a ModelSpec, we then must process and return a FeatureExtractor
def load_feature_extractor(model_spec: "ModelSpec", device: torch.device) -> FeatureExtractor:
    return _make_torchvision_feature_extractor(model_spec.name, device)


def to_unit_interval_01(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Per-dimension min-max scaling to [0,1] for a single dataset.
    z: (N, P), 2D array with one row per image and P embedding values per row.
    """
    z = np.asarray(z, dtype=np.float32)
    if z.size == 0:
        raise ValueError("Empty embeddings array.")
    if not np.isfinite(z).all():
        raise ValueError("Embeddings contain NaN/inf; cannot scale to [0,1].")
    zmin = z.min(axis=0, keepdims=True)
    zmax = z.max(axis=0, keepdims=True)
    denom = np.maximum(zmax - zmin, eps)
    z01 = (z - zmin) / denom
    return np.clip(z01, 0.0, 1.0)


# Dimensionality reduction (PCA):
# PCA standard way to compress vector dimension while keeping as much of original variation as possible.
# We learn compression step once on reference set, then reuse it to turn future embeddings into smaller,
# consistent representation.


@dataclass
class PcaProjector:
    _pca: PCA

    def transform(self, z: np.ndarray) -> np.ndarray:
        return self._pca.transform(z.astype(np.float32)).astype(np.float32)


def pca_projector(z_ref: np.ndarray, out_dim: int) -> PcaProjector:
    """Fit PCA on reference data."""
    _, d = z_ref.shape  # N samples, D dimensions

    if out_dim >= d:
        raise ValueError(
            f"Cannot fit PCA projector to out_dim={out_dim} >= original dimension {d}. "
            "PCA projector only supports reducing dimension."
        )

    pca = PCA(n_components=out_dim)
    pca.fit(z_ref.astype(np.float32))

    return PcaProjector(_pca=pca)
