import functools
import hashlib
import importlib
import json
import logging
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar

import numpy as np
import torch
from maite.protocols import ArrayLike

if TYPE_CHECKING:
    from torch import nn

P = ParamSpec("P")
R = TypeVar("R")

CHECKMAITE_PLUGINS_UNSUPPORTED_INSTALL_HINT = (
    "pip install 'checkmaite-plugins[unsupported] @ "
    "git+https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite-plugins.git@main'"
)


class MissingDependencyWarning(UserWarning):
    """Optional dependency missing for a code path."""


class CountAndDrop(logging.Filter):
    def __init__(self, predicate: Callable[[logging.LogRecord], bool]) -> None:
        super().__init__()
        self.predicate = predicate
        self.count = 0
        self.first = None

    def filter(self, record: logging.LogRecord) -> bool:
        if self.predicate(record):
            self.count += 1
            if self.first is None:
                self.first = record.getMessage()
            return False
        return True


@contextmanager
def squash_repeated_warnings(logger_prefix: str, match: Callable[[logging.LogRecord], bool]) -> Iterator[CountAndDrop]:
    """
    Temporarily suppress repeated warnings emitted under `logger_prefix` (e.g. "dataeval"),
    counting how many were suppressed and a sample message that was emitted.
    """
    lg = logging.getLogger(logger_prefix)
    filt = CountAndDrop(match)

    lg.addFilter(filt)

    try:
        yield filt

    finally:
        lg.removeFilter(filt)


def set_device(device: str | None | torch.device) -> torch.device:
    """Determine the appropriate `torch.device` based on the provided input.

    If `device` is None, it selects the best available option: "cuda" if a
    CUDA-capable GPU is available, "mps" if running on macOS with an Apple
    Metal backend, or "cpu" otherwise. If `device` is provided as a string,
    it must be a valid PyTorch device identifier such as "cpu", "cuda",
    "cuda:0", "mps", etc. For a complete list of valid device strings, see:
    https://pytorch.org/docs/stable/tensor_attributes.html#torch-device

    Parameters
    ----------
    device : str or None or torch.device
        The device to use. Can be a string (e.g., "cuda", "cpu"),
        a `torch.device` object, or None to auto-detect.

    Returns
    -------
    torch.device
        The selected `torch.device` object.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def get_index2label_from_model_config(
    config_path: str | Path, model_config: dict[str, Any], index2label_key: str
) -> dict[int, str]:
    """Extract index-to-label mapping from a model configuration.

    Parameters
    ----------
    config_path : str
        Path to the configuration file (used for error messages).
    model_config : dict[str, Any]
        The model configuration dictionary.
    index2label_key : str
        The key in `model_config` that holds the index-to-label mapping.

    Returns
    -------
    dict[int, str]
        A dictionary mapping class indices to label names.

    Raises
    ------
    FileNotFoundError
        If `index2label_key` is not found in `model_config`.
    TypeError
        If the value associated with `index2label_key` is not a list or
        dict.
    """
    if index2label_key not in model_config:
        raise FileNotFoundError(f"The config_file at {config_path} is missing a {index2label_key} key.")
    if isinstance(model_config[index2label_key], list):
        return dict(enumerate(model_config[index2label_key]))
    if isinstance(model_config[index2label_key], dict):
        return {int(key): val for key, val in model_config[index2label_key].items()}
    raise TypeError(f"index2label should be provided as a dict or list, not {type(model_config[index2label_key])}")


def get_default_index2label(
    torchvision_weights_constructor: Any,
) -> dict[int, str]:
    """Get the default index-to-label mapping from TorchVision weights.

    Parameters
    ----------
    torchvision_weights_constructor : Any
        The TorchVision weights constructor object (e.g.,
        `FasterRCNN_ResNet50_FPN_Weights`).

    Returns
    -------
    dict[int, str]
        A dictionary mapping class indices to label names.
    """
    default = torchvision_weights_constructor.DEFAULT
    return dict(enumerate(default.meta["categories"]))


def maybe_download_weights(
    model: Any,
    torchvision_weights_constructor: Any,
    device: torch.device,
    **kwargs: Any,
) -> "nn.Module":
    """Load a model with default TorchVision weights, downloading if necessary.

    Parameters
    ----------
    model : Any
        The model class (e.g.,
        `torchvision.models.detection.fasterrcnn_resnet50_fpn`).
    torchvision_weights_constructor : Any
        The TorchVision weights constructor (e.g.,
        `FasterRCNN_ResNet50_FPN_Weights`).
    device : torch.device
        The device to move the model to.
    **kwargs : Any
        Additional keyword arguments to pass to the model constructor.

    Returns
    -------
    nn.Module
        The instantiated model with loaded weights, moved to the specified
        device.
    """
    # if weights not already in cache, they are downloaded here
    default = torchvision_weights_constructor.DEFAULT
    return model(weights=default, **kwargs).to(device)


def validate_input_batch(input_batch: Sequence[ArrayLike]) -> None:
    """Validate the format and consistency of an input batch of images.

    Checks for CHW ordering and consistent shapes across images in the batch.

    Parameters
    ----------
    input_batch : Sequence[ArrayLike]
        A sequence of image-like arrays.

    Raises
    ------
    ValueError
        If input data is not CHW-ordered or if images in the batch have
        inconsistent shapes.
    """
    total_channels, orig_img_height, orig_img_width = np.asarray(input_batch[0]).shape
    # channels can be used as a proxy to confirm CHW-ordering
    if not (1 <= total_channels <= 4):
        raise ValueError(
            f"Input data must follow CHW-ordering, current shape: {total_channels, orig_img_height, orig_img_width}"
        )
    for val in input_batch:
        # required to convert to array to appease type-checker...
        npy_array = np.asarray(val)  # creates view, not copy
        if npy_array.shape != (total_channels, orig_img_height, orig_img_width):
            raise ValueError(
                f"All input images currently required to have identical shape, {npy_array.shape} "
                f"not equal to {(total_channels, orig_img_height, orig_img_width)}. Please "
                "contact checkmaite team if your use case requires unevenly shaped images."
            )


def to_torch_batch(input_batch: Sequence[ArrayLike], device: torch.device) -> torch.Tensor:
    """Convert a sequence of array-like images to a PyTorch tensor batch.

    Parameters
    ----------
    input_batch : Sequence[ArrayLike]
        A sequence of image-like arrays.
    device : torch.device
        The device to move the resulting tensor to.

    Returns
    -------
    torch.Tensor
        A PyTorch tensor representing the batch of images.
    """
    # we are not writing to the underlying array in this method and hence we
    # suppress this warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors.",
            category=UserWarning,
        )
        return torch.stack([torch.as_tensor(obj, device=device) for obj in input_batch])


def id_hash(**kwargs: Any) -> str:
    """Generate a consistent hash from keyword arguments.

    Parameters
    ----------
    **kwargs : Any
        Key-value pairs to include in the hash generation

    Returns
    -------
    str
        First 8 characters of the SHA-256 hash of the JSON-serialized kwargs
    """
    return hashlib.sha256(json.dumps(kwargs, default=str, sort_keys=True).encode()).hexdigest()[:8]


def deprecated(*, replacement: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function as deprecated, emitting a DeprecationWarning on call."""

    def deco(func: Callable[P, R]) -> Callable[P, R]:
        msg = f"'{func.__qualname__}' is deprecated."
        if replacement:
            msg += f" Use '{replacement}' instead."

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return deco


def requires_optional_dependency(
    module_name: str,
    *,
    install_hint: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Require an optional dependency; if missing, raise ImportError with an install hint."""

    def deco(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                importlib.import_module(module_name)
            except ImportError:
                hint = f"\nInstall: {install_hint}" if install_hint else ""
                msg = (
                    f"'{func.__qualname__}' requires optional dependency '{module_name}', which is not installed.{hint}"
                )
                raise ImportError(msg) from None
            return func(*args, **kwargs)

        return wrapper

    return deco


JaticOnnxIoInterface = Literal["IMAGE_CLASSIFICATION", "IMAGE_OBJECT_DETECTION"]

JATIC_ONNX_INTERFACE_NAME = "JATIC_ONNX"
JATIC_ONNX_INTERFACE_VERSION = "v1"
IMAGE_CLASSIFICATION_INTERFACE: JaticOnnxIoInterface = "IMAGE_CLASSIFICATION"
IMAGE_OBJECT_DETECTION_INTERFACE: JaticOnnxIoInterface = "IMAGE_OBJECT_DETECTION"
ONNX_INSTALL_HINT = "Install ONNX support with `pip install checkmaite[onnx]`."


def load_jatic_onnx_metadata(
    config_path: str | Path,
    *,
    expected_io_interface: JaticOnnxIoInterface,
    index2label_key: str = "index2label",
) -> tuple[dict[str, Any], dict[int, str]]:
    """Load and validate a JATIC_ONNX metadata JSON file.

    The JATIC Interoperability Requirements specify that ONNX model input/output metadata should be provided alongside
    the model in a metadata file such as ``model-metadata.json``. The standard fields identify the JATIC_ONNX interface
    version, the CV task interface, input channel/size constraints, and output dimensions. checkmaite additionally
    requires model wrappers to expose ``index2label`` metadata, so this loader requires that mapping in the same JSON
    file.

    Args:
        config_path: Path to the JATIC_ONNX metadata JSON file.
        expected_io_interface: Task interface expected by the caller, e.g. ``IMAGE_CLASSIFICATION`` or
            ``IMAGE_OBJECT_DETECTION``.
        index2label_key: Metadata key containing a list or mapping from class indices to class labels.

    Returns:
        The parsed metadata dictionary and normalized ``dict[int, str]`` label mapping.

    Raises:
        FileNotFoundError: If the metadata file or required ``index2label`` key is missing.
        TypeError: If the metadata file is not a JSON object or ``index2label`` has an unsupported type.
        ValueError: If the metadata does not declare the expected JATIC_ONNX interface or task.
    """
    try:
        with open(config_path) as f:
            metadata = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at path: {config_path}") from None

    if not isinstance(metadata, dict):
        raise TypeError(f"Configuration file at {config_path} must contain a JSON object.")

    interface = _require_dict(metadata, "interface", config_path)
    if interface.get("name") != JATIC_ONNX_INTERFACE_NAME:
        raise ValueError(
            f"ONNX metadata interface.name must be {JATIC_ONNX_INTERFACE_NAME!r}, got {interface.get('name')!r}."
        )
    if interface.get("version") != JATIC_ONNX_INTERFACE_VERSION:
        raise ValueError(
            f"ONNX metadata interface.version must be {JATIC_ONNX_INTERFACE_VERSION!r}, "
            f"got {interface.get('version')!r}."
        )

    io = _require_dict(metadata, "io", config_path)
    if io.get("interface") != expected_io_interface:
        raise ValueError(f"ONNX metadata io.interface must be {expected_io_interface!r}, got {io.get('interface')!r}.")
    _require_dict(io, "input", config_path)
    _require_dict(io, "output", config_path)

    index2label = get_index2label_from_model_config(config_path, metadata, index2label_key)
    return metadata, index2label


def validate_jatic_onnx_session(session: Any, *, expected_outputs: set[str]) -> None:
    """Validate ONNX Runtime input/output names against the JATIC_ONNX contract.

    ONNX itself can execute many graph shapes and naming conventions. The JATIC Interoperability Requirements
    intentionally constrain the model surface that JATIC products must consume: one input named ``image`` and
    task-specific outputs such as ``scores`` for image classification or ``boxes`` plus ``scores`` for object detection.
    This validation makes non-compliant exports fail early with a clear message before inference is attempted.

    Args:
        session: An ``onnxruntime.InferenceSession`` or compatible test double.
        expected_outputs: Exact set of output tensor names required by the task wrapper.

    Raises:
        ValueError: If the session input/output names do not match JATIC_ONNX v1 expectations.
    """
    input_names = [inp.name for inp in session.get_inputs()]
    if input_names != ["image"]:
        raise ValueError(f"JATIC ONNX models must have exactly one input named 'image', got {input_names}.")

    output_names = {out.name for out in session.get_outputs()}
    if output_names != expected_outputs:
        raise ValueError(
            f"JATIC ONNX model outputs must be exactly {sorted(expected_outputs)}, got {sorted(output_names)}."
        )


def get_onnx_providers(device: str | torch.device | None) -> tuple[torch.device, list[str]]:
    """Translate a checkmaite device request into ONNX Runtime execution providers.

    Existing checkmaite model wrappers accept torch-style device strings such as ``"cpu"`` and ``"cuda"``. ONNX Runtime
    selects hardware through execution providers instead. This helper preserves the wrapper-facing device API while
    selecting provider lists that ONNX Runtime understands. If no device is requested, the helper prefers CUDA when
    available, then CoreML/MPS on Apple platforms, and finally CPU.

    Args:
        device: ``None`` for automatic provider selection, or a torch-style device string/object.

    Returns:
        A torch device used for checkmaite metadata and the ONNX Runtime provider preference list.

    Raises:
        RuntimeError: If a requested accelerator provider is unavailable in the installed ONNX Runtime package.
        ValueError: If the requested device type is not supported by this wrapper.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            f"JATIC_ONNX model wrappers require optional dependency 'onnxruntime'. {ONNX_INSTALL_HINT}"
        ) from None

    available = set(ort.get_available_providers())

    if device is None:
        if "CUDAExecutionProvider" in available:
            return torch.device("cuda"), ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CoreMLExecutionProvider" in available:
            return torch.device("mps"), ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return torch.device("cpu"), ["CPUExecutionProvider"]

    torch_device = set_device(device)
    requested = torch_device.type

    if requested == "cpu":
        return torch_device, ["CPUExecutionProvider"]
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDA was requested for ONNX inference, but CUDAExecutionProvider is not available. "
                f"Available ONNX Runtime providers: {sorted(available)}"
            )
        return torch_device, ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if requested == "mps":
        if "CoreMLExecutionProvider" not in available:
            raise RuntimeError(
                "MPS/CoreML was requested for ONNX inference, but CoreMLExecutionProvider is not available. "
                f"Available ONNX Runtime providers: {sorted(available)}"
            )
        return torch_device, ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    raise ValueError(f"Unsupported ONNX inference device: {torch_device}")


def prepare_jatic_onnx_image_batch(
    input_batch: Sequence[ArrayLike],
    metadata: dict[str, Any],
    *,
    batch_size: int | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Convert a checkmaite CHW image batch into the JATIC_ONNX input tensor.

    checkmaite datasets and model wrappers use CHW image arrays. JATIC_ONNX v1 requires a single input named ``image``
    containing an FP32 NCHW batch with pixel values normalized to ``[0, 1]``. The metadata file declares whether the
    model expects RGB or grayscale images and whether height, width, or batch size are fixed. The JATIC Interoperability
    Requirements also allow user-provided height, width, and batch-size settings to override the metadata; the optional
    keyword arguments support that override path.

    Args:
        input_batch: Sequence of CHW image-like arrays.
        metadata: Parsed JATIC_ONNX metadata dictionary.
        batch_size: Optional runtime batch-size override. ``-1`` in metadata means unlimited.
        image_height: Optional runtime input-height override. ``-1`` in metadata means keep current height.
        image_width: Optional runtime input-width override. ``-1`` in metadata means keep current width.

    Returns:
        A normalized FP32 NCHW NumPy batch and each input image's original ``(height, width)``. Object-detection
        wrappers use the original sizes to convert JATIC_ONNX normalized boxes back to checkmaite pixel-coordinate
        boxes.

    Raises:
        ValueError: If inputs are not CHW, batch size exceeds the configured limit, or channel conversion is
            unsupported.
    """
    validate_input_batch(input_batch)

    io = _require_dict(metadata, "io", "metadata")
    input_meta = _require_dict(io, "input", "metadata")

    metadata_batch_size = int(batch_size if batch_size is not None else io.get("batchSize", -1))
    if metadata_batch_size != -1 and len(input_batch) > metadata_batch_size:
        raise ValueError(
            f"Input batch has {len(input_batch)} images, but ONNX metadata batchSize is {metadata_batch_size}."
        )

    channels = str(input_meta.get("channels", "RGB")).upper()
    if channels not in {"RGB", "GRAYSCALE"}:
        raise ValueError(f"ONNX metadata io.input.channels must be 'RGB' or 'GRAYSCALE', got {channels!r}.")

    target_height = int(image_height if image_height is not None else input_meta.get("height", -1))
    target_width = int(image_width if image_width is not None else input_meta.get("width", -1))

    arrays = []
    original_sizes = []
    for image in input_batch:
        arr = np.asarray(image)
        _, orig_h, orig_w = arr.shape
        original_sizes.append((orig_h, orig_w))
        # Normalize before channel conversion: RGB/RGBA -> grayscale luminance emits float32, so integer images
        # must be scaled to [0, 1] before that conversion.
        arrays.append(_convert_channels(_normalize_image(arr), channels))

    batch = np.stack(arrays).astype(np.float32, copy=False)

    if target_height != -1 or target_width != -1:
        _, _, current_height, current_width = batch.shape
        resize_height = current_height if target_height == -1 else target_height
        resize_width = current_width if target_width == -1 else target_width
        if (resize_height, resize_width) != (current_height, current_width):
            tensor = torch.from_numpy(batch)
            batch = torch.nn.functional.interpolate(
                tensor,
                size=(resize_height, resize_width),
                mode="bilinear",
                align_corners=False,
            ).numpy()

    return batch.astype(np.float32, copy=False), original_sizes


def _require_dict(obj: dict[str, Any], key: str, source: str | Path) -> dict[str, Any]:
    value = obj.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"ONNX metadata at {source} must include object field {key!r}.")
    return value


def _convert_channels(arr: np.ndarray, channels: str) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Input image must have CHW shape, got {arr.shape}.")

    channel_count = arr.shape[0]
    if channels == "RGB":
        if channel_count == 3:
            return arr
        if channel_count == 4:
            return arr[:3]
        if channel_count == 1:
            return np.repeat(arr, 3, axis=0)
    elif channels == "GRAYSCALE":
        if channel_count == 1:
            return arr
        if channel_count in {3, 4}:
            # Convert RGB/RGBA to luminance using the standard luma coefficients for red, green, and blue.
            # The tensordot applies those coefficients across the channel axis and leaves the image height/width
            # unchanged; [None, ...] restores the CHW convention by adding the single grayscale channel dimension.
            rgb = arr[:3].astype(np.float32, copy=False)
            return np.tensordot(np.array([0.299, 0.587, 0.114], dtype=np.float32), rgb, axes=(0, 0))[None, ...]

    raise ValueError(f"Cannot convert input with {channel_count} channel(s) to {channels}.")


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        if arr.size and np.min(arr) < 0:
            raise ValueError("Integer image inputs must be non-negative before normalization.")
        info = np.iinfo(arr.dtype)
        return arr.astype(np.float32) / float(info.max)

    out = arr.astype(np.float32, copy=False)
    if out.size and (not np.all(np.isfinite(out)) or np.min(out) < 0.0 or np.max(out) > 1.0):
        raise ValueError("Float image inputs for JATIC_ONNX must contain finite values in the range [0, 1].")
    return out
