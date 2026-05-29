# ONNX model wrappers

checkmaite supports ONNX computer-vision models through the JATIC_ONNX interface defined in the JATIC Interoperability Requirements.

The important design constraint is that checkmaite does **not** try to interpret every arbitrary ONNX export. ONNX provides a portable graph/runtime format, while JATIC_ONNX provides the task-level input/output convention that lets checkmaite convert model calls into MAITE-compatible predictions.

## Recommended custom model paths

Use the narrowest model-integration path that fits the model:

- Use existing torchvision wrappers for supported torchvision architectures.
- Use JATIC_ONNX for generic model interchange when the model can be exported with the JATIC_ONNX input/output contract.
- Use checkmaite plugins for Python/PyTorch models that require custom architecture code, preprocessing, or postprocessing.

checkmaite does not currently load arbitrary local Python model modules from the UI. Custom Python execution should live in a wrapper or plugin where it can be tested and reviewed explicitly.

## Supported interface

The wrappers currently target `JATIC_ONNX` version `v1` for:

- image classification (`IMAGE_CLASSIFICATION`)
- object detection (`IMAGE_OBJECT_DETECTION`)

The model is loaded from `model_weights_path`, which should point to an `.onnx` file. The model metadata is loaded from `model_config_path`, which should point to a JSON file containing the JATIC_ONNX fields and checkmaite's required `index2label` mapping.

A minimal metadata file looks like this:

```json
{
  "interface": {
    "name": "JATIC_ONNX",
    "version": "v1"
  },
  "io": {
    "batchSize": 1,
    "interface": "IMAGE_OBJECT_DETECTION",
    "input": {
      "channels": "RGB",
      "height": 640,
      "width": 640
    },
    "output": {
      "nBoxes": 100,
      "nClasses": 3
    }
  },
  "index2label": {
    "0": "background",
    "1": "person",
    "2": "vehicle"
  }
}
```

`index2label` is not called out as an ONNX model output by the JATIC Interoperability Requirements, but checkmaite model wrappers conventionally expose it in `metadata` and as `model.index2label`. The ONNX wrappers therefore require it in the same metadata file.

## Input assumptions

JATIC_ONNX v1 expects exactly one ONNX input tensor:

- name: `image`
- dtype: FP32
- shape: NCHW
- values: normalized pixels in `[0, 1]`

checkmaite callers still provide CHW image arrays. The shared ONNX helpers in `src/checkmaite/core/_utils.py` validate CHW input, convert channels according to metadata (`RGB` or `GRAYSCALE`), normalize integer images to `[0, 1]`, stack the batch, and resize to configured height/width when needed.

`io.batchSize`, `io.input.height`, and `io.input.width` are read from metadata. The wrappers also accept runtime overrides for `batch_size`, `image_height`, and `image_width`, matching the JATIC Interoperability Requirements expectation that user configuration should take precedence over metadata.

## Output assumptions

### Image classification

`OnnxICModel` expects exactly one output:

- `scores`: FP32 tensor of shape `(batchSize, nClasses)`

The tensor is interpreted as class probabilities and returned as MAITE image-classification targets.

### Object detection

`OnnxODModel` expects exactly two outputs:

- `boxes`: FP32 tensor of shape `(batchSize, nBoxes, 4)`
- `scores`: FP32 tensor of shape `(batchSize, nBoxes, nClasses)`

JATIC_ONNX boxes are normalized `(x0, y0, x1, y1)` coordinates in `[0, 1]`. checkmaite object-detection targets conventionally use pixel-coordinate `xyxy` boxes, so the wrapper scales boxes back to each original input image's width and height before returning `DetectionTarget` objects.

Class labels and confidence scores are derived per box with:

```python
labels = argmax(scores, axis=-1)
scores = max(scores, axis=-1)
```

The wrapper does not currently filter low-confidence detections or apply NMS. A JATIC_ONNX object-detection model should include any required postprocessing in the exported graph if it wants the wrapper to receive final candidate boxes.

## Runtime dependency extras

ONNX support is optional. Install the portable ONNX extra for CPU inference and platform providers bundled with the regular ONNX Runtime package:

```bash
pip install "checkmaite[onnx]"
```

This installs:

- `onnx`: validates/checks the ONNX model file.
- `onnxruntime`: runs inference on CPU and may expose additional non-CUDA providers depending on the platform.

On Mac arm64 / Apple Silicon, use `checkmaite[onnx]`. The regular `onnxruntime` package can expose `CoreMLExecutionProvider`, which is the ONNX Runtime provider that checkmaite uses when `device="mps"` is requested. There is no separate `onnxruntime-mps` package, and `onnxruntime-gpu` is not the correct package for Apple GPUs.

For NVIDIA CUDA environments, install the CUDA-specific extra instead:

```bash
pip install "checkmaite[onnx-cuda]"
```

This installs `onnxruntime-gpu` instead of `onnxruntime`. Use this only in environments where the CUDA ONNX Runtime wheel is available and compatible with the installed Python/CUDA stack. In particular, this extra is not expected to resolve on Mac arm64.

## Provider selection

Provider selection happens in `get_onnx_providers`:

- explicit `device="cpu"` uses `CPUExecutionProvider`
- explicit `device="cuda"` requires `CUDAExecutionProvider`
- explicit `device="mps"` maps to `CoreMLExecutionProvider`
- `device=None` prefers CUDA, then CoreML, then CPU based on the installed runtime's available providers

If an accelerator is requested but the installed ONNX Runtime package does not expose the corresponding provider, the wrapper raises a clear error showing the available providers.

## ONNX model checking

The wrappers load models through ONNX Runtime by default. To also run `onnx.checker.check_model` during wrapper initialization, pass `validate_onnx=True`. This extra check parses the model before ONNX Runtime loads it, so it is opt-in to avoid double-loading large ONNX files.
