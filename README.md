# CheckMAITE

`CheckMAITE` is an integration point for AI T&E tooling for computer vision that is based on the `maite` protocols (see [maite documentation](https://mit-ll-ai-technology.github.io/maite/)).

## Description

**What is CheckMAITE?**
CheckMAITE is a Python API which makes testing and evaluation of models and datasets straightforward and reproducible. It can be used to run a wide variety of **Model Evaluation** and **Dataset Analysis** investigations for both **Object Detection** and **Image Classification** computer vision problems.

It was built as an integration point for all [CDAO JATIC](https://cdao.pages.jatic.net/public/) tools and has since expanded focus. Our goal is to make T&E easier for analysts!

To learn more please visit our [published documentation](https://openteams-ai.github.io/checkmaite/).

## Supported environment

- **OS:** Linux x86_64 (CI-tested). macOS x86_64 and arm64 are supported but not CI-tested.
- **Python:** 3.10, 3.11, and 3.12 with `uv` or pip. conda supports 3.10 and 3.11 only. `checkmaite-plugins` is `<3.12` on both paths.
- **GPU:** CPU is the supported baseline. CUDA is optional for PyTorch.
- **Hardware, architectures, and internet access:** see [System requirements](https://openteams-ai.github.io/checkmaite/reference/system_requirements.html).
- **Container:** a batch container is built from the repository `Dockerfile` for Linux AMD64, with `cpu` and NVIDIA `cuda` targets. See [Run CheckMAITE in a container](https://openteams-ai.github.io/checkmaite/get-started/container.html).

## Limitations and prerequisites

<!-- --8<-- [start:limitations] -->

CheckMAITE evaluates models and datasets you supply.

- Built-in loaders cover COCO, YOLO, and VisDrone object detection, and YOLO image classification.
- Models and datasets expose an `index2label` map as `dict[int, str]`. Default torchvision weights build it from the weights. Custom torchvision weights and ONNX models need an `index2label` in their JSON config (default key `index2label`, override with `index2label_key`), given as a list or a dict; both are normalized to `dict[int, str]`. `num_classes` defaults to `len(index2label)`. For object detection, set it when the model head has a different class count (e.g. a background slot). For classification it must equal `len(index2label)`.
- Where a model's and a dataset's `index2label` overlap, a shared index must name the same class and a shared class name must use the same index. CheckMAITE does not validate this; mismatches produce silently wrong per-class results.
- ONNX models need the `onnx` (or `onnx-cuda`) extra. ONNX wrappers accept only `uint8` images or finite float images in `[0, 1]` (cast to `float32`); other integer dtypes raise `TypeError`. Non-finite or out-of-`[0, 1]` float images raise `ValueError`.
- Object-detection boxes are `float32` `xyxy` arrays of shape `(N, 4)`. See [Conventions][limitations-conventions].
- There is no web UI; configure and run capabilities through the Python API. PDF export needs the `reporting` extra.
- CPU is the supported baseline. Machine, OS, and Python bounds are in the [supported environment][limitations-supported-environment].

<!-- --8<-- [end:limitations] -->

[limitations-conventions]: https://openteams-ai.github.io/checkmaite/reference/conventions.html
[limitations-supported-environment]: #supported-environment

## Installation

For detailed installation instructions please refer to [Setup Guide](https://openteams-ai.github.io/checkmaite/get-started/install_setup.html).
To run CheckMAITE as a batch container instead of installing the package, see [Run CheckMAITE in a container](https://openteams-ai.github.io/checkmaite/get-started/container.html).

On macOS, `brew install libomp` is required (`dataeval` loads LightGBM, which needs `libomp.dylib`).

## Usage

To learn how to get started using `CheckMAITE`, please visit our [quick start guide](https://openteams-ai.github.io/checkmaite/get-started/index.html).

There are also [How-to Guides](https://openteams-ai.github.io/checkmaite/how-to/index.html) & [Tutorials](https://openteams-ai.github.io/checkmaite/tool-usage/index.html) that will help get new users up to speed on the available functionality.

## Contributing

The `CheckMAITE` team welcomes contributions of all forms - questions, documentation, and code contributions. Please visit our [contribution guide](https://openteams-ai.github.io/checkmaite/development/contributing.html).

## Authors and acknowledgment

This project was created for [CDAO JATIC](https://cdao.pages.jatic.net/public/) and is maintained by OpenTeams with collaborative community support. 

### CDAO Funding Acknowledgment

<!-- --8<-- [start:acknowledgment] -->

This material is based upon work supported by the Chief Digital and Artificial
Intelligence Office under Contract No. W519TC-25-9-2041. The views and
conclusions contained herein are those of the author(s) and should not be
interpreted as necessarily representing the official policies or endorsements,
either expressed or implied, of the U.S. Government.

<!-- --8<-- [end:acknowledgment] -->
