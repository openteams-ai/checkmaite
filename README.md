# CheckMAITE

`CheckMAITE` is an integration point for AI T&E tooling for computer vision that is based on the `maite` protocols (see [maite documentation](https://mit-ll-ai-technology.github.io/maite/)).

## Description

**What is CheckMAITE?**
CheckMAITE is an API and a UI application which makes testing and evaluation of models and datasets straightforward and reproducible. It can be used to run a wide variety of **Model Evaluation** and **Dataset Analysis** investigations for both **Object Detection** and **Image Classification** computer vision problems.

It was built as an integration point for all [CDAO JATIC](https://cdao.pages.jatic.net/public/) tools and has since expanded focus. Our goal is to make T&E easier for analysts!

To learn more please visit our [published documentation](https://openteams-ai.github.io/checkmaite/).

## Supported environment

- **OS:** Linux x86_64 (CI-tested). macOS x86_64 and arm64 are supported but not CI-tested.
- **Python:** 3.10, 3.11, and 3.12 with `uv` or pip. conda supports 3.10 and 3.11 only. `checkmaite-plugins` is `<3.12` on both paths.
- **GPU:** CPU is the supported baseline. CUDA is optional for PyTorch.
- **Hardware:** any machine that can install those Python versions.

## Installation

For detailed installation instructions please refer to [Setup Guide](https://openteams-ai.github.io/checkmaite/get-started/install_setup.html)

## Usage

To learn how to get started using `CheckMAITE`, please visit our [quick start guide](https://openteams-ai.github.io/checkmaite/get-started/index.html).

There are also [How-to Guides](https://openteams-ai.github.io/checkmaite/development/index.html) & [Tutorials](https://openteams-ai.github.io/checkmaite/tool-usage/index.html) that will help get new users up to speed on the available functionality.

## Contributing

The `CheckMAITE` team welcomes contributions of all forms - questions, documentation, and code contributions. Please visit our [contribution guide](https://openteams-ai.github.io/checkmaite/development/contributing.html).

## Authors and acknowledgment

This project was created for [CDAO JATIC](https://cdao.pages.jatic.net/public/) and is maintained by OpenTeams with collaborative community support. 
