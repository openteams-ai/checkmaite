# Get started

`CheckMAITE` workflows are executed through a Python __API__ for programmatic
access via Python object interfaces.

Install CheckMAITE, then work through one of the end-to-end tutorials.

## Install

Install CheckMAITE from [PyPI](https://pypi.org/project/checkmaite/) with pip,
uv, Poetry, or pip in a conda environment:

```bash
python -m pip install checkmaite
```

[Setup](install_setup.md) has the commands for each installer, optional
extras, and installing from source.

## Supported machines

CheckMAITE is CI-tested on Linux x86_64. macOS x86_64 and arm64 (Apple
silicon) are supported but not CI-tested. A CPU is enough; an NVIDIA GPU is
optional. See the [supported
environment](install_setup.md#supported-environment) for Python versions and
the [system requirements](../reference/system_requirements.md) for memory,
disk, architectures, and internet access.

## Example

--8<-- "README.md:example"

## Next steps

<div class="grid cards" markdown >

* :material-monitor-arrow-down-variant:{ .lg .middle } [__Setup__ :octicons-arrow-right-24:](install_setup.md)

    ---
    Install and setup `CheckMAITE` on your local machine to start using it!

* :material-connection:{ .lg .middle } [__Object Detection Workflow via API__ :octicons-arrow-right-24:](checkmaite_api_od.ipynb)

    ---
    Your first tutorial: run every capability on an object detection model and dataset.

* :material-connection:{ .lg .middle } [__Image Classification Workflow via
  API__ :octicons-arrow-right-24:](checkmaite_api_ic.ipynb)

    ---
    The same end-to-end tutorial for image classification.

</div>

Next, see the [tutorials](../tool-usage/index.md) for each JATIC tool, the
[how-to guides](../how-to/index.md) for specific tasks, and the
[glossary](../reference/glossary.md) for key terms.
