# Setup `CheckMAITE`

## Supported environment

- **OS:** Linux x86_64 (CI-tested). macOS x86_64 and arm64 are supported but not CI-tested.
- **Python:** 3.10, 3.11, and 3.12 with `uv` or pip (`requires-python = ">=3.10, <3.13"`). conda supports 3.10 and 3.11 only (`python = ">=3.10,<3.12"`). `checkmaite-plugins` is `<3.12` on both paths.
- **GPU:** CPU is the supported baseline. CUDA is optional for PyTorch.
- **Hardware:** any machine that can install those Python versions.

## Clone the project repository

Ensure you have the right permissions, and clone the project repository:

```bash
git clone https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite.git
```

## Create environment and install the package

The supported install path is `uv`. We also provide a conda-based alternative.

### Option 1: uv environment

To set up a uv environment, [install `uv`](https://docs.astral.sh/uv/getting-started/installation/), and build the environment with:

```bash
uv sync
```

Optional extras:

```bash
uv sync --extra ui         # UI dependencies
uv sync --extra reporting  # PDF report export (markdown + xhtml2pdf)
uv sync --extra cloud      # cloud storage dependencies (aws+gcs+azure)
```

JATIC tools which are not under active development are distributed through the separate `checkmaite-plugins` package.
Use these tools with caution because they may contain bugs or may be incompatible with the environment in the future.
`checkmaite-plugins` supports Python `<3.12` and is installed directly from GitLab instead of through a
`checkmaite` extra so that the PyPI package metadata remains valid.

```bash
uv pip install "checkmaite-plugins[unsupported] @ git+https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite-plugins.git@main"
```

If you are developing locally across both repositories:

```bash
pip install -e /path/to/checkmaite
pip install -e "/path/to/checkmaite-plugins[unsupported]"
```

### Option 2: conda environment

To set up a conda environment, [install `conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) on your machine and build the environment with:

```bash
# create env with conda-lock installed
conda create -n checkmaite "conda-lock>=3"

# activate the environment
conda activate checkmaite

# use conda-lock to install dependencies
conda-lock install -n checkmaite conda-lock.yml

# finally, install the `checkmaite` package
pip install . --no-deps
```
