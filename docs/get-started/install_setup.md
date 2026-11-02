# Setup `CheckMAITE`

## Supported environment

- **OS:** Linux x86_64 (CI-tested). macOS x86_64 and arm64 are supported but not
  CI-tested. The conda lock is Linux x86_64 only.
- **Python:** 3.11, 3.12, 3.13, and 3.14 with `uv` or pip
  (`requires-python = ">=3.11, <3.15"`). conda supports 3.11, 3.12, and 3.13
  (`python = ">=3.11,<3.14"`): conda-forge `nrtk` requires a `simsimd` release
  that has no 3.14 build. The committed `conda-lock.yml` is resolved for 3.13.
  `checkmaite-plugins` is `<3.12` on all paths.
- **GPU:** CPU is the supported baseline. CUDA is optional for PyTorch.
- **Hardware, architectures, and internet access:** see [System requirements](../reference/system_requirements.md).
- **Container:** CheckMAITE is also available as a batch container for Linux
  AMD64, with CPU and NVIDIA CUDA variants, built from the repository
  `Dockerfile`. It needs no Python environment on the host. See
  [Run CheckMAITE in a container](container.md).

## Install from PyPI

CheckMAITE is published on [PyPI](https://pypi.org/project/checkmaite/). It is
not published on conda-forge; in a conda environment, install it from PyPI with
pip (see [conda](#conda) below).

On macOS, install OpenMP first (`dataeval` → LightGBM needs `libomp.dylib`):

```bash
brew install libomp
```

### pip

```bash
python -m pip install checkmaite
```

### uv

In a uv project:

```bash
uv add checkmaite
```

Or in an existing virtual environment:

```bash
uv pip install checkmaite
```

### Poetry

In a Poetry project:

```bash
poetry add checkmaite
```

### conda

```bash
conda create -n checkmaite python=3.12 pip
conda activate checkmaite
python -m pip install checkmaite
```

### Optional extras

Extras add optional features. Name them with any of the installers above, for
example `python -m pip install "checkmaite[reporting,onnx]"`,
`uv add "checkmaite[reporting]"` or `poetry add "checkmaite[reporting]"`.

| Extra | Adds |
| --- | --- |
| `reporting` | PDF report export (`markdown`, `xhtml2pdf`) |
| `onnx` | ONNX model wrappers on the CPU (`onnx`, `onnxruntime`) |
| `onnx-cuda` | ONNX model wrappers on NVIDIA GPUs (`onnx`, `onnxruntime-gpu`) |
| `aws`, `gcs`, `azure` | Datasets and analytics stores on S3, GCS, or Azure |
| `cloud` | All three cloud storage backends |

## Install from source

Install from source to use an unreleased version or to work on CheckMAITE.

### Clone the project repository

Ensure you have the right permissions, and clone the project repository:

```bash
git clone https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite.git
cd checkmaite
```

On macOS, install OpenMP before creating the environment (`dataeval` → LightGBM
needs `libomp.dylib`):

```bash
brew install libomp
```

### Create environment and install the package

uv, conda, pip, and Poetry are supported; uv is recommended.

#### Option 1: uv environment

To set up a uv environment, [install
`uv`](https://docs.astral.sh/uv/getting-started/installation/), and build the
environment with:

```bash
uv sync
```

Optional extras:

```bash
uv sync --extra reporting  # PDF report export (markdown + xhtml2pdf)
uv sync --extra cloud      # cloud storage dependencies (aws+gcs+azure)
uv sync --extra visdrone   # VisDrone (CenterNet) object-detection wrapper
uv sync --extra onnx       # JATIC_ONNX model wrappers (onnx-cuda for NVIDIA GPUs)
```

The base install covers the torchvision model wrappers only. The VisDrone and
ONNX wrappers need the `visdrone` and `onnx` (or `onnx-cuda`) extras, which
forward to modelmaite's extras of the same name; with pip, install
`"checkmaite[visdrone]"` or `"checkmaite[onnx]"`.

JATIC tools which are not under active development are distributed through the
separate `checkmaite-plugins` package.
Use these tools with caution because they may contain bugs or may be
incompatible with the environment in the future.
`checkmaite-plugins` supports Python `<3.12` and is installed directly from
GitLab instead of through a
`checkmaite` extra so that the PyPI package metadata remains valid.

```bash
uv pip install "checkmaite-plugins[unsupported] @ git+https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite-plugins.git@main"
uv pip install "setuptools<82" "pybsm<0.16.1"
```

`setuptools<82` keeps `gradient` importable (it uses `pkg_resources`), and
`pybsm<0.16.1` is needed because `pybsm` 0.16.1 requires setuptools 83 or later
([checkmaite-plugins#2](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite-plugins/-/work_items/2)).

If you are developing locally across both repositories:

```bash
pip install -e /path/to/checkmaite
pip install -e "/path/to/checkmaite-plugins[unsupported]"
```

#### Option 2: conda environment

To set up a conda environment, [install
`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
on your machine and build the environment with:

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

#### Option 3: pip

In a virtual environment, install the cloned repository with pip:

```bash
python -m pip install .
```

#### Option 4: Poetry

In your Poetry project, add the cloned repository as a path dependency:

```bash
poetry add /path/to/checkmaite
```

## Check the installation

```bash
python -c "import checkmaite; print(checkmaite.__version__)"
```

Then run the [example](index.md#example) in Get started.
