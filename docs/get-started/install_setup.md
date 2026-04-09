# Setup `checkmaite`

## Clone the project repository

Ensure you have the right permissions, and clone the project repository:

```bash
git clone https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation.git
```

## Create environment and install the package

We provide both a poetry-based environment and conda-based environment option to setup your environment and install `checkmaite`.

### Option 1: poetry environment

To set up a poetry environment, [install `poetry`](https://python-poetry.org/docs/#installation) (the minimum supported version is `2.0.0`), and build the environment with:

```bash
poetry install
```

JATIC tools which are not under active development are distributed through the separate `checkmaite-plugins` package.
Use these tools with caution because they may contain bugs or may be incompatible with the environment in the future.

```bash
poetry install --extras unsupported
```

You can also install the plugin package directly:

```bash
pip install "checkmaite-plugins[unsupported]"
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
CONDA_ENV_NAME=checkmaite make conda-env

# finally, install the `checkmaite` package
pip install . --no-deps
```
