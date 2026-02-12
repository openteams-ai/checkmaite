# Setup RI

## Clone the project repository

Ensure you have the right permissions, and clone the project repository:

```bash
git clone https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation.git
```

## Create environment and install the package

We provide both poetry-based environment and conda-based environment options to setup your environment and install RI.

### Option 1: poetry environment

To set up a poetry environment, [install `poetry`](https://python-poetry.org/docs/#installation) (the minimum supported version is `2.0.0`), and build the environment with:

```bash
poetry install
```

JATIC tools which are not under active development have been moved to a poetry extras optional group called `unsupported`. Use these tools with caution because they may contain bugs or may be incompatible with the environment in the future. You can install these with:

```bash
poetry install --extras unsupported
```

### Option 2: conda environment

To set up a conda environment, [install `conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) on your machine and build the environment with:

```bash
# create env with conda-lock installed
conda create -n jatic-ri "conda-lock>=3"

# activate the environment
conda activate jatic-ri

# use conda-lock to install dependencies
CONDA_ENV_NAME=jatic-ri make conda-env

# finally, install the reference implementation package
pip install .
```
