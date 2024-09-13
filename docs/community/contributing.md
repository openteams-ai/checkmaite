# Contribution Guide

The JATIC Reference Implementation team welcomes contributions in all forms.

## Development Setup

First, fork the repository and setup SSH keys. Then, create a local copy of the repository:

```bash
git clone git@gitlab.jatic.net:jatic/increment-demos/reference-implementation.git
```

Navigate to the package directory:

```bash
cd reference-implementation
```

Set up your environment. Both a poetry-based environment and conda-based environment options. 

<details>
<summary>Setup conda environment</summary>

To set up a conda environment, install `conda` (or `mamba`, but these 
instructions are based on conda) on your machine and build the 
environment by running:

```bash
conda env create -f environemnt.yml -n jatic_env
```

Then activate the environment

```bash
conda activate jatic_env
```

</details>

<details>
<summary>Setup poetry environment</summary>

TODO add poetry setup instructions

</details>

Install the package itself

```bash
pip install -e .
```

This project utilized `pre-commit` for linting and formatting. Install the 
`pre-commit` hooks using: 

```bash
pre-commit install
```

## Testing

This project uses `pytest` for it's test suite. Run the full test suite with:

```bash
pytest tests -svv
```

## Linting and formatting

Linting and formatting are automated via `pre-commit` hooks. However, if you'd like to run them directly, you can run:

```bash
pre-commit run --all-files --verbose
```

Type checking is performed by `pyright`. The CI must report a type-completeness score of 100% for the public API. This can be run using:

```bash
pyright --ignoreexternal --verifytypes jatic_ri
```

## Setting minimum package versions

The JATIC Software Development Plan (SDP) requires that all dependencies include a minimum version. 
It is preferable that these minimums be valid minimums due to a real incompatibility with the 
previous version. However, discovering the true minimums in a complex environment is highly 
time consuming. For this reason, we ask that miminums be set (in compliance with the SDP), but 
that they be comment tagged as either "necessary" (you are aware of an incompatibility with the
previous version) or "arbitrary" (you set this version artitrarily and it may be lowered if
an issue with cross-compatibility arises). 
