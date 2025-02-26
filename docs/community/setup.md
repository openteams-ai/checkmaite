# Setup Guide
Set up your environment. Both a poetry-based environment and conda-based environment options. 

<details>
<summary>Setup conda environment</summary>

To set up a conda environment, install `conda` (or `mamba`, but these 
instructions are based on conda) on your machine and build the 
environment by running:

```bash
conda env create -f environment.yml -n jatic_env
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

Alternatively, if you have `poetry` and `make` installed on your machine, you can build the environment by running: 

```bash
make init
```

This will create a virtual environment with all regular dependencies, developer dependencies,
and pre-commit hooks installed. This environment can be found at can be found at `[your cloned RI directory]/.venv/`
and can be manually activated by running:
```bash
source ./venv/bin/activate
```

## Testing

This project uses `pytest` for it's test suite. Run the full test suite with:

```bash
pytest tests -svv
```

There are also tests which run on real data. These are time consuming to run so they are skipped by default. 
You can run them manually with:

```bash
pytest tests -svv -m real_data
```

NOTE:
If you have a poetry environment installed, you can also use this bash command from the root of your cloned RI 
directory to run tests with coverage:
```bash
make test
```

## Linting and formatting

Linting and formatting are automated via `pre-commit` hooks. However, if you'd like to run them directly, you can run:

```bash
pre-commit run --all-files --verbose
```

NOTE:
If you have a poetry environment installed, you can also use this bash command from the root of your cloned RI 
directory to run all pre-commit hooks:
```bash
make format
```

Type checking is performed by `pyright`. The CI must report a type-completeness score of 100% for the public API. This can be run using:

```bash
pyright --ignoreexternal --verifytypes jatic_ri
```

## Building the docs

The documentation is built using [`mkdocs`](https://www.mkdocs.org/) and deployed via CI to GitLab Pages. The RI also makes use of the [`mkdocs-jupyter`](https://github.com/danielfrg/mkdocs-jupyter) plugin which allows the docs to be build from notebooks as well as the standard markdown.

The docs can be built locally in two different ways. To build the docs with a live-reloading server, use:

```bash
mkdocs serve
```

This will create a live server running on the local machine. Any changes made to the document will be live-reloaded in the local website. 

Alternately, the docs can be built locally as a static site similar to the process in CI by running:

```bash
mkdocs build --site-dir site
```

The `site-dir` flag is optional and it defaults to building the site under `./site` in the directory in which you ran the command. 

The RI documentation website is deployed at [https://jatic.pages.jatic.net/reference-implementation/reference-implementation](https://jatic.pages.jatic.net/reference-implementation/reference-implementation/).

## Setting minimum package versions

The JATIC Software Development Plan (SDP) requires that all dependencies include a minimum version. 
It is preferable that these minimums be valid minimums due to a real incompatibility with the 
previous version. However, discovering the true minimums in a complex environment is highly 
time consuming. For this reason, we ask that miminums be set (in compliance with the SDP), but 
that they be comment tagged as either "necessary" (you are aware of an incompatibility with the
previous version) or "arbitrary" (you set this version artitrarily and it may be lowered if
an issue with cross-compatibility arises). 


## Maintaining and using the conda lock file

The RI contains a conda-lock file for linux-64 which is intended to be a stable conda environment for the 
latest version of the RI. 

The lockfile contains placeholders for private Gitlab username and tokens. 
In CI, these are replaced with valid tokens. To use this file locally, you will need to replace 
`gitlab-ci-token` with your Gitlab username and `${PRIVATE_TOKEN}` with your Personal Access Token (PAT). 

Once that is done, you can build a conda environment from the lockfile called `my-locked-env`, by running

```
conda-lock install -n my-locked-env
```

To update the lockfile, you'll first need to update the `environment-optional.yml` by replacing 
`gitlab-ci-token` with your Gitlab username and `${PRIVATE_TOKEN}` with your Personal Access Token (PAT). Then create the lockfile with:

```
conda-lock -f environment-optional.yml -p linux-64
```

Before committing this file to the repository, you will need to scrub your personal username and PAT
from the file and replace them (as above) with `gitlab-ci-token` and `${PRIVATE_TOKEN}`.
