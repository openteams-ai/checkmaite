# Setup Guide

## Creating an environment

We provide both poetry-based environment and conda-based environment options: 

<details>
<summary>Setup conda environment</summary>

To set up a conda environment, install `conda` on your machine and build the environment by running:

```bash
conda env create -f environment.yml -n jatic_env
```

Then activate the environment

```bash
conda activate jatic_env
```

This project utilizes `pre-commit` for linting and formatting. Install the `pre-commit` hooks using: 

```bash
pre-commit install
```

Finally, install the package itself

```bash
pip install -e .
```

</details>

<details>
<summary>Setup poetry environment</summary>

To set up a poetry environment, install `poetry` (the minimum supported version is `2.0.0`):

**osx / linux / bash on windows command**
```bash
curl -sSL https://install.python-poetry.org | python3 - --version 2.1.2
```

**windows powershell command**
```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py - --version 2.1.2
```

You can build the environment by running:

```bash
poetry install --extras dev
```

This project utilizes `pre-commit` for linting and formatting. Install the `pre-commit` hooks using: 

```bash
poetry run pre-commit install
```

Finally, install the package itself

```bash
poetry run pip install -e .
```
</details>

## Testing

(*In the following, instructions are only provided for `poetry`. Similar instructions are valid for `conda`.*)

This project uses `pytest` for it's test suite. Run the full test suite with:

```bash
poetry run pytest tests -svv
```

There are also tests which run on real data. These are time consuming to run so they are skipped by default. 
You can run them manually with:

```bash
poetry run pytest tests -svv -m real_data
```

**NOTE** If you have a poetry environment installed, you can also use this bash command from the root of your cloned RI 
directory to run tests with coverage:
```bash
make test
```

## Linting and formatting

(*In the following, instructions are only provided for `poetry`. Similar instructions are valid for `conda`.*)

Linting and formatting are automated via `pre-commit` hooks. However, if you'd like to run them directly, you can run:

```bash
poetry run pre-commit run --all-files --verbose
```

**NOTE** If you have a poetry environment installed, you can also use this bash command from the root of your cloned RI 
directory to run all pre-commit hooks:
```bash
make format
```

Type checking is performed by `pyright`. This can be run using:

```bash
poetry run pyright src
```

## Building the docs

(*In the following, instructions are only provided for `poetry`. Similar instructions are valid for `conda`.*)

The documentation is built using [`mkdocs`](https://www.mkdocs.org/) and deployed via CI to GitLab Pages. The RI also makes use of the [`mkdocs-jupyter`](https://github.com/danielfrg/mkdocs-jupyter) plugin which allows the docs to be build from notebooks as well as the standard markdown.

The docs can be built locally in two different ways. To build the docs with a live-reloading server, use:

```bash
poetry run mkdocs serve
```

This will create a live server running on the local machine. Any changes made to the document will be live-reloaded in the local website. 

Alternately, the docs can be built locally as a static site similar to the process in CI by running:

```bash
poetry run mkdocs build --site-dir public
```

The `site-dir` flag is optional and it defaults to building the site under `./public` in the directory in which you ran the command. 

The RI documentation website is deployed at [https://jatic.pages.jatic.net/reference-implementation/reference-implementation](https://jatic.pages.jatic.net/reference-implementation/reference-implementation/).

## Setting minimum package versions

The JATIC Software Development Plan (SDP) requires that all dependencies include a minimum version. 
It is preferable that these minimums be valid minimums due to a real incompatibility with the 
previous version. However, discovering the true minimums in a complex environment is highly 
time consuming. For this reason, we ask that miminums be set (in compliance with the SDP), but 
that they be comment tagged as either "necessary" (you are aware of an incompatibility with the
previous version) or "arbitrary" (you set this version artitrarily and it may be lowered if
an issue with cross-compatibility arises). 


## Conda lock file

### Usage

The RI contains a conda-lock file for linux-64 which is intended to be a stable conda environment for the 
latest version of the RI. 

You can build a conda environment from the lockfile called `my-locked-env`, by running

```
conda-lock install -n my-locked-env
```

Some of the dependencies come from private repositories.  You should be prompted for username/PAT credentials for each of those repositories.  If you'd like to avoid this, run the following command after replacing `<YOUR-USERNAME>` and `<YOUR-PASSWORD>` with valid credentials.

```bash
git config --global url."https://<YOUR-USERNAME>:<YOUR-PASSWORD>@gitlab.jatic.net/".insteadOf "https://gitlab.jatic.net/"
```

Alternatively, if you'd prefer not to modify your global git config, you could instead modify your `~/.netrc` file.

```bash
echo "machine gitlab.jatic.net login <YOUR-USERNAME> password <YOUR-PASSWORD>" >> ~/.netrc
```

### Updating the lockfile

To update the lockfile (e.g. after changing enviroment-optional.yaml), run the following:

```
conda-lock -f environment-optional.yml -p linux-64
```
