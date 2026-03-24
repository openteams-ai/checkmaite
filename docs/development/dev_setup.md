# Development Setup Guide

A guide for contributing developers.

## Clone the project repository

Ensure you have the right permissions, and clone the project repository:

```bash
# You need to set up personal-access-tokens
git clone https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation.git

# or

# You need to set up SSH keys
git clone git@gitlab.jatic.net:jatic/reference-implementation/reference-implementation.git
```

!!! warning "macOS: SSL certificates for non-Homebrew/non-conda Python"
    If you installed Python from [python.org](https://www.python.org/) (the framework installer), SSL connections will fail until you run the bundled certificate command:
    ```bash
    /Applications/Python\ 3.XX/Install\ Certificates.command
    ```
    Replace `3.XX` with your Python version (e.g. `3.12`). This is not needed if you installed Python via Homebrew or conda.

## Creating an environment

We provide both poetry-based environment and conda-based environment options:

### Option 1: Setup poetry environment

To set up a poetry environment, [install `poetry`](https://python-poetry.org/docs/#installation) (the minimum supported version is `2.0.0`).

#### For regular users

You can build the environment by running:

```bash
poetry install
```

#### For package developers

If you plan to contribute to the project or need development tools, install with development dependencies:

```bash
poetry install --with dev
```

To contribute to the documentation, install with the docs dependencies:

```bash
poetry install --with docs
```

There are also two optional dependency groups, `unsupported` and `ui`. You can install these via poetry:

```bash
poetry install --extras ui
```

If you want to install all the extras, you can use:

```bash
poetry install --all-extras
```

And if you want to install everything and the kitchen sink:

```bash
poetry install --with dev --all-extras
```

This project utilizes `pre-commit` for linting and formatting. **Developers** should also install the `pre-commit` hooks using:

```bash
poetry run pre-commit install
```

### Option 2: Setup conda environment

To set up a conda environment, [install `conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) on your machine and build the environment by running:

```bash
# create env with conda-lock installed
conda create -n checkmaite "conda-lock>=3"
# activate the environment
conda activate checkmaite
# use conda-lock to install dependencies
CONDA_ENV_NAME=checkmaite make conda-env
# finally, install the `checkmaite` package
pip install -e . --no-deps
```

This project utilizes `pre-commit` for linting and formatting. Install the `pre-commit` hooks using:

```bash
pre-commit install
```

## Testing

(*In the following, instructions are only provided for `poetry`. Similar instructions are valid for `conda`.*)

This project uses `pytest` for it's test suite. Run the full test suite with:

```bash
poetry run pytest tests -svv
```

This project also contains test markers to flag special test groups. The tests marked `real_data` are
time consuming tests that utilize real data. The tests marked `unsupported` are dependent on non-active 
JATIC tools which are not installed in the environment by default. Both of these test groups are skipped
by default and run nightly in CI. 

You can run them manually with:

```bash
poetry run pytest tests -svv -m real_data
```
or

```bash
poetry run pytest tests -svv -m "real_data and unsupported"
```

**NOTE** If you have a poetry environment installed, you can also use this bash command from the root of your cloned checkmaite
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

**NOTE** If you have a poetry environment installed, you can also use this bash command from the root of your cloned checkmaite directory to run all pre-commit hooks:

```bash
make format
```

Type checking is performed by `pyright`. This can be run using:

```bash
poetry run pyright src
```

## Building the docs

(*In the following, instructions are only provided for `poetry`. Similar instructions are valid for `conda`.*)

The documentation is built using [`mkdocs`](https://www.mkdocs.org/) and deployed via CI to GitLab Pages. checkmaite also makes use of the [`mkdocs-jupyter`](https://github.com/danielfrg/mkdocs-jupyter) plugin which allows the docs to be build from notebooks as well as the standard markdown.

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

The checkmaite documentation website is deployed at [https://jatic.pages.jatic.net/reference-implementation/reference-implementation](https://jatic.pages.jatic.net/reference-implementation/reference-implementation/).

## Setting minimum package versions

The JATIC Software Development Plan (SDP) requires that all dependencies include a minimum version.
It is preferable that these minimums be valid minimums due to a real incompatibility with the
previous version. However, discovering the true minimums in a complex environment is highly
time consuming. For this reason, we ask that miminums be set (in compliance with the SDP), but
that they be comment tagged as either "necessary" (you are aware of an incompatibility with the
previous version) or "arbitrary" (you set this version artitrarily and it may be lowered if
an issue with cross-compatibility arises).


## Conda lock file

### Using the Lock File Locally

run the following
```bash
CONDA_ENV_NAME=<env-name> make conda-env
```

The CONDA_ENV_NAME environment variable is optional and defaults to `checkmaite`.

### Updating the lockfile

To update the lockfile (e.g. after changing pyproject.yaml), run the following:

```bash
make conda-lock
```

## Running the GitLab CI locally

We use [gitlab-ci-local](https://github.com/firecow/gitlab-ci-local) to run the GitLab CI on our local machines. It is an open-source CLI that lets us run GitLab CI jobs locally using the same `.gitlab-ci.yml` that we use in GitLab. This is handy for debugging CI without pushing commits. Please see the `gitlab-ci-local` repository README for how to install `gitlab-ci-local`.

`gitlab-ci-local` can execute jobs via a Docker container or directly on your machine (shell executor). We follow the Docker approach and so you will need to have Docker available locally before you can use the tool.

To run GitLab CI locally, we have created a special file `.gitlab-ci-local-variables.yml`. You should generate a PAT on GitLab (contact the checkmaite team if you do not know how to do this) and then update the `GITLAB_CI_TOKEN:` field in `.gitlab-ci-local-variables.yml` with this value.

To run a single job, it's as simple as running `gitlab-ci-local run lint` (you can replace `lint` with any job name from `.gitlab-ci.yml` such as `test` or `pages-branch`). A Docker container should then start and you should be able to replicate the GitLab CI experience locally.

### How it works

The way that `gitlab-ci-local` works is that it parses your `.gitlab-ci.yml` alongside `.gitlab-ci-local-variables.yml`, builds the job graph, and then runs the job(s) you ask for. You can list the jobs it would run with `--list`. If a job specifies `image:` (this is what we recommend), it spins up that Docker image and runs your `before_script`/`script`/`after_script` inside it.
