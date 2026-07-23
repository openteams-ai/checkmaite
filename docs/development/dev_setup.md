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

We provide both uv-based environment and conda-based environment options:

### Option 1: Setup uv environment

To set up a uv environment, [install `uv`](https://docs.astral.sh/uv/getting-started/installation/).

#### For regular users

You can build the environment by running:

```bash
uv sync --no-dev
```

#### For package developers

If you plan to contribute to the project or need development tools, a plain sync
installs the `dev` dependency group by default:

```bash
uv sync
```

`dev` and `docs` are [PEP 735 dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups)
(not `pip` extras), so `pip install .[dev]` is not a supported path.

To contribute to the documentation, install with the docs dependencies:

```bash
uv sync --group docs
```

There are also optional dependency extras, `ui` and `reporting`:

```bash
uv sync --extra ui
uv sync --extra reporting   # PDF report export (markdown + xhtml2pdf)
```

JATIC tools which are not under active development are distributed through the separate
`checkmaite-plugins` package. It supports Python `<3.12` and is installed directly from GitLab instead of
through a `checkmaite` extra so that the PyPI package metadata remains valid:

```bash
uv pip install "checkmaite-plugins[unsupported] @ git+https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite-plugins.git@main"
```

If you want to install all the checkmaite extras, you can use:

```bash
uv sync --all-extras
```

And if you want to install everything and the kitchen sink:

```bash
uv sync --all-extras --all-groups
```

This project utilizes `pre-commit` for linting and formatting. **Developers** should also install the `pre-commit` hooks using:

```bash
uv run pre-commit install
```

### Option 2: Setup conda environment

To set up a conda environment, [install `conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) on your machine and build the environment by running:

```bash
# create env with conda-lock installed
conda create -n checkmaite "conda-lock>=3"
# activate the environment
conda activate checkmaite
# use conda-lock to install dependencies (include the `ui` extra so the full
# test suite can be collected)
conda-lock install -n checkmaite --extras ui conda-lock.yml
# finally, install the `checkmaite` package
pip install -e . --no-deps
```

The conda-lock environment intentionally excludes developer tooling such as
`pre-commit` (it is a PEP 735 `dev` dependency group, which conda-lock cannot
represent). Install `pre-commit` separately and then enable the hooks:

```bash
pip install pre-commit
pre-commit install
```

## Testing

(*In the following, instructions are only provided for `uv`. Similar instructions are valid for `conda`.*)

This project uses `pytest` for it's test suite. Run the full test suite with:

```bash
uv run pytest tests -svv
```

!!! note
    Ray >= 2.56 detects drivers launched via `uv run` and tries to propagate the
    uv environment to its workers, which breaks the Ray job-backend tests (worker
    startup times out). Disable the hook so workers use the already-synced
    environment: `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run pytest tests -svv`
    (CI sets this automatically).

This project also contains test markers to flag special test groups. The tests marked `real_data` are
time consuming tests that utilize real data. The `unsupported` capability tests are maintained in the
`checkmaite-plugins` repository.

You can run them manually with:

```bash
uv run pytest tests -svv -m real_data
```

## Linting and formatting

(*In the following, instructions are only provided for `uv`. Similar instructions are valid for `conda`.*)

Linting and formatting are automated via `pre-commit` hooks. However, if you'd like to run them directly, you can run:

```bash
uv run pre-commit run --all-files --verbose
```

Type checking is performed by `pyright`. This can be run using:

```bash
uv run pyright src
```

## Building the docs

(*In the following, instructions are only provided for `uv`. Similar instructions are valid for `conda`.*)

The documentation is built using [`mkdocs`](https://www.mkdocs.org/) and deployed via CI to GitLab Pages. checkmaite also makes use of the [`mkdocs-jupyter`](https://github.com/danielfrg/mkdocs-jupyter) plugin which allows the docs to be build from notebooks as well as the standard markdown.

The docs can be built locally in two different ways. To build the docs with a live-reloading server, use:

```bash
uv run mkdocs serve
```

This will create a live server running on the local machine. Any changes made to the document will be live-reloaded in the local website.

Alternately, the docs can be built locally as a static site similar to the process in CI by running:

```bash
uv run mkdocs build --site-dir public
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
conda-lock install -n <env-name> conda-lock.yml
```

### Updating the lockfile

To update the lockfile (e.g. after changing pyproject.yaml), run the following:

```bash
conda-lock -f pyproject.toml --extras dev -p linux-64 --lockfile conda-lock.yml
```

## Running the GitLab CI locally

We use [gitlab-ci-local](https://github.com/firecow/gitlab-ci-local) to run the GitLab CI on our local machines. It is an open-source CLI that lets us run GitLab CI jobs locally using the same `.gitlab-ci.yml` that we use in GitLab. This is handy for debugging CI without pushing commits. Please see the `gitlab-ci-local` repository README for how to install `gitlab-ci-local`.

`gitlab-ci-local` can execute jobs via a Docker container or directly on your machine (shell executor). We follow the Docker approach and so you will need to have Docker available locally before you can use the tool.

To run GitLab CI locally, we have created a special file `.gitlab-ci-local-variables.yml`. You should generate a PAT on GitLab (contact the checkmaite team if you do not know how to do this) and then update the `GITLAB_CI_TOKEN:` field in `.gitlab-ci-local-variables.yml` with this value.

To run a single job, it's as simple as running `gitlab-ci-local run lint` (you can replace `lint` with any job name from `.gitlab-ci.yml` such as `test` or `pages-branch`). A Docker container should then start and you should be able to replicate the GitLab CI experience locally.

### How it works

The way that `gitlab-ci-local` works is that it parses your `.gitlab-ci.yml` alongside `.gitlab-ci-local-variables.yml`, builds the job graph, and then runs the job(s) you ask for. You can list the jobs it would run with `--list`. If a job specifies `image:` (this is what we recommend), it spins up that Docker image and runs your `before_script`/`script`/`after_script` inside it.
