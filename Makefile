SHELL := /bin/bash
# help/list functionality from https://stackoverflow.com/a/59087509
.DEFAULT_GOAL := help

# Conda backend selection - defaults to conda, set CONDA_LOCK_ARGS to "--micromamba" to use micromamba
CONDA_LOCK_ARGS ?=
CONDA_ENV_NAME ?= jatic-ri

.PHONY: init
#: initialize project
init:
	poetry install --extras dev
	poetry run pre-commit install
	poetry env info
	$(HIDE)echo "Created virtual environment"

.PHONY: clean
#: Clean up the venv
clean:
	rm -rf .venv

.PHONY: reset
reset: clean init

.PHONY: format
#: run all formatting
format:
	poetry run pre-commit run --all-files --verbose
	poetry run pyright src/

.PHONY: test
#: Run tests with current python
test:
	poetry run pytest tests -vvv --cov=jatic_ri --cov-report term --cov-report xml:coverage_report.xml --cov-fail-under=90

.PHONY: check-conda-lock
#: Check if conda lockfile is consistent with pyproject.toml
check-conda-lock:
	poetry run python scripts/check_conda_lock.py

.PHONY: conda-env
#: Create (or update if exists) conda environment from lock file
conda-env:
# 	If the conda environment already exists, it will be updated
	conda-lock install -n $(CONDA_ENV_NAME) $(CONDA_LOCK_ARGS) conda-lock.yml

.PHONY: conda-lock
#: Generate conda-lock.yml file
conda-lock: 
#   -f pyproject.toml          : generate lockfile from pyproject.toml
#   --filter-extras            : don't include optional dependencies
#   --extras dev               : but do include dev dependencies
#   -p linux-64                : only solve for the linux platform
#   --lockfile conda-lock.yml  : name/path of output lockfile
	conda-lock $(CONDA_LOCK_ARGS) \
		-f pyproject.toml \
		--filter-extras \
		--extras dev \
		-p linux-64 \
		--lockfile conda-lock.yml
