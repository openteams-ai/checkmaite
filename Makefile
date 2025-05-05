SHELL := /bin/bash
# help/list functionality from https://stackoverflow.com/a/59087509
.DEFAULT_GOAL := help


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
