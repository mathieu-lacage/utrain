.PHONY: all containers check cram FORCE

all: containers

containers: containers/shakespeare-char containers/nanochat

containers/%: FORCE
	podman build -t utrain-$*:utrain -f containers/$*/Containerfile .

cram:
	uv run pytest tests/test_cram.py

test-matrix:
	uv run tox

check:
	uv run pre-commit run --all-files

cov:
	@rm -rf .coverage .coverage.*
	PYTHONUNBUFFERED=1 COVERAGE_PROCESS_START=$(shell pwd)/pyproject.toml COVERAGE_FILE=$(shell pwd)/.coverage uv run pytest

cov-report:
	COVERAGE_FILE=$(shell pwd)/.coverage uv run coverage combine --rcfile $(shell pwd)/pyproject.toml -a -q || true
	COVERAGE_FILE=$(shell pwd)/.coverage uv run coverage html -d cov --rcfile $(shell pwd)/pyproject.toml
	COVERAGE_FILE=$(shell pwd)/.coverage uv run coverage report
