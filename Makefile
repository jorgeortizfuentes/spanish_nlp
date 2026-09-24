SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c

.PHONY: install install_all lock test lint format build aider

install:
	@echo "Syncing environment with uv.lock (runtime + dev dependencies)..."
	uv sync

install_all:
	@echo "Syncing environment with all dependency groups (dev, docs, notebooks)..."
	uv sync --all-groups

lock:
	uv lock

aider:
	@echo "Setting up environment variables..."
	@echo "VERTEXAI_PROJECT=$(VERTEXAI_PROJECT)"
	@echo "VERTEXAI_LOCATION=$(VERTEXAI_LOCATION)"
	@echo "Executing 'aider'..."
	aider

test:
	@echo "Creating outputs directory..."
	@mkdir -p outputs
	@echo "Running tests with coverage..."
	uv run pytest --cov=spanish_nlp --cov-report=html:outputs/coverage --cov-report=term-missing -v tests/ | tee outputs/pytest-report.txt

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

build:
	uv build
