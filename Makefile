.PHONY: setup check lint fmt test

setup:
	python3.11 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

check: lint test

lint:
	.venv/bin/ruff check src tests scripts
	.venv/bin/ruff format --check src tests scripts

fmt:
	.venv/bin/ruff format src tests scripts
	.venv/bin/ruff check --fix src tests scripts

test:
	.venv/bin/pytest -q
