.PHONY: setup data test lint eval leaderboard
SPLIT ?= dev

setup:
	uv sync

data:
	uv run kosha data

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

eval:
	uv run kosha eval --config $(RUN) --split $(SPLIT)

leaderboard:
	uv run kosha leaderboard
