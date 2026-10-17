.PHONY: setup data test lint eval leaderboard compare
SPLIT ?= dev
REPEAT ?= 0

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
	uv run kosha eval --config $(RUN) --split $(SPLIT) --repeat $(REPEAT)

compare:
	uv run kosha compare $(A) $(B)

leaderboard:
	uv run kosha leaderboard
