# Common tasks. Everything goes through uv so the locked environment is used.
EXPERIMENT ?= configs/experiments/banking77.yaml
RUN = uv run

.PHONY: help install lint format typecheck test test-fast coverage data label label-dry-run \
        audit train benchmark export serve docker demo smoke clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install every extra and the dev tools
	uv sync --all-extras
	$(RUN) pre-commit install

lint:  ## Ruff lint + format check
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

format:  ## Auto-format
	$(RUN) ruff format src tests
	$(RUN) ruff check --fix src tests

typecheck:  ## mypy (strict)
	$(RUN) mypy

test:  ## Full test suite with the coverage gate
	$(RUN) pytest --cov

test-fast:  ## Skip tests that train neural students
	$(RUN) pytest -m "not slow"

data:  ## Load the dataset into the common schema
	$(RUN) distillkit data -c $(EXPERIMENT)

label-dry-run:  ## Estimate the teacher labeling cost
	$(RUN) distillkit label -c $(EXPERIMENT) --dry-run

label:  ## Label with the teacher (uses the cache; spends money on misses)
	$(RUN) distillkit label -c $(EXPERIMENT)

audit:  ## Teacher label quality vs. hidden human labels
	$(RUN) distillkit audit -c $(EXPERIMENT)

train:  ## Train every student x budget x seed
	$(RUN) distillkit train -c $(EXPERIMENT)
