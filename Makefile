# Common tasks. Everything goes through uv so the locked environment is used.
EXPERIMENT ?= configs/experiments/banking77.yaml
RUN = uv run

.PHONY: help install lint format typecheck test test-fast data label label-dry-run audit train \
        labels-pull benchmark report export serve docker demo smoke clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install every extra and the dev tools
	uv sync --all-extras
	$(RUN) pre-commit install

lint:  ## Ruff lint + format check
	$(RUN) ruff check src tests scripts demo examples
	$(RUN) ruff format --check src tests scripts demo examples

format:  ## Auto-format
	$(RUN) ruff format src tests scripts demo examples
	$(RUN) ruff check --fix src tests scripts demo examples

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

labels-pull:  ## Fetch published teacher labels (LABELS_REPO=<user/dataset>)
	$(RUN) distillkit labels pull -c $(EXPERIMENT) --repo-id $(LABELS_REPO)

benchmark:  ## Regenerate every number and chart from cached teacher labels (no API spend)
	$(RUN) distillkit run -c $(EXPERIMENT) --offline

report:  ## Write the README results section from results.json
	$(RUN) distillkit report -c $(EXPERIMENT)

export:  ## Export the best student to ONNX int8
	$(RUN) distillkit export -c $(EXPERIMENT)

serve:  ## Serve the exported student locally (MODEL=<export dir>)
	$(RUN) distillkit serve --model $(MODEL)

docker:  ## Build the CPU serving image (MODEL=<export dir>)
	docker build --build-arg MODEL_DIR=$(MODEL) -t distillkit-serve .

demo:  ## Gradio demo (MODEL=<export dir>)
	DISTILLKIT_MODEL_DIR=$(MODEL) $(RUN) python demo/app.py

smoke:  ## Whole pipeline on a small Banking77 slice with the simulated teacher (free)
	$(RUN) distillkit run -c $(EXPERIMENT) \
		--set teacher=configs/teachers/simulated.yaml \
		--set dataset.max_examples.train=1000 --set dataset.max_examples.test=500 \
		--set students=[configs/students/tfidf-logreg.yaml] --set seeds=[0] \
		--set budgets=[500,full] --set tracking.backend=none \
		--set artifacts_dir=artifacts/smoke --set results_dir=artifacts/smoke/results

clean:  ## Remove caches and generated artifacts (keeps the teacher cache)
	rm -rf artifacts mlruns build .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
