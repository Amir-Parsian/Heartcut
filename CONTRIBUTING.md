# Contributing

Thanks for helping. Heartcut aims to read like production code, so changes come with
tests and stay small enough to review.

## Setup

```bash
uv sync --all-extras
uv run pre-commit install
make lint typecheck test
```

`make test-fast` skips the tests that train (tiny) neural models.

## Ground rules

- **Never call a paid API from tests or CI.** Use the simulated teacher or a mocked
  HTTP transport (see `tests/test_backends.py` and `tests/test_integration.py`).
- **Never tune on the test set.** Anything that chooses (hyperparameters, seeds, the
  exported student, thresholds) must use teacher-labeled validation data.
- **Never hand-write benchmark numbers.** README results come from
  `heartcut report`; results produced with the simulated teacher are not published.
- **No secrets in the repo.** API keys are read from environment variables only;
  `.env` is git-ignored and `.env.example` documents the names.
- Keep `mypy --strict` and `ruff` clean, keep coverage of `src/` at 80% or more, and add
  docstrings to public functions.

## Pull requests

One topic per pull request, with a description of what changed and why, and how it was
tested. Update `CHANGELOG.md` under *Unreleased* for user-visible changes. If a change
affects results, say which numbers move and attach the regenerated `results.md`.

## Releasing

1. Set the new version in `pyproject.toml` and `src/heartcut/__init__.py`, and move the
   *Unreleased* entries in `CHANGELOG.md` under it.
2. Merge to `main` and wait for CI to pass.
3. Tag that commit and push the tag:
   `git tag v0.2.0 && git push origin v0.2.0`.

The `Release` workflow checks that the tag matches both version strings, builds the
wheel and sdist, and publishes them to PyPI through trusted publishing.

## Adding a dataset, teacher or student

- Dataset: a config in `configs/datasets/` and, if needed, a loader; see
  [docs/adding-a-task.md](docs/adding-a-task.md).
- Teacher: a config in `configs/teachers/`; a new API needs a `TeacherBackend` that only
  does transport (the shared runner handles caching, retries and cost) and a price entry
  in `configs/pricing.yaml` with its date.
- Student: a `Student` subclass registered in `students/registry.py`, a config model in
  `config.py`, and an export path in `serve/export.py` if it should be servable.
