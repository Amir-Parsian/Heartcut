# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Security
- After the eMudhra/emSign CA key compromise: certifi 2026.10.19 in the lockfile, and
  the Docker image installs dependencies with uv 0.13.4, copied from uv's official image
  instead of installed with the base image's pip.

### Added
- The Release workflow also creates the GitHub release, with the version's changelog
  section as its notes. README badge for the PyPI version.

### Changed
- The Hugging Face Spaces demo installs `heartcut` from PyPI instead of from GitHub, and
  its Space config pins Python 3.11, which Heartcut requires.

## [0.1.0] - 2026-10-20

### Added
- Typed YAML configs (Pydantic) with file references and `--set` overrides;
  fingerprinted, manifest-carrying stage outputs.
- Data stage: Hugging Face, CSV, JSONL and CLINC150 loaders, custom loaders by module or
  file path, fallback sources, deterministic stratified splits, hidden train labels.
  Configs for Banking77 and CLINC150.
- Teacher labeling: Anthropic and OpenAI-compatible (OpenAI, vLLM, Ollama) backends, a
  simulated teacher for tests, prompt templates with JSON-schema enums, answer
  validation, SQLite cache, async concurrency, rate limiting, retries with backoff,
  self-consistency voting, per-call token/cost logs, budget cap, offline mode and dry-run
  cost estimates.
- Label-quality audit against the hidden human labels.
- Students: TF-IDF + logistic regression, SetFit-style sentence-embedding model and a
  fine-tuned encoder; hard and soft (distillation) labels; labeling-budget experiment;
  MLflow tracking.
- Benchmark: accuracy, macro-F1, ECE, cost per 1,000 predictions, latency, throughput
  and size; Pareto chart, learning curve, error analysis, routing analysis, model card;
  README section generated from results.
- ONNX export with dynamic int8 quantisation and parity checks; torch-free CPU runtime;
  FastAPI service with an optional fallback router; Docker image; Gradio demo.
- Publishing and fetching teacher labels on the Hugging Face Hub.
- Packaging for PyPI: the wheel ships the price table, so teacher costs are reported
  outside a clone (a `configs/pricing.yaml` in the working directory takes precedence),
  and the PyPI description is the README with its links and charts pointing at the
  release's tag on GitHub.
- CI: ruff, strict mypy, tests with an 80% coverage gate, Docker build and `/predict`
  smoke test, and a check of the built wheel installed outside the repository.
- Release workflow: pushing a `v*` tag that matches the package version publishes to PyPI
  through trusted publishing.

[Unreleased]: https://github.com/Amir-Parsian/Heartcut/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Amir-Parsian/Heartcut/releases/tag/v0.1.0
