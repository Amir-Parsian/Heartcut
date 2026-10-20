# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-19

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
- CI: ruff, strict mypy, tests with an 80% coverage gate, Docker build and `/predict`
  smoke test.

[Unreleased]: https://github.com/Amir-Parsian/DistillKit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Amir-Parsian/DistillKit/releases/tag/v0.1.0
