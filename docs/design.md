# Design notes

Why Heartcut is built the way it is. Each section states the decision, the reason,
and what it costs.

## 1. Stages with fingerprinted outputs

**Decision.** Each stage (data, label, train, export, benchmark) is a separate CLI
command that writes to `artifacts/<experiment>/<stage>/<name>-<fingerprint>/` with a
`manifest.json`. The fingerprint hashes the stage's config plus the fingerprints of its
inputs.

**Why.** Stages can be rerun independently without stale mix-ups: changing the prompt
changes the labels fingerprint, which changes every student fingerprint downstream, and
nothing has to be invalidated by hand. Old outputs stay on disk, so two prompt versions
can be compared. The manifest records library versions, the git revision and timestamps.

**Cost.** Directory names are less readable than `labels/latest`. A small DVC-like
mechanism lives in ~150 lines instead of a dependency; DVC itself is deferred (see
Open questions).

## 2. Human labels are hidden by construction

**Decision.** The data stage moves the train/validation labels into
`hidden/gold.parquet`. Only the audit (and the explicitly simulated teacher used in
tests) reads it. A test monkeypatches the reader to fail during training.

**Why.** The project's claim is "a student trained only on teacher labels". Enforcing
that in code is stronger than a convention. Validation data is teacher-labeled too, so
model selection uses what a real team would have, and the human-labeled test set is
never used for any decision (early stopping, seed choice, "best student" choice).

## 3. Teacher: backends do transport, the runner does everything else

**Decision.** A backend only sends one prompt and reports text, token usage and latency.
Caching, rate limiting, retries, self-consistency, budget enforcement and cost
accounting live in one shared runner.

**Why.** Every backend behaves the same, and each concern is tested once. The SDKs'
own retries are disabled (`max_retries=0`) so every attempt is visible in the call log.

- **Backends.** Anthropic Messages API via the official SDK; one OpenAI-compatible
  backend that covers OpenAI's API and local open-weights servers (vLLM, Ollama); and a
  simulated teacher for tests. Two hosted options plus a local one let the
  teacher choice stay a config change.
- **Structured output.** The label set is enforced with a JSON-schema `enum`
  (`output_config.format` on Anthropic, `response_format` on OpenAI-compatible servers;
  vLLM uses guided decoding for it). Every answer is still validated against the label
  set; anything else is counted as invalid, never guessed.
- **Prompt caching.** The label list sits in the system prompt, which is identical for
  every message, so providers can serve it from their prompt cache (marked with
  `cache_control` on Anthropic). Cache reads and writes are priced separately.
- **Cache.** SQLite, keyed by SHA-256 of backend, model, endpoint, full rendered prompt,
  schema, sampling settings and sample index. Entries keep the original usage and
  latency, so regenerated reports stay truthful and reruns cost nothing.
- **Budget cap.** `max_budget_usd` stops new calls once the cap is reached (in-flight
  calls finish, so it can overshoot by at most `concurrency` calls). Everything finished
  is cached and a rerun resumes.
- **Refusals** (`stop_reason: refusal`) are recorded as invalid answers. Server-side
  model fallbacks are deliberately *not* enabled: silently answering with a different
  model would corrupt the teacher's benchmark row.

## 4. Self-consistency and soft labels

**Decision.** With `self_consistency: k`, each message is sampled k times; the majority
vote is the label and the vote share is the confidence. The vote distribution is the
soft label.

**Why.** It gives every teacher (including APIs without usable log-probabilities) a
confidence score for filtering and a distribution for distillation. Current Claude 5.x
models reject non-default sampling parameters, so diversity comes from their default
sampling; local servers use `temperature: 0.7`.

**Distillation.** The encoder uses the Hinton loss `α·T²·KL(p_teacher^T ‖ p_student^T) +
(1-α)·CE(hard)`, with the vote distribution tempered as `softmax(log p / T)`. The linear
heads (TF-IDF, SetFit) use *soft-target expansion*: each example becomes one weighted
row per class with non-zero probability. For logistic regression the weighted log loss
equals the cross-entropy against the soft targets, so this is exact, not an
approximation.

## 5. Students

- **TF-IDF + logistic regression.** Word 1-2 grams for keywords, character 2-5 grams
  (within word boundaries) for typos and inflections.
- **SetFit-style.** A compact re-implementation of SetFit on sentence-transformers:
  contrastive fine-tuning on balanced same/different-intent pairs, then a logistic
  regression head. Owning ~150 lines keeps every step visible and avoids pinning to the
  `setfit` package's release cadence.
- **Encoder.** A plain PyTorch loop (AdamW without decay on biases/norms, linear warm-up,
  AMP on CUDA, gradient clipping, early stopping on validation agreement with the
  teacher, best-epoch restore). A trainer would hide the loss and the selection rule,
  which are exactly what this project needs to show.
- **Missing classes.** With small budgets some intents may have no examples; the linear
  heads place their columns into the full label space with probability 0, and the ONNX
  export gives unseen classes a large negative bias.

## 6. Budget experiment

Budgets take the first N examples of a seeded permutation of the unlabeled pool, so the
500-example run is a subset of the 2,000-example run (nested subsets make the learning
curve a fair "what if we had labeled less"). The seed also changes the permutation, so
the error bars include data-sampling variance. The validation pool is the same for all
budgets; the cost axis counts training examples only.

## 7. Export and serving

- **One ONNX graph per student** from token ids to logits; for SetFit the body, pooling,
  normalisation and head are fused. The TorchScript exporter is used because it is
  stable for BERT-family encoders (the dynamo exporter's output failed int8 shape
  inference in testing).
- **Parity check.** Every export is compared with the original model on validation texts
  (fp32 max |diff| ≤ 1e-3 or the export fails); int8 vs. fp32 agreement is recorded and
  the served model's test accuracy is reported separately, so quantisation loss is
  visible rather than assumed.
- **TF-IDF stays scikit-learn.** `skl2onnx` cannot convert the char n-gram vectoriser.
- **Torch-free runtime.** The serving image installs only the `serve` extra (ONNX
  Runtime, `tokenizers`, FastAPI): about 220 MB of Python packages instead of several GB.

## 8. Cost model

Teacher cost per 1,000 predictions = mean logged dollars per prediction × 1,000. Student
cost = reference instance price per hour ÷ (measured requests/s × 3,600) × 1,000. This
assumes a busy instance; at low traffic a serverless deployment bills differently. The
instance and its price date are recorded in every results file. Teacher latency is the
logged API round trip; student latency is local CPU time, which is the comparison a
caller experiences.

## 9. Honesty mechanisms

- README numbers are generated from `results.json` (`heartcut report`, with `--check`).
- Results from the simulated teacher are watermarked and refused by the README generator.
- Prices carry an `as_of` date; the CPU price ships marked "unverified" until checked.
- Costs are reported as unknown, never as zero, for models without a known price.

## Open questions and current defaults

| Question | Current default | Notes |
|---|---|---|
| Main hosted teacher | `claude-opus-5-5` | Strongest labels, but the dry run estimates $37 (no thinking, no caching) to $302 (every call at `max_output_tokens`) for Banking77, so it can hit a $50 cap. `claude-haiku-5-5` is about $1 for Banking77 and $2 for CLINC150. One `--set teacher=...` switches. |
| Open-weights teacher | Qwen2.5-7B-Instruct via vLLM or Ollama | Configs for both. Local calls need a defensible price (a hosted price for the same weights, or GPU-hours / tokens), otherwise cost is reported as unknown. |
| Experiment tracker | MLflow, local SQLite store | No account needed. W&B would need a second `Run` implementation in `tracking.py`. |
| DVC | Deferred | Fingerprinted stage directories cover reproducibility; teacher labels are shared via the Hugging Face Hub. |
| Redistributing teacher labels | Repos are created private | The dataset card flags the source license (Banking77 CC BY 4.0, CLINC150 CC BY 3.0); confirm before publishing. |
