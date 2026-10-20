# Benchmark methodology

How every number in `results/<experiment>/results.json` is produced.

## Data

- **Test set:** the dataset's official test split with its human labels. It is the only
  ground truth and is never used to train or select anything.
- **Unlabeled pool:** the training split (and the validation split, carved out of train
  when the dataset has none) with labels hidden. The teacher labels it; students learn
  only from those labels.
- **Audit:** the hidden training labels are used once, after labeling, to measure
  teacher label quality (`audit.md`).

## Models in the table

| Row | What it is |
|---|---|
| teacher · single call | Sample 0 of the teacher on each test message: what you would deploy. |
| teacher · majority vote of k | Self-consistency vote (only with `self_consistency > 1`): what the students learned from. |
| student · served | Mean ± std over seeds of the full-budget runs; *served accuracy* is the exported artifact of the seed with the best validation agreement with the teacher. |

## Metrics

| Metric | Definition |
|---|---|
| Accuracy | share of test messages whose predicted label equals the human label |
| Macro-F1 | unweighted mean of per-intent F1 over the gold label set; invalid teacher answers count as misses |
| ECE | expected calibration error with 15 equal-width confidence bins; student confidence is the top-class probability, teacher confidence is the vote share (so it is undefined for a single call) |
| Cost per 1,000 predictions | teacher: mean logged dollars per prediction (input, output, cache-read and cache-write tokens × published price) × 1,000; student: CPU instance $/hour ÷ (measured requests/s × 3,600) × 1,000 |
| Latency p50 / p95 | teacher: logged API round trip per call (the slowest of the k parallel calls for the vote); student: one request with a batch of one, after `warmup` untimed calls, on the stated CPU and thread count |
| Throughput | batched requests per second (batch size `throughput_batch_size`), at least `throughput_seconds` of work |
| Size | files the runtime loads: ONNX model + `tokenizer.json`, or the scikit-learn artifact |

## Learning curve

Students are trained on the first N teacher-labeled examples of a seeded permutation of
the pool (N in `budgets`). The secondary x-axis converts N into labeling dollars using the
teacher's measured cost per labeled example.

## Fallback routing

For thresholds in `benchmark.routing.thresholds`: the share of test messages whose
student confidence is below the threshold, the accuracy if those are answered by the
teacher (single call) instead, and the blended cost: student cost + share × teacher cost.

## Reproducibility

- Seeds fix data sampling, initialisation and shuffling; dependencies are pinned in
  `uv.lock`.
- Teacher answers are cached by prompt hash, and can be shared with
  `heartcut labels push/pull`, so `make benchmark` reruns everything with no API spend.
- `results.json` records CPU, core count, thread count, platform, Python and library
  versions, git revision, the date of the teacher price table and the CPU instance price.
