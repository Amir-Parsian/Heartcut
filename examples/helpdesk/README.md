# End-to-end helpdesk distillation

Train a small classifier to route IT tickets into six queues, then use the saved
model on new tickets. This example includes 120 synthetic tickets, a custom loader,
a runnable demo and a config you can adapt to your own data. Run commands from the
repository root with Python 3.11+ and [uv](https://docs.astral.sh/uv/).

## Run the complete CPU demo

```bash
uv run --extra pipeline python examples/helpdesk/demo.py
```

The first run installs dependencies. After that, the default demo needs no network,
API key, external dataset or pretrained model download. It performs all six stages:

1. Split the tickets and hide training/validation labels.
2. Collect three teacher answers per ticket and cache them locally.
3. Audit the teacher against the hidden labels.
4. Train and save two TF-IDF classifiers, using both 36-example and full training
   budgets. `tfidf-hard` learns the teacher's majority label; `tfidf-soft` learns
   the distribution of its votes through weighted examples.
5. Benchmark on held-out labels, select the winner using teacher-labeled validation
   data, and export the students. TF-IDF uses scikit-learn artifacts; neural
   students use ONNX with int8 quantization.
6. Reload the exported winner and predict queues and confidence for new tickets.

The default **teacher is simulated**: it looks at hidden labels and adds noise.
Student training, export, timing and inference really run, but the teacher's accuracy
and pricing are illustrative. The small synthetic dataset demonstrates the process;
use real tickets and a real teacher to measure useful distillation quality.

Try your own tickets or replay from the local cache:

```bash
uv run --extra pipeline python examples/helpdesk/demo.py \
  --text "I need a password reset" --text "The VPN stopped working"
uv run --extra pipeline python examples/helpdesk/demo.py --offline
```

`--offline` requires an initial completed labeling run or matching cached answers.
The script stores the teacher cache inside the experiment's artifact directory.

## Inspect what was created

| Location | Contents |
|---|---|
| `artifacts/helpdesk-demo/data/` | Prepared splits and hidden training labels |
| `artifacts/helpdesk-demo/labels/` | Teacher labels, vote distributions, call/cost logs and audit |
| `artifacts/helpdesk-demo/students/` | Trained models, training metadata and predictions for each budget |
| `artifacts/helpdesk-demo/export/` | Exported models and `export.json` metadata |
| `artifacts/demo-results/helpdesk-demo/` | `results.md`, `results.json`, charts, error analysis, model card and audit |
| `artifacts/demo-results/helpdesk-demo/demo.json` | Selected export path and example predictions |

The runner prints the exact export and report paths. Repeating the same experiment
reuses completed labeling and training stages. Generated files are ignored by Git.

## Serve the model over HTTP

Use the serving command printed by the demo, or read its selected export path:

```bash
MODEL_DIR=$(uv run --extra pipeline python -c \
  "import json; print(json.load(open('artifacts/demo-results/helpdesk-demo/demo.json'))['export_dir'])")
uv run --extra pipeline --extra serve heartcut serve --model "$MODEL_DIR"
```

In another terminal:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["Please reset my password", "My VPN keeps disconnecting"]}'
```

Each prediction contains a queue label, confidence and measured latency. Keep the
`pipeline` extra installed for the TF-IDF model's scikit-learn runtime.

## Replace the simulation with a real teacher

Choose a teacher config from `configs/teachers/` and review its model, endpoint and
pricing. For example, using the repository's hosted teacher config:

```bash
export ANTHROPIC_API_KEY=...  # supply your own key through the environment

# Prepare the data and estimate cost without making teacher calls.
uv run --extra pipeline --extra teachers python examples/helpdesk/demo.py \
  --config examples/helpdesk/real.yaml \
  --set teacher.self_consistency=3 \
  --set teacher.max_budget_usd=2 --dry-run

# Run the same pipeline with the real teacher; this makes billable calls.
uv run --extra pipeline --extra teachers python examples/helpdesk/demo.py \
  --config examples/helpdesk/real.yaml \
  --set teacher.self_consistency=3 \
  --set teacher.max_budget_usd=2
```

Set the budget to cover the estimate. If the cap stops labeling, completed calls stay
cached: raise the cap and rerun to continue. Three samples enable soft targets; a
single sample falls back to hard labels. These distributions come from repeated
answers, rather than internal teacher logits.

For an existing local Ollama server:

```bash
uv run --extra pipeline --extra teachers python examples/helpdesk/demo.py \
  --set name=helpdesk-local \
  --set teacher=configs/teachers/qwen2.5-7b-ollama.yaml
```

That config needs no API key and defaults to one
sample (hard-label training); set `self_consistency: 3` in that config to collect
vote distributions. Its monetary cost stays unknown until you supply defensible pricing.

## Distill into a neural student

Add the encoder student to train DistilBERT using the temperature-scaled KL loss
plus hard-label cross-entropy. This requires PyTorch and a pretrained model download;
training can use the available GPU, and the exported model serves on CPU.

```bash
uv run --extra pipeline --extra teachers --extra students --extra export \
  python examples/helpdesk/demo.py \
  --config examples/helpdesk/real.yaml \
  --set name=helpdesk-neural \
  --set teacher.self_consistency=3 \
  --set teacher.max_budget_usd=2 \
  --set 'students=[configs/students/tfidf-logreg.yaml,configs/students/encoder-distilbert-soft.yaml]' \
  --set 'budgets=[full]'
```

Append `--dry-run` to estimate teacher cost first. To practice neural training with
the simulated teacher, use `--config examples/helpdesk/demo.yaml` and omit the two
`teacher...` overrides. Check both exported students under
`artifacts/helpdesk-neural/export/`: the baseline may win selection,
while the encoder still produces `model.onnx`, `model.int8.onnx` and its tokenizer.

This trains classifiers for the six helpdesk queues. It does not produce a general
chat model. For your own task, replace `tickets.jsonl`, adapt `dataset.yaml` and the
loader, and follow [Adding a task](../../docs/adding-a-task.md). The existing
`experiment.yaml` also provides a larger TF-IDF + MiniLM/SetFit configuration.
