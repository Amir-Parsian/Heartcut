# Adding your own task

DistillKit is task-agnostic: a new classification task needs a dataset config and, if
none of the built-in loaders fits, a loader function. Nothing in `src/` changes. The
worked example is [`examples/helpdesk`](../examples/helpdesk): IT tickets with a
`subject`, a `body` and a `queue`.

## 1. Can a built-in loader read your data?

| Loader | Use it for | Key arguments |
|---|---|---|
| `huggingface` | a dataset on the Hugging Face Hub | `path`, `name`, `text_column`, `label_column`, `splits` |
| `csv` | one CSV file (path or URL) per split | `files`, `text_column`, `label_column` |
| `jsonl` | one JSON-lines file per split | `files`, `text_column`, `label_column` |
| `clinc_json` | the original CLINC150 release | `file`, `include_oos` |

If it can, the config is all you need:

```yaml
# configs/datasets/tickets.yaml
name: tickets
task: routing customer-support tickets to the right team
sources:
  - loader: csv
    args:
      files: {train: data/tickets_train.csv, test: data/tickets_test.csv}
      text_column: message
      label_column: team
```

## 2. Otherwise, write a loader

A loader is any function that returns `{"train": DataFrame, "test": DataFrame}` (and
optionally `"validation"`) with a `text` and a `label` column. Labels are names, not
integer ids.

```python
# examples/helpdesk/loader.py
import pandas as pd
from distillkit.data.splits import stratified_split

def load_tickets(path: str, test_fraction: float = 0.25, seed: int = 0) -> dict[str, pd.DataFrame]:
    raw = pd.read_json(path, lines=True)
    frame = pd.DataFrame({"text": raw["subject"] + ". " + raw["body"], "label": raw["queue"]})
    train, test = stratified_split(frame, test_fraction, seed)
    return {"train": train, "test": test}
```

Point the config at it with a file path or a module path (both relative to where you
run `distillkit`, usually the repository root); `args` are passed as keyword arguments:

```yaml
sources:
  - loader: examples/helpdesk/loader.py:load_tickets   # or my_package.loaders:load_tickets
    args: {path: examples/helpdesk/tickets.jsonl}
```

DistillKit takes it from there: it carves a stratified validation pool out of train if
you did not return one, assigns ids, hides the train/validation labels from training, and
records the source and counts in the stage manifest. If your test split has no human
labels you can still label and train, but the benchmark needs them: get a few hundred
test messages labeled by people first.

`sources` is a list: DistillKit tries each source in order, which is how the Banking77
config falls back from the Hugging Face Hub to the upstream CSV files.

## 3. Write the task description

`task` is inserted into the teacher prompt ("You are an expert annotator creating
training data for $task"), so make it specific. If your label set has a catch-all class,
set `oos_label` and the prompt gains a rule for it. Label names are shown to the teacher
as they are, so `card_arrival` beats `C17`. To change the prompt itself, copy
`prompts/intent_classification.yaml` and point the teacher config's `prompt` at the copy.

## 4. Create an experiment and run it

```yaml
# examples/helpdesk/experiment.yaml
name: helpdesk
dataset: dataset.yaml
teacher: ../../configs/teachers/claude-haiku-5-5.yaml
students:
  - ../../configs/students/tfidf-logreg.yaml
  - ../../configs/students/setfit-minilm.yaml
budgets: [full]
```

```bash
uv run distillkit data  -c examples/helpdesk/experiment.yaml     # check counts and labels
uv run distillkit label -c examples/helpdesk/experiment.yaml --dry-run
uv run distillkit run   -c examples/helpdesk/experiment.yaml --set teacher=configs/teachers/simulated.yaml
uv run distillkit run   -c examples/helpdesk/experiment.yaml     # with the real teacher
```

Start with the simulated teacher to check the plumbing for free, then a small slice
(`--set dataset.max_examples.train=200`) with the real teacher before labeling
everything.
