# Results

Benchmark outputs, committed so the README and the blog post can link to them. One
directory per experiment, written by `distillkit benchmark` (and `distillkit audit`):

| File | Contents |
|---|---|
| `results.json` | every number, plus hardware, library versions, git revision and price dates |
| `results.md` | the results table, labeling-budget table, routing table, environment |
| `pareto.png`, `pareto-dark.png` | accuracy vs. cost per 1,000 predictions |
| `learning_curve.png`, `learning_curve-dark.png` | accuracy vs. teacher-labeled examples and labeling cost |
| `error_analysis.md` | the most confused intent pairs for the best student, with examples |
| `audit.md`, `audit.json` | teacher label quality vs. the hidden human labels |
| `MODEL_CARD.md` | model card for the best student |

No results are committed yet: they need the first real teacher labeling run (see the
README's Status section). Runs with the simulated teacher are for testing the pipeline
and are never committed here.
