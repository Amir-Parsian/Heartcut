# Draft: Replacing an LLM classifier with a model that runs on a laptop CPU

*Draft. Every bracketed value comes from `results/banking77/results.json` and must be
filled in from that file, not estimated. Remove this note before publishing.*

## The problem

Calling an LLM API is the fastest way to ship a classifier: write a prompt, list the
labels, done. For simple, high-volume tasks like routing support messages, the bill and
the latency grow with traffic, and the model is far bigger than the task needs.

The alternative is old: knowledge distillation. Let the big model label data once, then
train a small model on those labels. What has changed is how cheap the labeling step is.
I built DistillKit to measure the trade-off honestly, end to end.

## Setup

- **Task:** Banking77, 77 intents of online-banking customer messages.
- **Teacher:** [teacher model], prompted zero-shot with the label list and a JSON-schema
  constraint. Labeling the [N] unlabeled messages cost [$X] ([cost per example] each).
- **Rule:** the students never see a human label. Human labels are only used to score
  everything on the official test set, and once to audit the teacher.
- **Students:** TF-IDF + logistic regression, a SetFit-style MiniLM model, and a
  fine-tuned DistilBERT, each trained with 3 seeds.

## How good were the teacher's labels?

[Teacher label accuracy vs. hidden human labels], [worst intents], [what the confidence
filter did]. Distillation cannot beat its labels, so this number bounds everything else.

## Results

[Headline sentence from results.json]

[Results table and Pareto chart]

[What the gap is made of: the error analysis's top confused pairs, with an example or
two, and whether they are also teacher mistakes.]

## How much labeling do you need?

[Learning curve: accuracy at 500 / 2,000 / 5,000 / all examples, with the labeling cost
of each.] [Where the curve flattens.]

## Not all or nothing: fallback routing

[Routing table: at threshold X, Y% of traffic goes to the teacher, accuracy becomes Z,
blended cost per 1,000 is W.]

## What I would do differently

[Honest limitations: one dataset family, CPU instance price assumption, teacher latency
measured over the network, ...]

## Try it

`git clone`, `make labels-pull`, `make benchmark`: every number in this post is
regenerated from cached labels, without an API key. Live demo: [link].
