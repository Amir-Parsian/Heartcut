---
title: Heartcut
emoji: 🧪
colorFrom: blue
colorTo: gray
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# Heartcut demo

The distilled student and the teacher LLM side by side: label, confidence, latency and
cost per call for the message you type.

## Run locally

```bash
uv sync --extra serve --extra teachers --extra demo
export HEARTCUT_MODEL_DIR=artifacts/banking77/export/<student>-<fingerprint>
export HEARTCUT_RESULTS=results/banking77/results.json
export HEARTCUT_TEACHER_CONFIG=configs/teachers/claude-haiku-5-5.yaml  # optional
export ANTHROPIC_API_KEY=...                                            # optional
uv run python demo/app.py
```

## Deploy to Hugging Face Spaces

1. Upload the exported model directory to a model repo:
   `huggingface-cli upload <user>/heartcut-banking77-student artifacts/banking77/export/<dir> .`
2. Create a Gradio Space and copy `demo/app.py`, `demo/requirements.txt` and this
   README into it, plus `results/banking77/results.json` and the teacher config and
   prompt if you want the teacher column.
3. Set Space variables: `HEARTCUT_MODEL_REPO`, `HEARTCUT_RESULTS=results.json`, and
   optionally `HEARTCUT_TEACHER_CONFIG` with the API key as a **secret**. Without a key
   the teacher column is disabled, so the public demo never spends money unless you
   choose to.
