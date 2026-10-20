"""Gradio demo: the distilled student and the teacher LLM, side by side.

Runs locally or as a Hugging Face Space. Configuration (environment variables / Space
secrets):

    HEARTCUT_MODEL_DIR       exported model directory, or
    HEARTCUT_MODEL_REPO      a Hugging Face model repo holding the exported directory
    HEARTCUT_RESULTS         results.json of the benchmark (for the student's cost/call)
    HEARTCUT_TEACHER_CONFIG  teacher YAML; the teacher column is shown when its API key
                               (e.g. ANTHROPIC_API_KEY) is set

Every number shown is measured for the request just made (latency, teacher tokens) or
read from the benchmark results file (student cost per call); nothing is hard-coded.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from heartcut.config import load_teacher
from heartcut.serve.router import TeacherClassifier
from heartcut.serve.runtime import Predictor, load_predictor, predict

EXAMPLES = [
    "I still haven't received my new card, it's been two weeks.",
    "Why was I charged a fee for withdrawing cash?",
    "My top-up didn't go through but the money left my account.",
    "How do I change my PIN?",
    "Someone stole my wallet with my card in it.",
]


def load_student() -> Predictor:
    model_dir = os.environ.get("HEARTCUT_MODEL_DIR")
    repo = os.environ.get("HEARTCUT_MODEL_REPO")
    if not model_dir and repo:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(repo_id=repo)
    if not model_dir:
        raise RuntimeError("set HEARTCUT_MODEL_DIR or HEARTCUT_MODEL_REPO")
    threads = os.environ.get("HEARTCUT_THREADS")
    return load_predictor(Path(model_dir), threads=int(threads) if threads else None)


def load_teacher_classifier(predictor: Predictor) -> TeacherClassifier | None:
    path = os.environ.get("HEARTCUT_TEACHER_CONFIG")
    if not path:
        return None
    config = load_teacher(path)
    if config.api_key_env and not os.environ.get(config.api_key_env):
        return None
    return TeacherClassifier(
        config,
        predictor.labels,
        task=str(predictor.meta.get("task", "intent classification")),
        oos_label=predictor.meta.get("oos_label"),
    )


def student_cost_per_1k(predictor: Predictor) -> float | None:
    """The served student's benchmarked cost per 1,000 predictions, if available."""
    path = os.environ.get("HEARTCUT_RESULTS")
    if not path or not Path(path).exists():
        return None
    results: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    for model in results.get("models", []):
        if model.get("name") == predictor.name:
            cost = model.get("cost_per_1k_usd")
            return float(cost) if cost is not None else None
    return None


def _usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.6f}" if value < 0.01 else f"${value:.4f}"


def compare(
    text: str,
    predictor: Predictor,
    teacher: TeacherClassifier | None,
    cost_per_1k: float | None,
) -> tuple[str, str]:
    """Markdown for the student and teacher columns."""
    text = text.strip()
    if not text:
        return "Type a message first.", ""
    (prediction,), elapsed_ms = predict(predictor, [text])
    student = (
        f"### Student: `{prediction.label}`\n"
        f"- confidence: {prediction.confidence:.1%}\n"
        f"- latency: {elapsed_ms:.1f} ms (this request, CPU)\n"
        f"- cost per call: {_usd(None if cost_per_1k is None else cost_per_1k / 1000)}"
        f"{'' if cost_per_1k is not None else ' (no benchmark results loaded)'}\n"
        f"- model: {predictor.name} ({predictor.meta.get('format')})"
    )
    if teacher is None:
        return student, (
            "### Teacher: disabled\nSet `HEARTCUT_TEACHER_CONFIG` and the matching API key "
            "to compare with the LLM."
        )
    answer = asyncio.run(teacher.classify(text))
    label = f"`{answer.label}`" if answer.label else f"no valid label ({answer.error})"
    agree = "agrees" if answer.label == prediction.label else "disagrees"
    teacher_md = (
        f"### Teacher: {label}\n"
        f"- latency: {answer.latency_ms:.0f} ms (API round trip)\n"
        f"- cost per call: {_usd(answer.cost_usd)} (this request's tokens x list price)\n"
        f"- model: {teacher.config.model}\n\n"
        f"The student **{agree}** with the teacher."
    )
    return student, teacher_md


def build_demo(
    predictor: Predictor, teacher: TeacherClassifier | None, cost_per_1k: float | None
) -> Any:
    import gradio as gr

    with gr.Blocks(title="Heartcut: LLM vs. distilled student") as demo:
        gr.Markdown(
            "# Heartcut\nA tiny CPU model distilled from an LLM classifier. Type a "
            "customer message and compare the two."
        )
        message = gr.Textbox(label="Customer message", lines=2, placeholder=EXAMPLES[0])
        button = gr.Button("Classify", variant="primary")
        with gr.Row():
            student_out = gr.Markdown()
            teacher_out = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=message)

        def run(text: str) -> tuple[str, str]:
            return compare(text, predictor, teacher, cost_per_1k)

        button.click(run, inputs=message, outputs=[student_out, teacher_out])
        message.submit(run, inputs=message, outputs=[student_out, teacher_out])
    return demo


def main() -> None:
    predictor = load_student()
    build_demo(
        predictor, load_teacher_classifier(predictor), student_cost_per_1k(predictor)
    ).launch()


if __name__ == "__main__":
    main()
