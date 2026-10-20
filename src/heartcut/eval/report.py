"""Markdown rendering of benchmark results, including the README results section.

The README section is generated from ``results.json`` and nothing else, so the README
can only show numbers the benchmark produced. ``heartcut report --check`` fails when
the README and the results file disagree, and simulated results are never published.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

START = "<!-- results:start -->"
END = "<!-- results:end -->"


class SimulatedResultsError(RuntimeError):
    """Raised when asked to publish results produced by the simulated teacher."""


def fmt_pct(stat: dict[str, Any] | float | None, digits: int = 1) -> str:
    if stat is None:
        return "n/a"
    if isinstance(stat, dict):
        mean, std = stat.get("mean"), stat.get("std")
        if mean is None:
            return "n/a"
        if std is not None:
            return f"{100 * mean:.{digits}f} ± {100 * std:.{digits}f}%"
        return f"{100 * mean:.{digits}f}%"
    return f"{100 * stat:.{digits}f}%"


def fmt_ece(stat: dict[str, Any] | float | None) -> str:
    if stat is None:
        return "n/a"
    if isinstance(stat, dict):
        mean, std = stat.get("mean"), stat.get("std")
        if mean is None:
            return "n/a"
        return f"{mean:.3f} ± {std:.3f}" if std is not None else f"{mean:.3f}"
    return f"{stat:.3f}"


def fmt_usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1:
        return f"${value:,.2f}"
    if value == 0:
        return "$0"
    # Two significant digits without scientific notation: $0.012, $0.0000061.
    decimals = 1 - math.floor(math.log10(abs(value)))
    return f"${value:.{decimals}f}"


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f} ms" if value >= 100 else f"{value:.2g} ms"


def fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}/s"


def fmt_mb(value: float | None) -> str:
    return "n/a" if value is None else (f"{value:,.0f} MB" if value >= 10 else f"{value:.1f} MB")


def results_table(results: dict[str, Any]) -> list[str]:
    lines = [
        "| Model | Accuracy | Macro-F1 | ECE | Served accuracy | Cost / 1k predictions "
        "| Latency p50 / p95 | Throughput | Size |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in results["models"]:
        served = m.get("served")
        latency = m.get("latency") or {}
        name = f"**{m['name']}**" if m["name"] == results.get("best_student") else m["name"]
        lines.append(
            " | ".join(
                [
                    f"| {name} · {m['variant']}",
                    fmt_pct(m["accuracy"]),
                    fmt_pct(m["macro_f1"]),
                    fmt_ece(m["ece"]),
                    fmt_pct(served["accuracy"]) if served else "-",
                    fmt_usd(m["cost_per_1k_usd"]),
                    f"{fmt_ms(latency.get('p50_ms'))} / {fmt_ms(latency.get('p95_ms'))}",
                    fmt_rate(m["throughput_rps"]),
                    fmt_mb(m["size_mb"]) + " |",
                ]
            )
        )
    return lines


def table_notes(results: dict[str, Any]) -> list[str]:
    hw = results["hardware"]
    prices = results["prices"]
    cpu = prices["cpu_instance"]
    return [
        f"Test set: {results['dataset']['test']:,} human-labeled messages, "
        f"{results['dataset']['labels']} intents. Students: mean ± std over "
        f"{results['seeds']} seeds; *served accuracy* is the exported artifact of the seed "
        "chosen on teacher-labeled validation data. Teacher latency is the logged API round "
        "trip; student latency is a single request on "
        f"{hw['cpu']} ({hw['threads'] or 'all'} threads). Student cost = "
        f"{cpu['instance']} at ${cpu['usd_per_hour']}/h (price {cpu['price_as_of']}) ÷ "
        f"measured batched throughput. Teacher prices as of {prices['teacher_as_of'] or 'n/a'}.",
    ]


def render_results_markdown(results: dict[str, Any]) -> str:
    lines = [f"# Benchmark results: {results['experiment']}", ""]
    if results.get("simulated"):
        lines += [
            "> **SIMULATED TEACHER.** These numbers come from the test-only simulated teacher",
            "> and say nothing about real models. Do not publish them.",
            "",
        ]
    lines += [f"Generated {results['generated_at']}.", "", f"**{results['headline']['text']}**", ""]
    lines += results_table(results)
    lines += ["", *table_notes(results), ""]

    lines += [
        "## Labeling budget",
        "",
        "| Student | Teacher-labeled examples | Labeling cost | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for r in results["learning_curve"]:
        lines.append(
            f"| {r['student']} | {r['examples']:,.0f} | {fmt_usd(r['labeling_cost_usd'])} "
            f"| {fmt_pct(r['accuracy'])} |"
        )
    lines += [
        "",
        "## Fallback routing",
        "",
        f"Send {results['best_student']} predictions below a confidence threshold to the "
        "teacher (single call):",
        "",
        "| Threshold | Share sent to teacher | Accuracy | Blended cost / 1k |",
        "|---:|---:|---:|---:|",
    ]
    for r in results["routing"]:
        lines.append(
            f"| {r['threshold']:.2f} | {fmt_pct(r['routed_share'])} | {fmt_pct(r['accuracy'])} "
            f"| {fmt_usd(r['cost_per_1k_usd'])} |"
        )
    lines += ["", "## Charts", ""]
    for name in ("pareto", "learning_curve"):
        if name in results.get("charts", {}):
            lines.append(f"![{name}]({results['charts'][name]})")
    lines += ["", "See [error_analysis.md](error_analysis.md) for the most confused intents.", ""]
    env = results["environment"]
    lines += [
        "## Environment",
        "",
        f"- Python {env['python']} on {env['platform']}",
        f"- CPU: {env['cpu']} ({env['cpu_count']} cores)",
        f"- Git revision: {env.get('git_revision') or 'n/a'}",
        "- Packages: " + ", ".join(f"{k} {v}" for k, v in sorted(env["packages"].items())),
        "",
    ]
    return "\n".join(lines)


def render_error_analysis(results: dict[str, Any]) -> str:
    ea = results["error_analysis"]
    lines = [
        f"# Error analysis: {ea['student']} on {results['experiment']}",
        "",
        f"{ea['errors']:,} errors on {ea['test_examples']:,} test messages. The most frequent "
        "confusions (human label -> predicted label):",
        "",
    ]
    for i, pair in enumerate(ea["pairs"], 1):
        lines.append(f"{i}. `{pair['gold']}` -> `{pair['predicted']}` ({pair['count']} times)")
        for text in pair["examples"]:
            lines.append(f'   - "{text}"')
    lines.append("")
    return "\n".join(lines)


def render_model_card(results: dict[str, Any], audit: dict[str, Any] | None = None) -> str:
    """Model card for the best student, filled from the benchmark (and audit) results."""
    best = next(m for m in results["models"] if m["name"] == results["best_student"])
    teacher = next(m for m in results["models"] if m["role"] == "teacher")
    served = best["served"]
    data = results["dataset"]
    latency = best.get("latency") or {}
    lines = [f"# Model card: {best['name']} ({results['experiment']})", ""]
    if results.get("simulated"):
        lines += [
            "> **SIMULATED TEACHER.** Generated from a pipeline test run; not a real model.",
            "",
        ]
    lines += [
        "## Model details",
        "",
        f"- **Type:** {best['kind']} student ({best.get('base_model') or 'no pre-trained base'})"
        f", served as {best['variant'].removeprefix('served: ')}, {fmt_mb(best['size_mb'])}.",
        f"- **Trained on:** labels from the teacher LLM `{results['teacher']['model']}` "
        f"(self-consistency samples: {results['teacher']['self_consistency']}); no human "
        "labels were used for training or model selection.",
        f"- **Selected by:** highest mean agreement with the teacher on a teacher-labeled "
        f"validation set; seed {served['seed']} of {results['seeds']}.",
        "- **Code license:** MIT. The base model and the source dataset keep their own licenses.",
        "",
        "## Intended use",
        "",
        f"Classifying short English messages into the {data['labels']} intents of "
        f"{data['name']}, as a cheap, fast replacement for calling the teacher LLM. Pair it "
        "with the fallback router (low-confidence predictions go to the teacher) where "
        "errors are costly, and keep a human in the loop for consequential decisions.",
        "",
        "**Out of scope:** other languages or domains, long documents, intents outside the "
        + (
            f"label set (they should land in `{data['oos_label']}`, but check that class's "
            "recall before relying on it)"
            if data.get("oos_label")
            else "label set (there is no out-of-scope class, so the model always picks one of "
            "its known intents)"
        )
        + ", and any use where an automated misroute could cause harm without review.",
        "",
        "## Evaluation",
        "",
        f"Human-labeled test set: {data['test']:,} messages, never used for training or selection.",
        "",
        "| | Accuracy | Macro-F1 | ECE | Cost / 1k | Latency p50 / p95 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {best['name']} (served) | {fmt_pct(served['accuracy'])} "
        f"| {fmt_pct(served['macro_f1'])} | {fmt_ece(served['ece'])} "
        f"| {fmt_usd(best['cost_per_1k_usd'])} | {fmt_ms(latency.get('p50_ms'))} / "
        f"{fmt_ms(latency.get('p95_ms'))} |",
        f"| {best['name']} (mean ± std over seeds) | {fmt_pct(best['accuracy'])} "
        f"| {fmt_pct(best['macro_f1'])} | {fmt_ece(best['ece'])} | | |",
        f"| teacher `{teacher['name']}` | {fmt_pct(teacher['accuracy'])} "
        f"| {fmt_pct(teacher['macro_f1'])} | {fmt_ece(teacher['ece'])} "
        f"| {fmt_usd(teacher['cost_per_1k_usd'])} | "
        f"{fmt_ms((teacher.get('latency') or {}).get('p50_ms'))} / "
        f"{fmt_ms((teacher.get('latency') or {}).get('p95_ms'))} |",
        "",
        *table_notes(results),
        "",
        "## Known failure modes",
        "",
    ]
    if audit is not None:
        pooled = audit["pooled"]
        worst = ", ".join(f"`{w['label']}`" for w in pooled["worst_classes"][:5])
        lines.append(
            f"- **Inherited teacher errors.** The teacher's labels agree with the hidden human "
            f"labels {fmt_pct(pooled['accuracy'])} of the time; its weakest intents are "
            f"{worst}. The student learns these mistakes."
        )
    pairs = results["error_analysis"]["pairs"][:5]
    if pairs:
        confusions = "; ".join(f"`{p['gold']}` -> `{p['predicted']}` ({p['count']})" for p in pairs)
        lines.append(
            f"- **Confusable intents.** Most frequent test errors: {confusions}. "
            "See error_analysis.md for examples."
        )
    lines += [
        f"- **Calibration.** Expected calibration error is {fmt_ece(served['ece'])}; check "
        "the routing table in results.md before choosing a fallback threshold.",
        "- **Distribution shift.** Accuracy is measured on the benchmark's test split; new "
        "products, phrasing or languages will lower it. Re-label a sample with the teacher "
        "periodically and compare.",
        "",
    ]
    return "\n".join(lines)


def render_readme_section(results: dict[str, Any], results_dir: str) -> str:
    """The block between the README markers. ``results_dir`` is relative to the README."""
    if results.get("simulated"):
        raise SimulatedResultsError("refusing to publish results from the simulated teacher")
    charts_ = results.get("charts", {})
    lines = [START, "", f"**{results['headline']['text']}**", ""]
    lines += results_table(results)
    lines += ["", *table_notes(results), ""]
    if "pareto" in charts_:
        dark = charts_.get("pareto-dark", charts_["pareto"])
        lines += [
            "<picture>",
            f'  <source media="(prefers-color-scheme: dark)" srcset="{results_dir}/{dark}">',
            f'  <img alt="Accuracy vs. cost per 1,000 predictions" '
            f'src="{results_dir}/{charts_["pareto"]}" width="720">',
            "</picture>",
            "",
        ]
    lines += [
        f"Full results, labeling-budget curve, fallback routing and error analysis: "
        f"[{results_dir}/results.md]({results_dir}/results.md).",
        "",
        END,
    ]
    return "\n".join(lines)


def replace_section(readme: str, section: str) -> str:
    if START not in readme or END not in readme:
        raise ValueError(f"README has no {START} ... {END} block")
    before, rest = readme.split(START, 1)
    _, after = rest.split(END, 1)
    return before + section + after


def update_readme(
    readme_path: Path, results: dict[str, Any], results_dir: str, *, check: bool = False
) -> bool:
    """Write (or, with ``check``, verify) the README results block. Returns True if in sync."""
    readme = readme_path.read_text(encoding="utf-8")
    updated = replace_section(readme, render_readme_section(results, results_dir))
    if check:
        return updated == readme
    readme_path.write_text(updated, encoding="utf-8")
    return True
