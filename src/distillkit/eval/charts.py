"""README charts: accuracy-vs-cost Pareto chart and the labeling-budget learning curve.

Both are rendered twice, for light and dark backgrounds, so the README can pick the
right one with ``<picture>``. Colors follow a fixed, colorblind-validated categorical
order (students take slots in config order; the teacher is drawn in neutral ink), every
point and line is direct-labeled, and there is a single y-axis per chart: labeling
cost is shown on a *secondary x-axis* (dollars are a fixed multiple of the example
count), never as a second y-scale.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
)
THEMES = (LIGHT, DARK)


def _figure(theme: Theme) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=160)
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.axis)
    ax.tick_params(colors=theme.muted, labelcolor=theme.ink_secondary, labelsize=9)
    ax.grid(True, which="major", color=theme.grid, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _titles(ax: Axes, theme: Theme, title: str, subtitle: str, y: float = 1.02) -> None:
    """Left-aligned title with a subtitle line under it, both above the plot (and above a
    top axis when ``y`` is raised)."""
    ax.text(
        0, y, subtitle, transform=ax.transAxes, color=theme.ink_secondary, fontsize=9, va="bottom"
    )
    ax.text(
        0,
        y + 0.055,
        title,
        transform=ax.transAxes,
        color=theme.ink,
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )


def _percent(value: float, _: int) -> str:
    return f"{100 * value:.0f}%"


def _dollars(value: float, _: int) -> str:
    if value >= 1:
        return f"${value:,.0f}"
    if value <= 0:
        return "$0"
    decimals = max(2, -math.floor(math.log10(value)))
    return f"${value:.{decimals}f}"


def student_colors(names: Sequence[str], theme: Theme) -> dict[str, str]:
    """Fixed slot per student, in config order, so filters never repaint survivors."""
    return {name: theme.series[i % len(theme.series)] for i, name in enumerate(names)}


def pareto_frontier(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points not beaten on both cost (lower) and accuracy (higher), sorted by cost."""
    frontier: list[tuple[float, float]] = []
    for cost, acc in sorted(points, key=lambda p: (p[0], -p[1])):
        if not frontier or acc > frontier[-1][1]:
            frontier.append((cost, acc))
    return frontier


def plot_pareto(results: dict[str, Any], path: Path, theme: Theme = LIGHT) -> Path:
    models = [m for m in results["models"] if m.get("cost_per_1k_usd")]
    students = [m["name"] for m in results["models"] if m["role"] == "student"]
    colors = student_colors(students, theme)
    fig, ax = _figure(theme)
    ax.set_xscale("log")

    frontier = pareto_frontier([(m["cost_per_1k_usd"], m["accuracy"]["mean"]) for m in models])
    if len(frontier) > 1:
        ax.plot(
            [c for c, _ in frontier],
            [a for _, a in frontier],
            color=theme.muted,
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            zorder=1,
            label="Pareto frontier",
        )

    for m in models:
        is_teacher = m["role"] == "teacher"
        color = theme.ink if is_teacher else colors[m["name"]]
        std = m["accuracy"].get("std") or 0.0
        # Teacher variants share the neutral ink; the voted variant is drawn hollow.
        hollow = is_teacher and m["variant"] != "single call"
        ax.errorbar(
            m["cost_per_1k_usd"],
            m["accuracy"]["mean"],
            yerr=std if std else None,
            fmt="D" if is_teacher else "o",
            markersize=8,
            color=color,
            markerfacecolor=theme.surface if hollow else color,
            markeredgecolor=color if hollow else theme.surface,
            markeredgewidth=1.5,
            elinewidth=1.2,
            capsize=0,
            zorder=3,
            label=m["label"],
        )
        ax.annotate(
            m["label"],
            (m["cost_per_1k_usd"], m["accuracy"]["mean"]),
            xytext=(8, 6),
            textcoords="offset points",
            color=theme.ink_secondary,
            fontsize=8.5,
        )

    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.xaxis.set_major_formatter(FuncFormatter(_dollars))
    ax.set_xlabel("Cost per 1,000 predictions (USD, log scale)", color=theme.ink_secondary)
    ax.set_ylabel("Accuracy on the human-labeled test set", color=theme.ink_secondary)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo / 1.5, hi * 3)  # room for the direct labels on the right
    legend = ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(theme.ink_secondary)
    subtitle = (
        f"{results['experiment']} · students: mean ± std over {results['seeds']} seeds · "
        f"{results['hardware']['cpu']}"
    )
    _titles(ax, theme, "Accuracy vs. cost per 1,000 predictions", subtitle)
    return _save(fig, path)


def plot_learning_curve(results: dict[str, Any], path: Path, theme: Theme = LIGHT) -> Path:
    curve = results["learning_curve"]
    students = [m["name"] for m in results["models"] if m["role"] == "student"]
    colors = student_colors(students, theme)
    fig, ax = _figure(theme)
    ax.set_xscale("log")

    for name in students:
        rows = sorted((r for r in curve if r["student"] == name), key=lambda r: r["examples"])
        if not rows:
            continue
        x = [r["examples"] for r in rows]
        y = [r["accuracy"]["mean"] for r in rows]
        sd = [r["accuracy"]["std"] or 0.0 for r in rows]
        ax.fill_between(
            x,
            [a - s for a, s in zip(y, sd, strict=True)],
            [a + s for a, s in zip(y, sd, strict=True)],
            color=colors[name],
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            x,
            y,
            color=colors[name],
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=theme.surface,
            markeredgewidth=1.5,
            label=name,
            zorder=3,
        )
        ax.annotate(
            name,
            (x[-1], y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=theme.ink_secondary,
            fontsize=8.5,
        )

    teacher = next((m for m in results["models"] if m["role"] == "teacher"), None)
    if teacher is not None:
        ax.axhline(
            teacher["accuracy"]["mean"],
            color=theme.muted,
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            zorder=1,
            label=teacher["label"],
        )
        ax.annotate(
            teacher["label"],
            (ax.get_xlim()[0], teacher["accuracy"]["mean"]),
            xytext=(4, -4),
            textcoords="offset points",
            va="top",
            color=theme.ink_secondary,
            fontsize=8.5,
        )

    # Ticks at the budgets actually run, written as plain numbers.
    budgets = sorted({r["examples"] for r in curve})
    ax.xaxis.set_major_locator(FixedLocator(budgets))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_xlabel("Teacher-labeled training examples (log scale)", color=theme.ink_secondary)
    ax.set_ylabel("Accuracy on the human-labeled test set", color=theme.ink_secondary)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 2.2)

    per_example = results.get("labeling_cost_per_example_usd")
    if per_example:
        top = ax.secondary_xaxis(
            "top", functions=(lambda n: n * per_example, lambda d: d / per_example)
        )
        top.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        top.xaxis.set_major_formatter(FuncFormatter(_dollars))
        top.xaxis.set_minor_formatter(NullFormatter())
        top.tick_params(colors=theme.muted, labelcolor=theme.ink_secondary, labelsize=9)
        top.spines["top"].set_color(theme.axis)
        top.set_xlabel("Teacher labeling cost (USD)", color=theme.ink_secondary, fontsize=9)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(theme.ink_secondary)
    subtitle = f"{results['experiment']} · mean ± std over {results['seeds']} seeds"
    _titles(
        ax,
        theme,
        "How accuracy scales with labeling spend",
        subtitle,
        y=1.13 if per_example else 1.02,
    )
    return _save(fig, path)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def render_all(results: dict[str, Any], directory: Path) -> dict[str, str]:
    """Render light and dark versions of both charts; return file names by key."""
    files: dict[str, str] = {}
    for theme in THEMES:
        suffix = "" if theme is LIGHT else "-dark"
        files[f"pareto{suffix}"] = plot_pareto(
            results, directory / f"pareto{suffix}.png", theme
        ).name
        if results.get("learning_curve"):
            files[f"learning_curve{suffix}"] = plot_learning_curve(
                results, directory / f"learning_curve{suffix}.png", theme
            ).name
    return files
