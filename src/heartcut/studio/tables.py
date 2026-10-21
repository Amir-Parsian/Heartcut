"""Read the spreadsheets people drop into the studio and check they can train a model.

Everything here works on a :class:`pandas.DataFrame` of strings, whatever the file
format, and every problem is reported in words meant for the person who uploaded the
file, not for a developer.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from heartcut.data.schema import LABEL, TEXT

FORMATS = {
    ".csv": "CSV",
    ".tsv": "TSV",
    ".tab": "TSV",
    ".txt": "CSV",
    ".jsonl": "JSON lines",
    ".ndjson": "JSON lines",
    ".json": "JSON",
    ".xlsx": "Excel",
    ".xlsm": "Excel",
}
MAX_ROWS = 200_000
MIN_ROWS = 20
MIN_PER_CATEGORY = 5
MAX_CATEGORIES = 300
JOIN = ". "

_TEXT_HINTS = re.compile(r"text|message|body|content|description|comment|query|question|subject")
_LABEL_HINTS = re.compile(r"label|category|class|intent|queue|topic|tag|type|team|department")


class TableError(ValueError):
    """A problem with an uploaded file, worded for the person who uploaded it."""


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded file into a frame of stripped strings (empty cells are NA)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".xls":
        raise TableError(
            "Old Excel files (.xls) can't be read. In Excel, choose File > Save As and pick "
            "'Excel Workbook (.xlsx)' or 'CSV', then drop the new file here."
        )
    if suffix not in FORMATS:
        raise TableError(
            f"Heartcut can't read '{suffix or filename}' files. Use CSV, Excel (.xlsx), TSV "
            "or JSON lines."
        )
    if not data.strip():
        raise TableError("This file is empty.")
    try:
        frame = _parse(data, suffix)
    except TableError:
        raise
    except ImportError as exc:
        raise TableError(
            "Reading Excel files needs one more package: pip install 'heartcut[studio]'. "
            "Or save the sheet as CSV and drop that instead."
        ) from exc
    except Exception as exc:
        raise TableError(
            f"This doesn't look like a valid {FORMATS[suffix]} file ({exc}). Check that it "
            "opens in a spreadsheet program, or save it as CSV and try again."
        ) from exc
    frame.columns = pd.Index(_column_names(list(frame.columns)))
    frame = frame.astype("string")
    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    frame = frame.replace("", pd.NA).dropna(how="all").reset_index(drop=True)
    if frame.empty or not len(frame.columns):
        raise TableError("No rows found in this file.")
    if len(frame) > MAX_ROWS:
        raise TableError(
            f"This file has {len(frame):,} rows; the studio takes up to {MAX_ROWS:,}. "
            "A few thousand sorted examples are usually plenty."
        )
    return frame


def _parse(data: bytes, suffix: str) -> pd.DataFrame:
    if suffix in (".xlsx", ".xlsm"):
        sheet: pd.DataFrame = pd.read_excel(io.BytesIO(data), sheet_name=0, dtype=str)
        return sheet
    text = _decode(data)
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(io.StringIO(text), lines=True, dtype=False)
    if suffix == ".json":
        try:
            return pd.read_json(io.StringIO(text), dtype=False)
        except ValueError:
            return pd.read_json(io.StringIO(text), lines=True, dtype=False)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(io.StringIO(text), sep="\t", dtype=str, keep_default_na=False)
    return pd.read_csv(io.StringIO(text), sep=_delimiter(text), dtype=str, keep_default_na=False)


def _delimiter(text: str) -> str:
    """Comma, semicolon (spreadsheets in many locales), tab or pipe; comma if unclear.

    A header without any of them is a single column, such as a list of messages, one per
    line, whose unquoted commas belong to the text.
    """
    header = text.lstrip().split("\n", 1)[0]
    if not any(d in header for d in ",;\t|"):
        return "\x1f"  # a separator no text contains
    try:
        return csv.Sniffer().sniff(text[:20_000], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _column_names(columns: Sequence[Any]) -> list[str]:
    """Unique, non-empty column names (spreadsheets often have blank or repeated headers)."""
    names: list[str] = []
    for i, column in enumerate(columns):
        name = str(column).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"Column {i + 1}"
        base, n = name, 2
        while name in names:
            name, n = f"{base} ({n})", n + 1
        names.append(name)
    return names


# --------------------------------------------------------------------------- inspection


def describe_columns(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for name in frame.columns:
        values = frame[name].dropna()
        out.append(
            {
                "name": name,
                "filled": len(values),
                "unique": int(values.nunique()),
                "avg_length": float(values.str.len().mean()) if len(values) else 0.0,
                "sample": str(values.iloc[0])[:80] if len(values) else "",
            }
        )
    return out


def preview(frame: pd.DataFrame, rows: int = 8) -> list[dict[str, str]]:
    head = frame.head(rows).astype(object).where(frame.head(rows).notna(), "")
    return [{str(k): str(v)[:200] for k, v in row.items()} for row in head.to_dict("records")]


def guess_columns(frame: pd.DataFrame) -> dict[str, Any]:
    """A first guess at the message and category columns, for the person to correct."""
    columns = describe_columns(frame)
    n = len(frame)

    def label_score(c: dict[str, Any]) -> float:
        hint = 1.0 if _LABEL_HINTS.search(c["name"].lower()) else 0.0
        return hint - float(c["unique"]) / max(n, 1) - float(c["avg_length"]) / 1000

    candidates = [c for c in columns if 2 <= c["unique"] <= min(MAX_CATEGORIES, max(n // 2, 2))]
    label = max(candidates, key=label_score)["name"] if candidates else None

    def text_score(c: dict[str, Any]) -> float:
        hint = 20.0 if _TEXT_HINTS.search(c["name"].lower()) else 0.0
        return float(c["avg_length"]) + hint

    texts = [c for c in columns if c["name"] != label]
    text = max(texts, key=text_score)["name"] if texts else None
    return {"text_columns": [text] if text else [], "label_column": label}


def join_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Each row's non-empty values in ``columns``, joined into one message."""
    rows = frame[list(columns)].itertuples(index=False, name=None)
    return [JOIN.join(v for v in row if isinstance(v, str)) for row in rows]


def build_examples(
    frame: pd.DataFrame, text_columns: Sequence[str], label_column: str
) -> pd.DataFrame:
    """``text``/``label`` rows: the message columns joined, rows missing either dropped."""
    unknown = [c for c in [*text_columns, label_column] if c not in frame.columns]
    if unknown:
        raise TableError(f"The file has no column named {', '.join(repr(c) for c in unknown)}.")
    examples = pd.DataFrame(
        {TEXT: join_columns(frame, text_columns), LABEL: frame[label_column].to_list()}
    )
    examples = examples.replace("", pd.NA).dropna().reset_index(drop=True)
    return examples.astype(str)


def profile(
    frame: pd.DataFrame, text_columns: Sequence[str], label_column: str | None
) -> dict[str, Any]:
    """Can these columns train a model? Counts, per-category sizes, errors and warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    out: dict[str, Any] = {
        "rows": len(frame),
        "usable": 0,
        "skipped": 0,
        "categories": [],
        "errors": errors,
        "warnings": warnings,
        "ready": False,
    }
    if not text_columns:
        errors.append("Drag at least one column onto Message: the text the model should read.")
    if not label_column:
        errors.append("Drag one column onto Category: the answer the model should give.")
    if label_column and label_column in text_columns:
        errors.append(
            f"'{label_column}' can't be both the message and the category: the model would "
            "just copy the answer."
        )
    unknown = [c for c in [*text_columns, label_column] if c and c not in frame.columns]
    if unknown:
        errors.append(f"The file has no column named {', '.join(repr(c) for c in unknown)}.")
    if errors or label_column is None:
        return out

    examples = build_examples(frame, text_columns, label_column)
    counts = examples[LABEL].value_counts()
    out.update(
        usable=len(examples),
        skipped=len(frame) - len(examples),
        categories=[{"name": str(k), "count": int(v)} for k, v in counts.items()],
    )
    if out["skipped"]:
        warnings.append(
            f"{out['skipped']:,} rows are skipped because their message or category is empty."
        )
    if len(examples) < MIN_ROWS:
        errors.append(
            f"Only {len(examples)} rows have both a message and a category. Heartcut needs at "
            f"least {MIN_ROWS}, and a few hundred give much better results."
        )
    if len(counts) < 2:
        errors.append(
            "Every row has the same category, so there is nothing to learn. Pick the column "
            "that sorts your messages into groups."
        )
    elif len(counts) > MAX_CATEGORIES or (
        len(examples) >= MIN_ROWS and len(counts) > len(examples) / 2
    ):
        errors.append(
            f"'{label_column}' has {len(counts):,} different values in {len(examples):,} rows, "
            "so almost every row is its own category. Pick a column with a short list of "
            "repeating values, such as a team, a topic or a type."
        )
    small = [str(k) for k, v in counts.items() if v < MIN_PER_CATEGORY]
    if small and not errors:
        listed = ", ".join(small[:5]) + (f" and {len(small) - 5} more" if len(small) > 5 else "")
        warnings.append(
            f"{len(small)} {'category has' if len(small) == 1 else 'categories have'} fewer than "
            f"{MIN_PER_CATEGORY} examples ({listed}). The model will rarely get "
            "them right, and they can't be tested fairly. Add examples or merge them."
        )
    out["ready"] = not errors
    return out
