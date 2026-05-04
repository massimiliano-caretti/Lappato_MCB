"""Small CSV helpers shared by domain manifests.

The helpers stay stdlib-only and intentionally conservative: malformed
or missing cells are ignored, and empty inputs return neutral values so
evidence checks fail closed rather than firing on bad files.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def f(row: dict, *names: str, default: float | None = None) -> float | None:
    for name in names:
        raw = row.get(name)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return default


def values(path: Path, *names: str) -> list[float]:
    out: list[float] = []
    for row in rows(path):
        val = f(row, *names)
        if val is not None:
            out.append(val)
    return out


def mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def max_value(path: Path, *names: str) -> float:
    vals = values(path, *names)
    return max(vals) if vals else 0.0


def min_value(path: Path, *names: str) -> float:
    vals = values(path, *names)
    return min(vals) if vals else 0.0


def spread(path: Path, *names: str) -> float:
    vals = values(path, *names)
    return (max(vals) - min(vals)) if vals else 0.0


def first_last_ratio(path: Path, *names: str) -> float:
    vals = values(path, *names)
    if len(vals) < 2 or vals[0] == 0:
        return 0.0
    return vals[-1] / vals[0]


def count_rows(path: Path) -> int:
    return len(rows(path))


def summary(path: Path, metric: str, value: float) -> dict:
    return {"rows": count_rows(path), metric: round(value, 4)}


__all__ = [
    "count_rows",
    "f",
    "first_last_ratio",
    "max_value",
    "mean",
    "min_value",
    "rows",
    "spread",
    "summary",
    "values",
]
