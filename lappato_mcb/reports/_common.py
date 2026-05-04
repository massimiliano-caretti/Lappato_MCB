"""Helpers shared across report packs.

Centralises the few primitives every domain needs: CSV reading,
numeric coercion, sample-stat aggregation with explicit N. Keeping
these in one place avoids divergent definitions of "mean ± sd" across
domains and keeps the rigour audit trail short.
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any


def read_csv_dicts(path: Path) -> list[dict]:
    """Read a CSV into a list of dicts; empty list if absent or unreadable."""
    if not path.exists():
        return []
    try:
        with path.open() as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def coerce_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def sample_stats(xs: list[float]) -> tuple[float, float, int]:
    """Return (mean, sample_sd, N) — sample_sd uses Bessel's correction.

    Conventions:
      - empty input  -> (NaN, NaN, 0)
      - single value -> (value, 0.0, 1)   (sd undefined; reported as 0.0)
      - N >= 2       -> (mean, stdev with ddof=1, N)
    """
    n = len(xs)
    if n == 0:
        return (math.nan, math.nan, 0)
    if n == 1:
        return (float(xs[0]), 0.0, 1)
    return (statistics.mean(xs), statistics.stdev(xs), n)


__all__ = ["read_csv_dicts", "coerce_float", "sample_stats"]
