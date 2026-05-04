"""LAPPATO_MCB self-instrumentation — quantitative metrics on its own behaviour.

This module records one CSV row per cycle with low-level metrics
LAPPATO_MCB can observe about its own work:

  - cycle, ts_started_iso, wall_seconds
  - n_active_weaknesses, n_queries_executed
  - n_arxiv_hits_raw, n_openalex_hits_raw, n_crossref_hits_raw
    n_joss_hits_raw
                                                   (hits before any dedup)
  - n_arxiv_errors, n_openalex_errors, n_crossref_errors,
    n_joss_errors                                  (HTTP failures after all retries)
  - n_id_dedup_blocked, n_fp_dedup_blocked         (separated dedup paths)
  - n_papers_kept                                   (rows appended to JSONL)
  - first_paper_seconds                             (time-to-first-paper)

The CSV is written to ``checkpoints/<run_tag>_lappato_mcb_meta.csv`` and
consumed by ``analysis.py`` for the report tables and plots. This is
purely observational instrumentation — no behaviour change in the
harvest path.

Backwards compatibility note: rows written by older LAPPATO_MCB versions
without the ``n_<source>_errors`` columns parse cleanly because
``analysis.py`` defaults missing columns to ``None``. Likewise, header
upgrades are append-only — never remove or rename existing columns.
"""
from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Header columns; new sources require a new ``n_<src>_hits_raw`` entry
# here AND a matching attribute on CycleMetrics below. Backwards-
# compatible: a CSV missing a column is treated as zero by analysis.py.
_HEADER = [
    "cycle", "ts_started_iso", "wall_seconds",
    "n_active_weaknesses", "n_queries_executed",
    "n_arxiv_hits_raw", "n_openalex_hits_raw", "n_crossref_hits_raw",
    "n_joss_hits_raw",
    "n_arxiv_errors", "n_openalex_errors", "n_crossref_errors",
    "n_joss_errors",
    "n_id_dedup_blocked", "n_fp_dedup_blocked",
    "n_papers_kept", "first_paper_seconds",
]


@dataclass
class CycleMetrics:
    """Per-cycle counters LAPPATO_MCB increments while harvesting."""
    cycle: int
    ts_started_iso: str
    _t0: float = field(default_factory=time.monotonic, repr=False)
    n_active_weaknesses: int = 0
    n_queries_executed: int = 0
    n_arxiv_hits_raw: int = 0
    n_openalex_hits_raw: int = 0
    n_crossref_hits_raw: int = 0
    n_joss_hits_raw: int = 0
    n_arxiv_errors: int = 0
    n_openalex_errors: int = 0
    n_crossref_errors: int = 0
    n_joss_errors: int = 0
    n_id_dedup_blocked: int = 0
    n_fp_dedup_blocked: int = 0
    n_papers_kept: int = 0
    first_paper_seconds: float | None = None

    def add_source_hits(self, source: str, n: int) -> None:
        """Generic increment helper — keeps core.py source-name agnostic.

        Maps the canonical source name to the right counter attribute.
        Unknown source names are silently ignored (the harvest loop
        already guards against this, but the helper is defensive).
        """
        attr = {
            "arXiv": "n_arxiv_hits_raw",
            "OpenAlex": "n_openalex_hits_raw",
            "Crossref": "n_crossref_hits_raw",
            "JOSS": "n_joss_hits_raw",
        }.get(source)
        if attr is None:
            return
        setattr(self, attr, getattr(self, attr) + int(n))

    def add_source_error(self, source: str, n: int = 1) -> None:
        """Increment the error counter for ``source`` after all retries failed.

        Mirrors ``add_source_hits``. Unknown source names are ignored to
        keep the meta-log schema stable when a manifest opts into a
        narrower source pipeline.
        """
        attr = {
            "arXiv": "n_arxiv_errors",
            "OpenAlex": "n_openalex_errors",
            "Crossref": "n_crossref_errors",
            "JOSS": "n_joss_errors",
        }.get(source)
        if attr is None:
            return
        setattr(self, attr, getattr(self, attr) + int(n))

    def mark_first_paper(self) -> None:
        """Record latency to first paper kept this cycle (idempotent)."""
        if self.first_paper_seconds is None:
            self.first_paper_seconds = round(time.monotonic() - self._t0, 3)

    def finalise(self) -> dict:
        """Freeze the row that will be appended to the CSV."""
        wall = round(time.monotonic() - self._t0, 3)
        out = {k: getattr(self, k) for k in (
            "cycle", "ts_started_iso",
            "n_active_weaknesses", "n_queries_executed",
            "n_arxiv_hits_raw", "n_openalex_hits_raw", "n_crossref_hits_raw",
            "n_joss_hits_raw",
            "n_arxiv_errors", "n_openalex_errors", "n_crossref_errors",
            "n_joss_errors",
            "n_id_dedup_blocked", "n_fp_dedup_blocked",
            "n_papers_kept",
        )}
        out["wall_seconds"] = wall
        out["first_paper_seconds"] = (
            self.first_paper_seconds if self.first_paper_seconds is not None else ""
        )
        return out


class MetaLogWriter:
    """Append-only writer for LAPPATO_MCB's per-cycle meta-log CSV.

    Thread-safe (a lock serialises writes from the daemon thread and
    any future shutdown hook). Header is written exactly once on the
    first append; subsequent runs that target the same file append
    rows under the existing header.
    """

    def __init__(self, csv_path: Path):
        self._path = Path(csv_path)
        self._lock = threading.Lock()

    def append(self, row: dict) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self._path.exists() or self._path.stat().st_size == 0
            with self._path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=_HEADER)
                if new_file:
                    w.writeheader()
                w.writerow({k: row.get(k, "") for k in _HEADER})
                fh.flush()


__all__ = ["CycleMetrics", "MetaLogWriter"]
