"""Offline replay-from-corpus mode — air-gapped harvesting.

LAPPATO_MCB's online path issues HTTP queries to arXiv and OpenAlex.
In on-premise clinical environments and on aircraft, network access
is precluded. This module records every paper LAPPATO_MCB harvests
across runs into a single JSONL corpus and lets a subsequent run
search that corpus locally instead of hitting the wire.

The corpus format is one JSON object per line, the same shape as the
LAPPATO_MCB's existing ``checkpoints/*_lappato_mcb_papers.jsonl`` sidecar
but accumulated across runs and across pipelines. Search is a
deterministic in-process scan: tokenise the query, score each row by
the count of distinct query tokens that appear in either the title or
the abstract, return the top-K. No vector store, no external service.

The store is intentionally append-only: deletions and updates are out
of scope (LAPPATO_MCB treats abstracts as immutable). A future extension
can add embedding-based ranking.
"""
from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterable
from pathlib import Path

from .fingerprint import normalise_title

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")


class CorpusCache:
    """Append-only local corpus of harvested papers + offline search.

    Two modes — ``record(paper)`` adds a row during an online run;
    ``search(query, k)`` returns the top-K rows by token-overlap score
    during an offline run. Both modes are safe to mix in a single run
    (online harvesting always records into the corpus).
    """

    def __init__(self, corpus_path: Path):
        self._path = Path(corpus_path)
        self._lock = threading.Lock()
        self._rows_cache: list[dict] | None = None  # lazy load

    # ── write path ────────────────────────────────────────────────
    def record(self, paper: dict) -> None:
        """Append one paper row to the corpus.

        ``paper`` should carry at least ``id``, ``title``, ``abstract``;
        any extra keys are preserved verbatim. Cross-run dedup against
        the corpus is intentionally NOT done here — the corpus is the
        raw harvest history; the consuming LAPPATO_MCB does dedup.
        """
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(paper, ensure_ascii=False) + "\n")
                fh.flush()
            self._rows_cache = None  # invalidate lazy cache

    # ── read path ─────────────────────────────────────────────────
    def _load(self) -> list[dict]:
        if self._rows_cache is not None:
            return self._rows_cache
        rows: list[dict] = []
        if self._path.exists():
            with self._path.open(encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        rows.append(json.loads(s))
                    except json.JSONDecodeError:
                        continue
        self._rows_cache = rows
        return rows

    def __len__(self) -> int:
        return len(self._load())

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return up to k rows ranked by query-token overlap.

        Score = |distinct query tokens present in (title + abstract)|.
        Ties broken by recency (year desc) when comparable.
        """
        q_tokens = _tokens(query)
        if not q_tokens:
            return []
        scored: list[tuple[int, int, dict]] = []
        for row in self._load():
            haystack = (row.get("title", "") or "") + " " \
                       + (row.get("abstract", "") or "")
            h_tokens = _tokens(haystack)
            score = len(q_tokens & h_tokens)
            if score == 0:
                continue
            try:
                year = int(str(row.get("year", "0")) or "0")
            except ValueError:
                year = 0
            scored.append((score, year, row))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [r for _, _, r in scored[:k]]

    @property
    def stats(self) -> dict:
        rows = self._load()
        sources = {}
        for r in rows:
            s = r.get("source", "?")
            sources[s] = sources.get(s, 0) + 1
        return {
            "corpus_path": str(self._path),
            "n_rows": len(rows),
            "by_source": sources,
        }


def _tokens(text: str) -> set[str]:
    """Lowercase distinct tokens >= 3 chars. Mirrors arXiv-style filtering."""
    if not text:
        return set()
    norm = normalise_title(text)
    return {m.group(0) for m in _TOKEN_RE.finditer(norm)}


def merge_jsonl_into_corpus(jsonl_paths: Iterable[Path], corpus_path: Path) -> int:
    """Bootstrap helper: merge per-run LAPPATO_MCB JSONL sidecars into the corpus.

    Idempotent on identical inputs (the corpus is append-only; callers
    can dedupe upstream if needed). Returns the number of rows added.
    """
    cache = CorpusCache(corpus_path)
    added = 0
    for p in jsonl_paths:
        p = Path(p)
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache.record(row)
                added += 1
    return added


__all__ = ["CorpusCache", "merge_jsonl_into_corpus"]
