"""One-shot literature search across the scholarly sources LAPPATO_MCB knows.

The daemon only ever queries sources on behalf of an active weakness.
This module exposes the same per-source adapters as a plain function,
for callers that want an ad-hoc query (the MCP ``search_literature``
tool, notebooks, scripts) without building a manifest.

Online mode issues one request per source with no retry and no
cross-source dedup: a failing source is reported in ``errors`` and
does not prevent the others from answering. Offline mode searches the
persistent corpus written by previous daemon runs (see
:mod:`lappato_mcb.cache`) and never touches the network.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..cache import CorpusCache
from ..core import _SOURCE_FN, GENERAL_SOURCES, SOURCES

MAX_RESULTS_PER_SOURCE = 25


def search_literature(
    query: str,
    *,
    sources: Sequence[str] = GENERAL_SOURCES,
    max_results: int = 5,
    offline: bool = False,
    corpus_path: Path | None = None,
) -> dict:
    """Search ``sources`` for ``query`` and return the hits grouped in one list.

    Parameters:
        query        : free-text query, sent verbatim to every source
        sources      : subset of :data:`lappato_mcb.SOURCES`
        max_results  : hits per source, 1..``MAX_RESULTS_PER_SOURCE``
        offline      : search the local corpus instead of the network
        corpus_path  : corpus JSONL to read in offline mode (required then)

    Returns a dict with ``query``, ``sources``, ``offline``, ``hits``
    (each hit carries a ``source`` key) and ``errors`` (source -> message).

    Raises :class:`ValueError` on an empty query, an unknown source, an
    out-of-range ``max_results``, or offline mode without ``corpus_path``.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query must be a non-empty string")
    sources = list(dict.fromkeys(sources))
    unknown = [s for s in sources if s not in SOURCES]
    if unknown or not sources:
        raise ValueError(
            f"sources must be a non-empty subset of {list(SOURCES)}, got {sources!r}"
        )
    if not 1 <= max_results <= MAX_RESULTS_PER_SOURCE:
        raise ValueError(
            f"max_results must be in 1..{MAX_RESULTS_PER_SOURCE}, got {max_results}"
        )
    if offline and corpus_path is None:
        raise ValueError("offline search requires corpus_path")

    hits: list[dict] = []
    errors: dict[str, str] = {}
    if offline:
        cache = CorpusCache(Path(corpus_path))
        ranked = cache.search(query, k=len(cache))
        for source in sources:
            rows = [r for r in ranked if r.get("source") == source]
            hits.extend(dict(r) for r in rows[:max_results])
    else:
        for source in sources:
            try:
                rows = _SOURCE_FN[source](query, max_results) or []
            except Exception as exc:  # one bad source must not sink the rest
                errors[source] = f"{type(exc).__name__}: {exc}"
                continue
            hits.extend({**r, "source": source} for r in rows[:max_results])

    return {
        "query": query,
        "sources": sources,
        "offline": bool(offline),
        "hits": hits,
        "errors": errors,
    }
