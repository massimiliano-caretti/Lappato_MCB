"""Stdlib-only trigram-Jaccard reranker.

This is the baseline reranker we use to **decide whether shipping a
real embedding reranker is worth it** (Phase 0.5 micro-benchmark).
It is intentionally weak: it captures morphological variants
(*calibration* / *calibrated* / *calibrating* share trigrams) but
**does not** capture true synonyms (*vehicle* vs *car*).

If this baseline already moves the needle on cases where token
overlap fails, a real semantic embedding model (Phase 2) will
likely move it further. If it doesn't, embedding-based reranking
deserves stronger evidence before we commit to its dependency
footprint.

Reuses ``fingerprint.trigrams`` + ``fingerprint.jaccard`` so there
is no duplicated logic and the determinism contract of the dedup
layer extends to the reranking layer.
"""
from __future__ import annotations

from ..fingerprint import jaccard, normalise_title, trigrams


class TrigramJaccardReranker:
    """Reranker scoring papers by char-trigram Jaccard vs the query.

    The score is the Jaccard similarity between
    ``trigrams(query)`` and ``trigrams(title + " " + abstract)``,
    in ``[0.0, 1.0]``.

    Determinism: identical to the existing
    :func:`lappato_mcb.fingerprint.jaccard` used by the cross-source
    dedup layer.

    Limitations (stated honestly so the user can decide whether
    Phase 2 is needed):

      - Captures **morphological variants** (e.g.
        *calibration* / *calibrated*) because they share trigrams.
      - Does **not** capture **true synonyms** (e.g. *vehicle* vs
        *car*) because they share no trigrams.
      - Does not understand **negation, scope, or polarity**
        (e.g. *no calibration* vs *good calibration* score similarly).

    Construction is free; it does not load any model. Per-call cost
    is O(|query| + Σ|paper blob|) trigrams computed once per call.
    """

    model_name = "trigram-jaccard"
    model_version = "1.0"
    model_sha = "stdlib-builtin"  # no external weights to hash

    def __init__(self, *, lowercase: bool = True) -> None:
        # ``lowercase`` is the default; exposed for users who want to
        # disable the case-folding from ``normalise_title``. The
        # underlying ``normalise_title`` already lowercases, so this
        # flag is mostly a future-proof knob.
        self._lowercase = bool(lowercase)

    def score(self, query: str, hits: list[dict]) -> list[float]:
        if not query or not hits:
            return [0.0] * len(hits)
        q_tri = trigrams(query)
        if not q_tri:
            return [0.0] * len(hits)
        out: list[float] = []
        for h in hits:
            blob = (h.get("title") or "") + " " + (h.get("abstract") or "")
            blob_norm = normalise_title(blob) if self._lowercase else blob
            blob_tri = trigrams(blob_norm)
            out.append(jaccard(q_tri, blob_tri))
        return out


__all__ = ["TrigramJaccardReranker"]
