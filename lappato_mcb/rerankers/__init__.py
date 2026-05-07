"""Optional, stdlib-only Reranker implementations.

This sub-package is INSTALLED with LAPPATO_MCB but never imported by
the daemon core. Users opt in by importing a reranker explicitly and
passing it to ``LAPPATO_MCB(..., reranker=...)``.

Available baselines (all stdlib-only):

  - :class:`lappato_mcb.rerankers.trigram.TrigramJaccardReranker` —
    character-trigram Jaccard similarity between query and (title +
    abstract). Reuses ``fingerprint.trigrams`` + ``fingerprint.jaccard``.
    Captures morphological variants (e.g. *calibration* vs
    *calibrating*) but not true synonyms (e.g. *vehicle* vs *car*).

A semantic-embedding reranker is planned for a separate
``lappato_mcb[embed]`` extras package (Phase 2) — *if* the
Phase 0.5 benchmark in this sub-package shows that a reranker
delivers measurable uplift on the bundled gold sets.
"""
