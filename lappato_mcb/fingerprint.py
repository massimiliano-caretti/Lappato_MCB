"""Title fingerprint + Jaccard near-duplicate detection — stdlib only.

Used by LAPPATO_MCB to deduplicate harvested papers ACROSS sources.
The original dedup key was the source-native id (arXiv URL, OpenAlex id),
so a paper indexed by both arXiv and OpenAlex with different ids passed
through twice. Here we lift the dedup to the level of the title: two
titles whose normalised character-trigram sets have Jaccard >= TAU are
treated as the same paper.

The choice of trigram + Jaccard (vs. token-bag + cosine, or a learned
embedding) is dictated by LAPPATO_MCB's stdlib-only constraint and by a
practical observation: paper titles in arXiv vs OpenAlex differ mostly
by punctuation, capitalisation, sub-title separators ("X: Y" vs "X-Y"),
and trailing dataset-version annotations. Character trigrams are robust
to all of those without a vocabulary or a tokeniser.
"""
from __future__ import annotations

import re
import unicodedata

# Default Jaccard threshold for "same paper". Tuned on a small set of
# arXiv/OpenAlex cross-listings (Specter, SciBERT, LightGBM) where the
# arXiv title and the venue-published title differ in punctuation /
# capitalisation but share >= 80% of trigrams.
DEFAULT_JACCARD_TAU = 0.80

_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalise_title(raw: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace + strip accents.

    Cheap, deterministic, locale-independent. Empty input maps to "".
    """
    if not raw:
        return ""
    nfkd = unicodedata.normalize("NFKD", raw)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    lowered = ascii_only.lower()
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def trigrams(text: str) -> set[str]:
    """Set of character trigrams of the normalised text.

    Strings shorter than 3 characters are returned as a singleton set
    containing the (left-padded) string itself, so they remain
    comparable with longer titles via Jaccard.
    """
    norm = normalise_title(text)
    if len(norm) < 3:
        return {norm.ljust(3)} if norm else set()
    return {norm[i : i + 3] for i in range(len(norm) - 2)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two sets. Empty/empty -> 0.0."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a) + len(b) - inter
    return inter / union


class TitleDeduper:
    """Streaming near-duplicate detector for harvested paper titles.

    Holds an incrementally-built corpus of trigram fingerprints and,
    for each candidate title, returns True if a near-duplicate is
    already present (Jaccard >= tau). Add the candidate's fingerprint
    to the corpus when calling `seen_or_register`.

    The detector is O(N) per query in the worst case where N is the
    corpus size. For the scale of a single LAPPATO_MCB run (10^1-10^2
    papers) this is negligible. A sub-linear LSH index would be
    over-engineering for the prototype's scale.
    """

    def __init__(self, tau: float = DEFAULT_JACCARD_TAU):
        self._tau = float(tau)
        self._fingerprints: list[tuple[str, set[str]]] = []
        self._duplicates_blocked = 0
        self._unique_kept = 0

    def is_near_duplicate(self, title: str) -> tuple[bool, float, str]:
        """Return (is_dup, best_similarity, matched_title).

        Pure query — does not mutate the corpus. Use
        ``seen_or_register`` to combine query and registration.
        """
        fp = trigrams(title)
        if not fp:
            return (False, 0.0, "")
        best_sim = 0.0
        best_match = ""
        for ref_title, ref_fp in self._fingerprints:
            sim = jaccard(fp, ref_fp)
            if sim > best_sim:
                best_sim = sim
                best_match = ref_title
                if sim >= 1.0:
                    break
        return (best_sim >= self._tau, best_sim, best_match)

    def seen_or_register(self, title: str) -> tuple[bool, float, str]:
        """Query + register in one shot.

        If the title is a near-duplicate of something already in the
        corpus, returns (True, sim, matched_title) WITHOUT registering.
        Otherwise registers the title and returns (False, sim, "").
        """
        is_dup, sim, match = self.is_near_duplicate(title)
        if is_dup:
            self._duplicates_blocked += 1
            return (True, sim, match)
        fp = trigrams(title)
        if fp:
            self._fingerprints.append((title, fp))
            self._unique_kept += 1
        return (False, sim, "")

    @property
    def stats(self) -> dict[str, int | float]:
        seen_total = self._unique_kept + self._duplicates_blocked
        rate = (self._duplicates_blocked / seen_total) if seen_total else 0.0
        return {
            "unique_kept": self._unique_kept,
            "duplicates_blocked": self._duplicates_blocked,
            "dedup_rate": rate,
            "tau": self._tau,
        }


__all__ = [
    "DEFAULT_JACCARD_TAU",
    "normalise_title",
    "trigrams",
    "jaccard",
    "TitleDeduper",
]
